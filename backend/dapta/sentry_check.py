"""Send one tagged test event and two linked transactions to Sentry, and report what happened to each.

Run where SENTRY_DSN is set (for example on the RunPod pod, after sourcing runtime.env):
    PYTHONPATH=backend python -m dapta.sentry_check --service controller
    PYTHONPATH=backend python -m dapta.sentry_check --service controller --confirm

Each item is reported at the strongest level actually observed:
  queued    the SDK built the envelope; nothing is known about delivery
  accepted  Sentry's ingest endpoint answered 2xx; it can still be dropped later
            (inbound filters, quota, spike protection), so this is not proof of storage
  stored    --confirm only: the event was read back through the Sentry API
            (needs SENTRY_AUTH_TOKEN with event:read and SENTRY_ORG; the project and API region
            come from SENTRY_DSN unless SENTRY_PROJECT / SENTRY_API_URL are set)
Exits 0 only when every item reached "accepted" (or "stored" with --confirm).
No document, prompt or model content is sent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Sequence

from .observability import continued_transaction, initialize_sentry, sentry_sdk, span, trace_headers

try:
    from sentry_sdk.transport import HttpTransport
except ImportError:  # pragma: no cover - sentry-sdk is optional until deployment
    HttpTransport = object  # type: ignore[assignment,misc]

POLL_INTERVAL = 3.0


class RecordingTransport(HttpTransport):  # type: ignore[misc,valid-type]
    """The SDK's own HTTP transport, additionally keeping the ingest response for each event id."""

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        self.queued: dict[str, str] = {}  # event id -> item type
        self.responses: dict[str, int | str] = {}  # event id -> HTTP status, or the network error
        self._sending: str | None = None

    def capture_envelope(self, envelope: Any) -> None:
        event_id = envelope.headers.get("event_id")
        kinds = [item.type for item in envelope.items if item.type in ("event", "transaction")]
        if event_id and kinds:
            self.queued[event_id] = kinds[0]
        super().capture_envelope(envelope)

    def _send_request(self, body: bytes, headers: dict[str, str], endpoint_type: Any = None,
                      envelope: Any = None) -> None:
        self._sending = envelope.headers.get("event_id") if envelope is not None else None
        try:
            if endpoint_type is None:
                super()._send_request(body, headers, envelope=envelope)
            else:
                super()._send_request(body, headers, endpoint_type=endpoint_type, envelope=envelope)
        except Exception as error:
            if self._sending:
                self.responses[self._sending] = f"network error ({type(error).__name__})"
            raise
        finally:
            self._sending = None

    def _request(self, method: str, endpoint_type: Any, body: Any, headers: Any) -> Any:
        response = super()._request(method, endpoint_type, body, headers)
        if self._sending:
            self.responses[self._sending] = response.status
        return response


def _ingest_outcome(transport: RecordingTransport, event_id: str) -> tuple[bool, str]:
    status = transport.responses.get(event_id)
    if status is None:
        return False, "queued only: no response from Sentry ingest (dropped before sending, or flush timed out)"
    if isinstance(status, str):
        return False, f"not delivered: {status}"
    if 200 <= status < 300:
        return True, f"accepted by Sentry ingest (HTTP {status}); not yet proof it was stored"
    if status == 429:
        return False, "rejected: rate limited by Sentry (HTTP 429)"
    return False, f"rejected by Sentry ingest (HTTP {status}); check the DSN"


def _confirm_stored(event_ids: Sequence[str], *, api_url: str, org: str, project: str, token: str,
                    timeout: float) -> dict[str, str]:
    """Poll the Sentry API until each event can be read back; returns event id -> outcome."""
    pending = list(event_ids)
    outcomes: dict[str, str] = {}
    deadline = time.monotonic() + timeout
    while pending:
        for event_id in list(pending):
            path = "/api/0/projects/{}/{}/events/{}/".format(
                urllib.parse.quote(org, safe=""), urllib.parse.quote(project, safe=""), event_id)
            request = urllib.request.Request(api_url.rstrip("/") + path,
                                             headers={"Authorization": f"Bearer {token}"})
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    if response.status == 200:
                        outcomes[event_id] = "stored: read back through the Sentry API"
                        pending.remove(event_id)
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    for remaining in pending:
                        outcomes[remaining] = (f"unconfirmed: Sentry API refused the token (HTTP {error.code}); "
                                               "it needs event:read on this project")
                    return outcomes
                if error.code != 404:
                    outcomes[event_id] = f"unconfirmed: Sentry API returned HTTP {error.code}"
                    pending.remove(event_id)
            except (urllib.error.URLError, TimeoutError) as error:
                outcomes[event_id] = f"unconfirmed: could not reach the Sentry API ({type(error).__name__})"
                pending.remove(event_id)
        if pending and time.monotonic() >= deadline:
            for event_id in pending:
                outcomes[event_id] = (f"unconfirmed: not found after {timeout:.0f}s in {org}/{project}; check "
                                      "SENTRY_ORG and that no inbound filter or quota dropped it")
            break
        if pending:
            time.sleep(POLL_INTERVAL)
    return outcomes


def _dsn_api_defaults(dsn: str) -> tuple[str, str]:
    """The project id and Sentry API base URL implied by a DSN (the API accepts ids or slugs)."""
    parsed = urllib.parse.urlsplit(dsn.strip())
    project = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    host = parsed.hostname or ""
    if host == "sentry.io" or host.endswith(".sentry.io"):
        api_url = "https://de.sentry.io" if ".de.sentry.io" in f".{host}" else "https://sentry.io"
    else:  # self-hosted, or a local stand-in
        api_url = f"{parsed.scheme}://{parsed.netloc.rsplit('@', 1)[-1]}" if host else "https://sentry.io"
    return project, api_url


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify that this process's Sentry events reach Sentry.")
    parser.add_argument("--service", default="controller", choices=("controller", "qwen-worker"))
    parser.add_argument("--confirm", action="store_true",
                        help="read the events back through the Sentry API (SENTRY_AUTH_TOKEN, SENTRY_ORG)")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for --confirm")
    args = parser.parse_args(argv)

    confirm = None
    if args.confirm:
        confirm = {"token": os.environ.get("SENTRY_AUTH_TOKEN", "").strip(),
                   "org": os.environ.get("SENTRY_ORG", "").strip()}
        missing = [variable for name, variable in (("token", "SENTRY_AUTH_TOKEN"), ("org", "SENTRY_ORG"))
                   if not confirm[name]]
        if missing:
            print("--confirm needs " + ", ".join(missing), file=sys.stderr)
            return 2
        dsn_project, dsn_api_url = _dsn_api_defaults(os.environ.get("SENTRY_DSN", ""))
        confirm["project"] = os.environ.get("SENTRY_PROJECT", "").strip() or dsn_project
        confirm["api_url"] = os.environ.get("SENTRY_API_URL", "").strip() or dsn_api_url
        if confirm["project"].isdigit() and dsn_project and confirm["project"] != dsn_project:
            print(f"SENTRY_PROJECT={confirm['project']} is not the project that owns SENTRY_DSN "
                  f"({dsn_project}); events would never be found there.", file=sys.stderr)
            return 2

    if sentry_sdk is not None:
        # This check needs its own transport, so replace any client initialized at import time.
        sentry_sdk.get_global_scope().set_client(None)
    if not initialize_sentry(service=args.service, transport=RecordingTransport):
        print("Sentry is not active: set SENTRY_DSN (and install sentry-sdk) first.", file=sys.stderr)
        return 2
    client = sentry_sdk.get_client()
    transport = client.transport

    check_id = uuid.uuid4().hex[:12]
    sentry_sdk.set_tag("dapta.check", check_id)
    message_id = sentry_sdk.capture_message(f"Dapta Sentry check {check_id}", level="info")
    with sentry_sdk.start_transaction(op="dapta.check", name="Dapta Sentry check") as root:
        handoff = trace_headers()
    # Same pattern as a comparison: the trace continues in another unit of work.
    with continued_transaction(handoff, op="dapta.check", name="Dapta Sentry check (continued)"):
        with span("dapta.check.step", "linked child span"):
            pass
    client.flush(timeout=15)

    print(f"service:      {args.service}")
    print(f"environment:  {client.options.get('environment')}")
    print(f"release:      {client.options.get('release')}")
    print(f"check tag:    dapta.check:{check_id}")
    print(f"trace id:     {root.trace_id}")

    items = [(message_id, "message")] + [(event_id, kind) for event_id, kind in transport.queued.items()
                                         if kind == "transaction"]
    if len(items) < 3:
        print("expected one message and two transactions; the SDK dropped some before sending "
              "(check SENTRY_TRACES_SAMPLE_RATE > 0)")
    accepted = {event_id: _ingest_outcome(transport, event_id) for event_id, _ in items if event_id}
    stored: dict[str, str] = {}
    if confirm is not None:
        to_confirm = [event_id for event_id, (ok, _) in accepted.items() if ok]
        stored = _confirm_stored(to_confirm, api_url=confirm["api_url"], org=confirm["org"],
                                 project=confirm["project"], token=confirm["token"], timeout=args.timeout)

    ok = len(items) == 3 and bool(message_id)
    for event_id, kind in items:
        if not event_id:
            print(f"{kind:12}  not queued (the SDK dropped it)")
            ok = False
            continue
        delivered, outcome = accepted[event_id]
        if confirm is not None and delivered:
            outcome = stored[event_id]
            delivered = outcome.startswith("stored")
        ok = ok and delivered
        print(f"{kind:12}  {event_id}  {outcome}")

    if ok and confirm is None:
        print("Accepted by ingest. To confirm storage, rerun with --confirm, or in Sentry search "
              f"Issues for dapta.check:{check_id} and Traces for the trace id above.")
    elif ok:
        print("Confirmed: every item is stored in Sentry.")
    else:
        print("NOT CONFIRMED: see the lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
