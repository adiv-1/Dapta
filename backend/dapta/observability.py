from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

try:
    import sentry_sdk
except ImportError:  # pragma: no cover - optional until deployment
    sentry_sdk = None


SENSITIVE_KEYS = {
    "authorization",
    "document",
    "document_text",
    "messages",
    "prompt",
    "response",
    "secret",
    "token",
}


def initialize_sentry() -> None:
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn or sentry_sdk is None:
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("SENTRY_ENVIRONMENT", os.environ.get("DAPTA_ENV", "development")),
        release=os.environ.get("SENTRY_RELEASE"),
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        profiles_sample_rate=float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0.25")),
        enable_logs=True,
        send_default_pii=False,
        before_send=_scrub_event,
    )


def _scrub_event(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    del hint
    request = event.get("request")
    if isinstance(request, dict):
        request.pop("data", None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            for key in list(headers):
                if key.lower() in {"authorization", "cookie", "x-dapta-token"}:
                    headers[key] = "[Filtered]"
    event.pop("breadcrumbs", None)
    return _scrub_mapping(event)


def _scrub_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[Filtered]" if key.lower() in SENSITIVE_KEYS else _scrub_mapping(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub_mapping(item) for item in value]
    return value


@contextmanager
def span(op: str, name: str, **attributes: Any) -> Iterator[Any]:
    if sentry_sdk is None or not os.environ.get("SENTRY_DSN"):
        yield None
        return
    with sentry_sdk.start_span(op=op, name=name) as active:
        for key, value in attributes.items():
            if key.lower() not in SENSITIVE_KEYS and value is not None:
                active.set_data(key, value)
        yield active
