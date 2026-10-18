from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import socket
import sqlite3
import urllib.error
import uuid
from contextlib import ExitStack, contextmanager
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator, Mapping

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
SENTRY_TRACE = re.compile(r"^[0-9a-f]{32}-[0-9a-f]{16}(-[01])?$")
# Header values that can appear inside free text, e.g. a middleware's repr of the request headers.
SECRET_IN_TEXT = re.compile(
    r"""(?i)(['"]?(?:x-dapta-token|authorization|cookie)['"]?\s*[:=]\s*)(?:(['"])[^'"]*\2|[^\s,;'"}]+)""")
BEARER_IN_TEXT = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
HTTP_STATUS_IN_TEXT = re.compile(r"\bHTTP (?:Error )?([1-5][0-9]{2})\b")
HEALTH_PATHS = {"/health", "/healthz"}
# Routine browser polling: low-value when successful, so sampled down (errors are always sent).
POLL_ROUTES = re.compile(r"^/v1/(jobs|runs|adapters|benchmarks)(/[^/]+)?(/events)?/?$")
_SERVICE = {"name": "controller"}


def initialize_sentry(service: str = "controller", **overrides: Any) -> bool:
    """Start Sentry once per process; returns whether a client is active.

    `service` tags every event so the controller and the Qwen worker stay apart when they
    share one DSN. `overrides` replace init options (tests pass a capturing transport).
    """
    if sentry_sdk is None:
        return False
    if sentry_sdk.get_client().is_active():
        return True
    dsn = overrides.pop("dsn", None) or os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    options: dict[str, Any] = {
        "dsn": dsn,
        "environment": os.environ.get("SENTRY_ENVIRONMENT", os.environ.get("DAPTA_ENV", "development")),
        "release": os.environ.get("SENTRY_RELEASE"),
        # Keeps upstream decisions (browser, job traces); sets the rate only where a trace starts here.
        "traces_sampler": _traces_sampler,
        # CPU profiles for this share of sampled transactions (Python stacks, not CUDA kernels).
        "profiles_sample_rate": _rate("SENTRY_PROFILES_SAMPLE_RATE", 0.25),
        "enable_logs": True,
        "send_default_pii": False,
        # Documents, prompts and model output must never leave the controller or worker.
        "max_request_body_size": "never",
        "include_local_variables": False,
        "before_send": _scrub_event,
        "before_send_transaction": _scrub_event,
        # Logs (including third-party library records) bypass before_send.
        "before_send_log": _scrub_log,
        # No automatic trace headers on outgoing requests: OpenRouter, Hugging Face, Supabase and
        # Devin are third parties. The Qwen worker hop adds them explicitly via trace_headers().
        "trace_propagation_targets": [],
    }
    try:
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        # Middleware spans repeat the request headers (including X-Dapta-Token) in their descriptions.
        options["integrations"] = [StarletteIntegration(middleware_spans=False),
                                   FastApiIntegration(middleware_spans=False)]
    except ImportError:  # pragma: no cover - FastAPI is always installed alongside the SDK here
        pass
    options.update(overrides)
    sentry_sdk.init(**options)
    # Global scope, so background work in a fresh isolation scope keeps the tag too.
    sentry_sdk.get_global_scope().set_tag("dapta.service", service)
    _SERVICE["name"] = service
    return True


def _rate(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return min(1.0, max(0.0, value))


def _traces_sampler(context: Mapping[str, Any]) -> float:
    parent = context.get("parent_sampled")
    if parent is not None:
        return 1.0 if parent else 0.0
    scope = context.get("asgi_scope") or {}
    path = str(scope.get("path") or "")
    if path in HEALTH_PATHS:
        return 0.0
    if scope.get("method") == "GET" and POLL_ROUTES.match(path):
        return _rate("SENTRY_POLL_TRACES_SAMPLE_RATE", 0.05)
    return _rate("SENTRY_TRACES_SAMPLE_RATE", 1.0)


def _enabled() -> bool:
    return sentry_sdk is not None and sentry_sdk.get_client().is_active()


def _scrub_event(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    error = (hint.get("exc_info") or (None, None, None))[1]
    if isinstance(error, BaseException):
        event.setdefault("tags", {})["dapta.error_category"] = error_category(error)
        safe_message = getattr(error, "dapta_sentry_message", None)
        values = (event.get("exception") or {}).get("values") or []
        if safe_message and values:
            values[-1]["value"] = safe_message
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


def _scrub_log(log: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    del hint
    # Log arguments can be anything a library chose to format (request payloads, model output),
    # so send the message template, never the formatted arguments.
    attributes = {key: value for key, value in (log.get("attributes") or {}).items()
                  if not key.startswith("sentry.message.parameter")}
    template = attributes.get("sentry.message.template")
    log["body"] = _scrub_mapping(template if isinstance(template, str) else log.get("body"))
    log["attributes"] = _scrub_mapping(attributes)
    return log


def _scrub_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[Filtered]" if key.lower() in SENSITIVE_KEYS else _scrub_mapping(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub_mapping(item) for item in value]
    if isinstance(value, str):
        value = BEARER_IN_TEXT.sub("Bearer [Filtered]", value)
        return SECRET_IN_TEXT.sub(lambda match: f"{match[1]}{match[2] or ''}[Filtered]{match[2] or ''}", value)
    return value


def trace_headers() -> dict[str, str]:
    """The active trace as `sentry-trace`/`baggage`, for handing work to another thread or process."""
    if not _enabled():
        return {}
    headers = {}
    traceparent = sentry_sdk.get_traceparent()
    if traceparent and SENTRY_TRACE.match(traceparent):
        headers["sentry-trace"] = traceparent
        baggage = sentry_sdk.get_baggage()
        if baggage:
            headers["baggage"] = baggage
    return headers


@contextmanager
def continued_transaction(trace: Mapping[str, str] | None, *, op: str, name: str,
                          **attributes: Any) -> Iterator[Any]:
    """Run background work as a transaction in the trace captured by trace_headers().

    The work gets a fresh isolation scope: the threading integration would otherwise hand it the
    request's scope, whose HTTP event processors rename the transaction after the request route and
    attach that request's metadata. Without a captured trace this starts a new one.
    """
    if not _enabled():
        yield None
        return
    from sentry_sdk.scope import Scope, ScopeType, use_isolation_scope

    with use_isolation_scope(Scope(ty=ScopeType.ISOLATION)), sentry_sdk.new_scope():
        transaction = sentry_sdk.continue_trace(dict(trace or {}), op=op, name=name, source="custom")
        with sentry_sdk.start_transaction(transaction) as active:
            for key, value in attributes.items():
                if key.lower() not in SENSITIVE_KEYS and value is not None:
                    active.set_data(key, value)
            yield active


@contextmanager
def span(op: str, name: str, **attributes: Any) -> Iterator[Any]:
    if not _enabled():
        yield None
        return
    with sentry_sdk.start_span(op=op, name=name) as active:
        for key, value in attributes.items():
            if key.lower() not in SENSITIVE_KEYS and value is not None:
                active.set_data(key, value)
        yield active


@contextmanager
def root_transaction(name: str, **attributes: Any) -> Iterator[Any]:
    if not _enabled():
        yield None
        return
    with sentry_sdk.start_transaction(op="ai.pipeline", name=name) as transaction:
        for key, value in attributes.items():
            if key.lower() not in SENSITIVE_KEYS and value is not None:
                transaction.set_data(f"dapta.{key}", value)
        yield transaction


def capture_exception(error: Exception, **attributes: Any) -> None:
    if not _enabled():
        return
    with sentry_sdk.new_scope() as scope:
        for key, value in attributes.items():
            if key.lower() not in SENSITIVE_KEYS and value is not None:
                scope.set_tag(f"dapta.{key}", str(value)[:200])
        sentry_sdk.capture_exception(error)


# --- Error categories -------------------------------------------------------------------------

def error_category(error: BaseException) -> str:
    """A stable, content-free failure class used for tags, log fields and span data."""
    explicit = getattr(error, "dapta_error_category", None)
    if explicit:
        return str(explicit)
    name = type(error).__name__
    text = str(error).lower()
    status = getattr(error, "status", None)
    if not isinstance(status, int):
        status = getattr(error, "code", None) if isinstance(error, urllib.error.HTTPError) else None
    if not isinstance(status, int):
        embedded = HTTP_STATUS_IN_TEXT.search(str(error))
        status = int(embedded[1]) if embedded else None
    if name == "OutOfMemoryError" or "out of memory" in text:
        return "gpu_oom"
    if isinstance(error, (TimeoutError, socket.timeout)) or "timed out" in text or status in (408, 504):
        return "timeout"
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "auth"
    if isinstance(status, int) and status >= 500:
        return "provider_error"
    if isinstance(status, int) and status >= 400:
        return "bad_request"
    if isinstance(error, sqlite3.Error):
        return "persistence"
    if isinstance(error, (urllib.error.URLError, ConnectionError)):
        return "network"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_response"
    if name == "OpenRouterError":
        if "exhausted the completion budget" in text:
            return "output_truncated"
        if "network error" in text:
            return "network"
        return "invalid_response" if "invalid json" in text else "invalid_output"
    if name == "DevinError" and "ended with status" in text:
        return "session_failed"
    if name in {"JobConflict", "JobBlocked", "InvalidTransition"}:
        return {"JobConflict": "conflict", "JobBlocked": "blocked", "InvalidTransition": "invalid_transition"}[name]
    if isinstance(error, ValueError):
        return "validation"
    return "internal"


def redact_for_sentry(error: BaseException, message: str, *, category: str | None = None) -> BaseException:
    """Send `message` to Sentry instead of str(error), which may carry model or training output.

    The original message is untouched for the job store and the UI.
    """
    try:
        error.dapta_sentry_message = message  # type: ignore[attr-defined]
        if category:
            error.dapta_error_category = category  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - built-in exceptions without __dict__
        pass
    return error


_SPAN_STATUS = {
    "timeout": "deadline_exceeded", "rate_limited": "resource_exhausted", "gpu_oom": "resource_exhausted",
    "auth": "unauthenticated", "network": "unavailable", "provider_error": "unavailable",
}


def mark_failed(active: Any, error: BaseException, **attributes: Any) -> str:
    category = error_category(error)
    if active is not None:
        active.set_status(_SPAN_STATUS.get(category, "internal_error"))
        active.set_data("dapta.error_category", category)
        active.set_data("error.type", type(error).__name__)
        for key, value in attributes.items():
            if value is not None:
                active.set_data(f"dapta.{key}", value)
    return category


# --- Structured logs --------------------------------------------------------------------------

def log_event(event: str, *, level: str = "info", **fields: Any) -> None:
    """One structured Sentry log line; correlated with the active trace by the SDK.

    Fields become `dapta.<name>` attributes (ids, stages, models, durations, categories; never content).
    """
    if not _enabled():
        return
    attributes: dict[str, Any] = {"dapta.event": event, "dapta.service": _SERVICE["name"]}
    for key, value in fields.items():
        if value is None or key.lower() in SENSITIVE_KEYS:
            continue
        attributes[f"dapta.{key}"] = value if isinstance(value, (bool, int, float, str)) else str(value)
    getattr(sentry_sdk.logger, level)(event, attributes=attributes)


# --- AI Agent Monitoring ----------------------------------------------------------------------

@contextmanager
def ai_agent(agent: str, *, system: str, model: str | None = None, **attributes: Any) -> Iterator[Any]:
    """A `gen_ai.invoke_agent` span (Sentry AI Agents view); model calls inside are its children."""
    with span("gen_ai.invoke_agent", f"invoke_agent {agent}", **{
        "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": agent, "gen_ai.system": system,
        "gen_ai.request.model": model, **attributes,
    }) as active:
        yield active


@contextmanager
def ai_chat(model: str, *, system: str, agent: str | None = None, **attributes: Any) -> Iterator[Any]:
    """A `gen_ai.chat` span for one model call. Never carries prompts or completions."""
    with span("gen_ai.chat", f"chat {model}", **{
        "gen_ai.operation.name": "chat", "gen_ai.system": system, "gen_ai.request.model": model,
        "gen_ai.agent.name": agent, **attributes,
    }) as active:
        yield active


_USAGE_ATTRIBUTES = {
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "total_tokens": "gen_ai.usage.total_tokens",
    "reasoning_tokens": "gen_ai.usage.output_tokens.reasoning",
    "cached_tokens": "gen_ai.usage.input_tokens.cached",
}


def record_ai_usage(active: Any, usage: Mapping[str, Any]) -> None:
    """Token counts and cost exactly as reported; unreported values are listed, never set to 0."""
    if active is None:
        return
    missing = []
    for key, attribute in _USAGE_ATTRIBUTES.items():
        value = usage.get(key)
        if value is None:
            missing.append(key)
        else:
            active.set_data(attribute, int(value))
    cost = usage.get("cost_usd")
    try:
        cost_value = None if cost is None else float(Decimal(str(cost)))
    except (InvalidOperation, ValueError):
        cost_value = None
    if cost_value is None:
        missing.append("cost_usd")
    else:
        active.set_data("dapta.cost_usd", cost_value)
    active.set_data("dapta.cost_source", str(usage.get("cost_source") or ("none" if cost_value is None else "unknown")))
    if usage.get("response_model"):
        active.set_data("gen_ai.response.model", str(usage["response_model"]))
    if usage.get("provider_name"):
        active.set_data("dapta.provider_name", str(usage["provider_name"]))
    if missing:
        active.set_data("dapta.usage.missing", ",".join(missing))


# --- Factory traces ---------------------------------------------------------------------------

def job_trace_id(job_id: str) -> str:
    try:
        return uuid.UUID(str(job_id)).hex
    except ValueError:
        return hashlib.sha256(str(job_id).encode()).hexdigest()[:32]


def _job_root_span(job_id: str) -> str:
    return hashlib.sha256(f"dapta.job:{job_id}".encode()).hexdigest()[:16]


def job_trace(job_id: str) -> dict[str, str]:
    """Every transaction of one factory job joins one trace (id derived from the job id), sampled.

    The job's stages run for minutes to hours, in a background thread, across controller restarts,
    so they are separate short transactions correlated by this trace, job id, stage and attempt.
    """
    return {"sentry-trace": f"{job_trace_id(job_id)}-{_job_root_span(job_id)}-1"}


def emit_job_root(job_id: str, **attributes: Any) -> None:
    """The job trace's root: a short transaction at job creation that every stage hangs under."""
    if not _enabled():
        return
    from sentry_sdk.scope import Scope, ScopeType, use_isolation_scope
    from sentry_sdk.tracing import Transaction

    with use_isolation_scope(Scope(ty=ScopeType.ISOLATION)), sentry_sdk.new_scope() as scope:
        scope.set_tag("dapta.job_id", job_id)
        transaction = Transaction(
            op="dapta.factory.job", name="factory job", source="custom", sampled=True,
            trace_id=job_trace_id(job_id), span_id=_job_root_span(job_id),
        )
        with sentry_sdk.start_transaction(transaction) as active:
            active.set_data("dapta.job_id", job_id)
            for key, value in attributes.items():
                if value is not None and key.lower() not in SENSITIVE_KEYS:
                    active.set_data(f"dapta.{key}", value)
    current = sentry_sdk.get_current_span()
    if current is not None:
        current.set_data("dapta.job_trace_id", job_trace_id(job_id))


_CURRENT_STAGE: contextvars.ContextVar["FactoryStage | None"] = contextvars.ContextVar(
    "dapta_factory_stage", default=None,
)


class FactoryStage:
    """Telemetry for one execution attempt of one factory stage.

    A stage runs as a transaction in the job trace. Work that only waits for minutes or hours (Devin
    sessions, the GPU training process) calls suspend() so no transaction stays open, then resume()
    to record the outcome in a new transaction of the same trace, job, stage and attempt.
    """

    def __init__(self, job_id: str, stage: str, *, round: int, op: str, name: str) -> None:
        self.job_id, self.stage, self.round, self.op, self.name = job_id, stage, round, op, name
        self.attempt_id = uuid.uuid4().hex[:16]
        self.segment = 0
        self._stack = ExitStack()
        self._transaction: Any = None
        self._token: Any = None

    @property
    def fields(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "stage": self.stage, "round": self.round, "attempt_id": self.attempt_id}

    @property
    def active(self) -> Any:
        return self._transaction

    def __enter__(self) -> "FactoryStage":
        self._token = _CURRENT_STAGE.set(self)
        if _enabled():
            from sentry_sdk.scope import Scope, ScopeType, use_isolation_scope

            self._stack.enter_context(use_isolation_scope(Scope(ty=ScopeType.ISOLATION)))
            scope = self._stack.enter_context(sentry_sdk.new_scope())
            for key, value in self.fields.items():
                scope.set_tag(f"dapta.{key}", str(value))
            # Logs and errors carry the job trace id even while the stage is suspended.
            scope.generate_propagation_context(job_trace(self.job_id))
            self._start(self.name)
        return self

    def _start(self, name: str) -> None:
        transaction = sentry_sdk.continue_trace(job_trace(self.job_id), op=self.op, name=name, source="custom")
        self.segment += 1
        self._transaction = sentry_sdk.start_transaction(transaction)
        self._transaction.__enter__()
        for key, value in {**self.fields, "segment": self.segment}.items():
            self._transaction.set_data(f"dapta.{key}", value)

    def _finish(self, error: BaseException | None = None) -> None:
        if self._transaction is None:
            return
        if error is not None:
            mark_failed(self._transaction, error)
        else:
            self._transaction.set_status("ok")
        self._transaction.__exit__(None, None, None)
        self._transaction = None

    def suspend(self, reason: str) -> None:
        if self._transaction is not None:
            self._transaction.set_data("dapta.suspended_for", reason)
            self._finish()

    def resume(self, label: str) -> None:
        if _enabled() and self._transaction is None:
            self._start(f"{self.name}: {label}")

    def __exit__(self, kind: Any, error: BaseException | None, traceback: Any) -> bool:
        try:
            if error is not None and type(error).__name__ != "JobBlocked" and _enabled():
                self.resume("failed")
                sentry_sdk.capture_exception(error)
            self._finish(error if type(error).__name__ != "JobBlocked" else None)
        finally:
            self._stack.close()
            _CURRENT_STAGE.reset(self._token)
        return False


def current_factory_stage() -> FactoryStage | None:
    return _CURRENT_STAGE.get()
