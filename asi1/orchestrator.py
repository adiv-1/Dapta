from __future__ import annotations

import json
import re
import uuid
from typing import Any, Protocol

from .dapta_client import DaptaClientError
from .session_store import AgentSession, SessionStore
from .validation import validate_output


JOB_ID = re.compile(
    r"\b(?:job-)?[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
ADAPTER_ID = re.compile(r"\badapter\s+([A-Za-z0-9._/-]{3,300})", re.IGNORECASE)
DOCUMENT_MARKER = re.compile(r"\b(?:document|input|receipt|invoice)\s*:\s*", re.IGNORECASE)
MAX_MESSAGE_CHARS = 500_000
MAX_WORKFLOW_CHARS = 20_000


class FactoryClient(Protocol):
    def health(self) -> dict[str, Any]: ...
    def list_adapters(self) -> list[dict[str, Any]]: ...
    def route_workflow(self, workflow: str) -> dict[str, Any]: ...
    def create_adapter(self, *, workflow: str, idempotency_key: str,
                       skill_id: str | None = None) -> dict[str, Any]: ...
    def get_job_status(self, job_id: str) -> dict[str, Any]: ...
    def execute_adapter(self, *, run_id: str, workflow: str, document_text: str,
                        adapter_id: str) -> dict[str, Any]: ...


class LoraFactoryOrchestrator:
    def __init__(self, client: FactoryClient, sessions: SessionStore,
                 *, default_adapter_id: str | None = None,
                 factory_builds_enabled: bool = False):
        self.client = client
        self.sessions = sessions
        self.default_adapter_id = default_adapter_id
        self.factory_builds_enabled = factory_builds_enabled

    def handle(self, sender: str, text: str) -> list[str]:
        message = text.strip()
        if not message:
            return ["Please describe a workflow, request adapter status, or provide a document to process."]
        if len(message) > MAX_MESSAGE_CHARS:
            return ["The message is too large. Send no more than 500,000 characters."]
        session = self.sessions.get(sender)
        try:
            lowered = message.lower()
            if lowered in {"cancel", "cancel build", "never mind", "nevermind"}:
                return [self._cancel(session)]
            if session.pending_action == "confirm_create" and self._is_confirmation(lowered):
                return [self._confirm_create(session)]
            if session.pending_action == "await_document":
                marker = DOCUMENT_MARKER.search(message)
                document_text = message[marker.end():].strip() if marker else message
                return [self._execute(session, document_text)]
            if self._is_help(lowered):
                return [self._help()]
            if self._is_list(lowered):
                return [self._list_adapters(session)]
            if self._is_status(lowered):
                return [self._status(session, message)]
            if self._is_create(lowered):
                return [self._prepare_create(session, message)]
            if self._is_run(lowered):
                return [self._prepare_run(session, message)]
            return [
                "I can create a LoRA adapter, list ready adapters, check a factory job, "
                "or run a ready adapter. Say `help` for examples."
            ]
        except DaptaClientError as error:
            return [f"The DAPTA operation could not be completed: {error}"]

    def _prepare_create(self, session: AgentSession, workflow: str) -> str:
        if not self.factory_builds_enabled:
            return (
                "New adapter builds are disabled for this public agent. A DAPTA operator "
                "can temporarily enable them for a controlled build window."
            )
        if len(workflow) < 10:
            return "Please describe the repeatable workflow in more detail."
        if len(workflow) > MAX_WORKFLOW_CHARS:
            return "The workflow description must be no more than 20,000 characters."
        session.pending_action = "confirm_create"
        session.pending_workflow = workflow
        session.idempotency_key = f"asi1-{uuid.uuid4()}"
        self.sessions.save(session)
        return (
            "DAPTA can route this workflow and create a new LoRA adapter if no existing "
            "specialist matches. Confirmation authorizes paid model calls and may start "
            "GPU training.\n\n"
            "Reply `confirm build` to authorize creation, or `cancel`."
        )

    def _confirm_create(self, session: AgentSession) -> str:
        if not session.pending_workflow or not session.idempotency_key:
            return "There is no adapter build awaiting confirmation."
        decision = self.client.route_workflow(session.pending_workflow)
        if decision.get("decision") == "existing_adapter" and decision.get("adapter_id"):
            session.adapter_id = str(decision["adapter_id"])
            session.pending_action = None
            session.idempotency_key = None
            self.sessions.save(session)
            return (
                f"DAPTA found an existing adapter `{session.adapter_id}` for this workflow; "
                "no new training job was started. Send `run adapter` followed by "
                "`Document: ...` to execute it."
            )
        if decision.get("decision") == "unsupported":
            session.pending_action = None
            session.idempotency_key = None
            self.sessions.save(session)
            return "DAPTA classified this workflow as unsuitable for adapter creation."

        readiness = self.client.health()
        required = {
            "factory_autostart": "factory runner",
            "openrouter_configured": "OpenRouter/JEV",
            "devin_configured": "Devin integration",
            "training_configured": "GPU training",
            "publishing_configured": "adapter publishing",
        }
        unavailable = [label for key, label in required.items() if not readiness.get(key)]
        if unavailable:
            return (
                "DAPTA is not ready to start the build. Missing or inactive: "
                + ", ".join(unavailable)
                + ". No factory job was created."
            )
        job = self.client.create_adapter(
            workflow=session.pending_workflow,
            idempotency_key=session.idempotency_key,
        )
        session.job_id = str(job.get("job_id") or "") or None
        session.pending_action = None
        self.sessions.save(session)
        if not session.job_id:
            raise DaptaClientError("DAPTA did not return a factory job ID")
        status = str(job.get("status") or "queued")
        return (
            f"Adapter factory job `{session.job_id}` was created with status `{status}`. "
            "Ask `status` at any time. I will not claim the adapter is ready until DAPTA reports it."
        )

    def _status(self, session: AgentSession, message: str) -> str:
        match = JOB_ID.search(message)
        job_id = match.group(0) if match else session.job_id
        if not job_id:
            return "I do not have a factory job for this conversation yet."
        job = self.client.get_job_status(job_id)
        session.job_id = job_id
        if job.get("adapter_id"):
            session.adapter_id = str(job["adapter_id"])
        self.sessions.save(session)
        status = str(job.get("status") or "unknown")
        progress = job.get("progress") if isinstance(job.get("progress"), dict) else {}
        percent = progress.get("percent")
        detail = f", {percent}% complete" if isinstance(percent, (int, float)) else ""
        response = f"Factory job `{job_id}` is `{status}`{detail}."
        if status == "ready" and session.adapter_id:
            response += f" Ready adapter: `{session.adapter_id}`."
        if status == "failed":
            response += " The build failed; inspect the DAPTA job evidence before retrying."
        return response

    def _list_adapters(self, session: AgentSession) -> str:
        adapters = self.client.list_adapters()
        if not adapters:
            return "DAPTA has no registered adapters."
        lines = []
        for adapter in adapters[:10]:
            adapter_id = str(adapter.get("adapter_id") or "unknown")
            version = str(adapter.get("version") or "unversioned")
            lines.append(f"- `{adapter_id}` ({version})")
        if len(adapters) == 1:
            session.adapter_id = str(adapters[0].get("adapter_id") or "") or None
            self.sessions.save(session)
        return "Registered DAPTA adapters:\n" + "\n".join(lines)

    def _prepare_run(self, session: AgentSession, message: str) -> str:
        adapter_match = ADAPTER_ID.search(message)
        if adapter_match and adapter_match.group(1).lower() not in {"on", "with"}:
            session.adapter_id = adapter_match.group(1)
        session.adapter_id = session.adapter_id or self.default_adapter_id
        if not session.adapter_id:
            return "Choose an adapter first by saying `list adapters`, then name the adapter to run."
        adapters = self.client.list_adapters()
        selected = next(
            (
                adapter
                for adapter in adapters
                if str(adapter.get("adapter_id") or "") == session.adapter_id
            ),
            None,
        )
        if selected is None:
            return f"Adapter `{session.adapter_id}` is not registered in DAPTA."
        metadata = selected.get("metadata")
        if isinstance(metadata, dict):
            skill_prompt = metadata.get("skill_prompt")
            if isinstance(skill_prompt, str) and skill_prompt.strip():
                session.pending_workflow = skill_prompt.strip()
        marker = DOCUMENT_MARKER.search(message)
        if marker and message[marker.end():].strip():
            return self._execute(session, message[marker.end():].strip())
        session.pending_action = "await_document"
        if not session.pending_workflow:
            session.pending_workflow = "Execute the selected learned workflow and return valid JSON."
        self.sessions.save(session)
        return (
            f"Ready to run `{session.adapter_id}`. Send the synthetic input as `Document: ...`. "
            "Do not send personal or confidential information."
        )

    def _execute(self, session: AgentSession, document_text: str) -> str:
        adapter_id = session.adapter_id or self.default_adapter_id
        if not adapter_id:
            session.pending_action = None
            self.sessions.save(session)
            return "No adapter is selected. Say `list adapters` first."
        run_id = f"asi1-{uuid.uuid4()}"
        result = self.client.execute_adapter(
            run_id=run_id,
            workflow=session.pending_workflow or "Execute the selected learned workflow.",
            document_text=document_text,
            adapter_id=adapter_id,
        )
        output = result.get("output")
        validation = validate_output(output)
        session.pending_action = None
        self.sessions.save(session)
        rendered = json.dumps(output, indent=2, ensure_ascii=False)
        if len(rendered) > 8000:
            rendered = rendered[:7997] + "..."
        return (
            f"LoRA execution completed with adapter `{adapter_id}`.\n"
            f"Run ID: `{run_id}`\n"
            f"Latency: {result.get('latency_ms', 'unknown')} ms\n"
            f"Tokens: {result.get('input_tokens', 'unknown')} input / "
            f"{result.get('output_tokens', 'unknown')} output\n"
            f"Validation: `{json.dumps(validation, separators=(',', ':'))}`\n\n"
            f"```json\n{rendered}\n```"
        )

    def _cancel(self, session: AgentSession) -> str:
        session.pending_action = None
        session.pending_workflow = None
        session.idempotency_key = None
        self.sessions.save(session)
        return "The pending action was cancelled. No adapter build was started."

    @staticmethod
    def _is_confirmation(message: str) -> bool:
        return message in {"confirm", "confirm build", "yes", "yes, confirm", "proceed"}

    @staticmethod
    def _is_help(message: str) -> bool:
        return message in {"help", "/help", "what can you do?", "what can you do"}

    @staticmethod
    def _is_list(message: str) -> bool:
        return "list" in message and ("adapter" in message or "skill" in message)

    @staticmethod
    def _is_status(message: str) -> bool:
        return "status" in message or bool(JOB_ID.search(message))

    @staticmethod
    def _is_create(message: str) -> bool:
        return any(word in message for word in ("create", "build", "train")) and any(
            word in message for word in ("adapter", "lora", "skill")
        )

    @staticmethod
    def _is_run(message: str) -> bool:
        return any(word in message for word in ("run", "execute", "process", "use adapter"))

    @staticmethod
    def _help() -> str:
        return (
            "I am the DAPTA Adapter Factory. Try:\n"
            "- `list adapters`\n"
            "- `create a LoRA adapter that ...`\n"
            "- `status`\n"
            "- `run adapter <adapter-id>` followed by `Document: ...`"
        )
