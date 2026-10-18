from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .comparisons import TERMINAL_RUN_STATES, ComparisonNotFound, ComparisonService, ComparisonStore, QwenGateway
from .config import Settings
from .devin import DevinClient, DevinError
from .factory_runtime import (
    AgentDatasetSource,
    DevinIntegrator,
    FactoryRunner,
    HuggingFacePublisher,
    QwenLifecycle,
    RunPodTrainingBackend,
    SavedEvaluation,
)
from .jobs import (
    AdapterJobOrchestrator,
    InvalidTransition,
    JobBlocked,
    JobConflict,
    JobNotFound,
    JobStore,
)
from .ledger import UsageLedger
from .observability import capture_exception, initialize_sentry, span
from .openrouter import FactoryAgents, OpenRouterClient
from .routing import WorkflowRouter
from .supabase_store import SupabaseFactoryStore


class CreateJobRequest(BaseModel):
    workflow: str = Field(min_length=10, max_length=20_000)
    skill_id: Optional[str] = Field(default=None, max_length=200)
    route: Optional[dict[str, Any]] = None


class RouteRequest(BaseModel):
    workflow: str = Field(min_length=3, max_length=20_000)


class ComparisonRequest(BaseModel):
    workflow: str = Field(min_length=3, max_length=20_000)
    document_text: str = Field(min_length=1, max_length=500_000)
    adapter_id: str = Field(min_length=3, max_length=300)
    attachment_name: Optional[str] = Field(default=None, max_length=300)


class AgentToolRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = Field(default=None, min_length=8, max_length=200)


def create_app(database_path: Path | str | None = None) -> FastAPI:
    initialize_sentry()
    settings = Settings.from_env()
    path = Path(database_path or os.environ.get("DAPTA_LEDGER_PATH", "artifacts/dapta.sqlite3"))
    store = JobStore(path)
    ledger = UsageLedger(path)
    comparison_store = ComparisonStore(path)
    # This process owns all comparison threads: anything still in flight belonged to a previous one.
    comparison_store.recover_interrupted()
    agents = _factory_agents(ledger)
    openrouter = _openrouter(ledger)
    qwen = QwenGateway(url=settings.qwen_url, token=settings.qwen_api_token) if settings.qwen_url else None
    comparison_service = ComparisonService(
        store=comparison_store,
        ledger=ledger,
        settings=settings,
        openrouter=openrouter,
        qwen=qwen,
    )
    router = WorkflowRouter()
    devin = _devin_client()
    training_enabled = os.environ.get("DAPTA_FACTORY_TRAINING_ENABLED", "false").lower() in {
        "1", "true", "yes",
    }
    lifecycle = (
        QwenLifecycle(settings.qwen_url, settings.qwen_api_token)
        if settings.qwen_url and training_enabled
        else None
    )
    trainer = RunPodTrainingBackend(ledger, lifecycle) if lifecycle else None
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = os.environ.get("HF_ADAPTER_REPO")
    publisher = HuggingFacePublisher(hf_repo, hf_token) if hf_token and hf_repo else None
    orchestrator = AdapterJobOrchestrator(
        store,
        ledger,
        agents,
        integrator=DevinIntegrator(devin, store) if devin else None,
        dataset_source=AgentDatasetSource(agents) if agents else None,
        trainer=trainer,
        evaluator=SavedEvaluation(regenerator=trainer.reevaluate if trainer else None),
        publisher=publisher,
    )
    supabase = _supabase_store()
    runner = FactoryRunner(orchestrator, store, mirror=supabase)
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        del application
        if os.environ.get("DAPTA_FACTORY_AUTOSTART", "false").lower() in {
            "1", "true", "yes",
        }:
            runner.start()
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(title="Dapta", version="0.3.0", lifespan=lifespan)
    app.state.factory_runner = runner
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Idempotency-Key", "X-Dapta-Token", "sentry-trace", "baggage"],
    )
    if os.environ.get("DAPTA_SEED_DEMO_DATA", "").lower() in {"1", "true", "yes"}:
        _seed_demo_data(store, comparison_store)

    def authorize(x_dapta_token: Optional[str] = Header(default=None)) -> None:
        if settings.controller_token and x_dapta_token != settings.controller_token:
            raise HTTPException(status_code=401, detail="unauthorized")

    protected = [Depends(authorize)]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "environment": settings.environment,
            "qwen_configured": qwen is not None,
            "openrouter_configured": openrouter is not None,
            "devin_configured": devin is not None,
            "factory_autostart": bool(runner.thread and runner.thread.is_alive()),
            "training_configured": trainer is not None,
            "publishing_configured": publisher is not None,
            "supabase_configured": supabase is not None,
        }

    @app.post("/v1/route", dependencies=protected)
    def route_workflow(request: RouteRequest) -> dict[str, Any]:
        return router.decide(request.workflow, store.list_adapters()).as_dict()

    @app.post("/v1/documents/extract", dependencies=protected)
    async def extract_document(file: UploadFile = File(...)) -> dict[str, Any]:
        contents = await file.read(10 * 1024 * 1024 + 1)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="file exceeds 10 MB")
        try:
            suffix = Path(file.filename or "document.txt").suffix.lower()
            with span("document.preprocess", "Extract document text", **{
                "dapta.document.bytes": len(contents), "dapta.document.extension": suffix,
            }) as extraction:
                text = _extract_text(contents, suffix)
                if extraction is not None:
                    extraction.set_data("dapta.document.output_chars", len(text))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"filename": file.filename, "text": text, "sha256": hashlib.sha256(contents).hexdigest()}

    @app.post("/v1/comparisons", status_code=202, dependencies=protected)
    def create_comparison(
        request: ComparisonRequest,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=200),
    ) -> dict[str, Any]:
        try:
            adapter = store.get_adapter(request.adapter_id)
            return comparison_service.start(
                workflow=request.workflow,
                document_text=request.document_text,
                adapter=adapter,
                idempotency_key=idempotency_key,
                attachment_name=request.attachment_name,
            )
        except JobNotFound as error:
            raise HTTPException(status_code=404, detail="adapter not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/v1/runs", dependencies=protected)
    def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        return comparison_store.list(max(1, min(limit, 100)))

    @app.get("/v1/runs/{run_id}", dependencies=protected)
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            run = comparison_store.get(run_id)
            if run["status"] in TERMINAL_RUN_STATES:
                try:
                    run["usage"] = ledger.run_summary(run_id)
                except ValueError:
                    pass
            return run
        except ComparisonNotFound as error:
            raise HTTPException(status_code=404, detail="run not found") from error

    @app.get("/v1/runs/{run_id}/events", dependencies=protected)
    def stream_run(run_id: str) -> StreamingResponse:
        try:
            comparison_store.get(run_id)
        except ComparisonNotFound as error:
            raise HTTPException(status_code=404, detail="run not found") from error

        def events():
            last_sequence = 0
            while True:
                current = comparison_store.get(run_id)
                for event in current["events"]:
                    if event["sequence"] > last_sequence:
                        last_sequence = event["sequence"]
                        yield f"id: {last_sequence}\nevent: progress\ndata: {json.dumps(event)}\n\n"
                if current["status"] in TERMINAL_RUN_STATES:
                    yield f"event: complete\ndata: {json.dumps({'status': current['status']})}\n\n"
                    break
                time.sleep(1)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/adapters", dependencies=protected)
    def list_v1_adapters() -> list[dict[str, Any]]:
        return store.list_adapters()

    @app.get("/v1/benchmarks", dependencies=protected)
    def list_v1_benchmarks() -> list[dict[str, Any]]:
        return comparison_store.list_benchmarks()

    @app.get("/v1/agent/tools", dependencies=protected)
    def list_agent_tools() -> list[dict[str, Any]]:
        return _agent_tool_catalog()

    @app.post("/v1/agent/tools/{tool_name}", dependencies=protected)
    def call_agent_tool(tool_name: str, request: AgentToolRequest) -> Any:
        arguments = request.arguments
        if tool_name == "list_adapters":
            return store.list_adapters()
        if tool_name == "get_job_status":
            return _get_job(store, str(arguments.get("job_id") or ""))
        if tool_name == "create_adapter":
            from .factory_limits import smoke_training_limits, training_access_open

            if not training_access_open():
                raise HTTPException(status_code=403, detail="Adapter training access is disabled")
            key = request.idempotency_key or str(arguments.get("idempotency_key") or "")
            if len(key) < 8:
                raise HTTPException(status_code=422, detail="idempotency_key is required")
            payload = CreateJobRequest(
                workflow=str(arguments.get("workflow") or ""),
                skill_id=arguments.get("skill_id"),
            )
            job = _create_job(orchestrator, payload, key, runner)
            store.set_artifact(str(job["job_id"]), "training_limits.json", smoke_training_limits())
            return job
        if tool_name == "run_comparison":
            key = request.idempotency_key or str(arguments.get("idempotency_key") or "")
            if len(key) < 8:
                raise HTTPException(status_code=422, detail="idempotency_key is required")
            adapter_id = str(arguments.get("adapter_id") or "")
            try:
                adapter = store.get_adapter(adapter_id)
            except JobNotFound as error:
                raise HTTPException(status_code=404, detail="adapter not found") from error
            return comparison_service.start(
                workflow=str(arguments.get("workflow") or ""),
                document_text=str(arguments.get("document_text") or ""),
                adapter=adapter,
                idempotency_key=key,
                attachment_name=arguments.get("attachment_name"),
            )
        raise HTTPException(status_code=404, detail="unknown agent tool")

    @app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=protected)
    def create_v1_job(
        request: CreateJobRequest,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=200),
    ) -> dict[str, Any]:
        return _create_job(orchestrator, request, idempotency_key, runner)

    @app.get("/v1/jobs", dependencies=protected)
    def list_v1_jobs(limit: int = 100) -> list[dict[str, Any]]:
        return [_job_with_usage(job, ledger) for job in store.list_jobs(limit=limit)]

    @app.get("/v1/jobs/{job_id}", dependencies=protected)
    def get_v1_job(job_id: str) -> dict[str, Any]:
        return _job_with_usage(_get_job(store, job_id), ledger)

    @app.get("/v1/jobs/{job_id}/events", dependencies=protected)
    def stream_job(job_id: str) -> StreamingResponse:
        try:
            store.get_job(job_id)
        except JobNotFound as error:
            raise HTTPException(status_code=404, detail="job not found") from error

        def events():
            last_sequence = 0
            while True:
                current = store.get_job(job_id)
                for event in current["events"]:
                    if event["sequence"] > last_sequence:
                        last_sequence = event["sequence"]
                        yield f"id: {last_sequence}\nevent: progress\ndata: {json.dumps(event)}\n\n"
                if current["status"] in {"ready", "failed", "cancelled"}:
                    yield f"event: complete\ndata: {json.dumps({'status': current['status']})}\n\n"
                    break
                time.sleep(1)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/jobs/{job_id}/advance", dependencies=protected)
    def advance_job(job_id: str) -> dict[str, Any]:
        try:
            return orchestrator.advance(job_id)
        except (JobNotFound, InvalidTransition, JobBlocked, JobConflict, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/jobs/{job_id}/retry", dependencies=protected)
    def retry_job(job_id: str) -> dict[str, Any]:
        try:
            job = store.retry_failed(job_id)
            runner.wake()
            return job
        except (JobNotFound, InvalidTransition, JobConflict) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/jobs/{job_id}/cancel", status_code=202, dependencies=protected)
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return _job_with_usage(runner.cancel(job_id), ledger)
        except JobNotFound as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.post("/v1/jobs/{job_id}/repair-evaluation", dependencies=protected)
    def repair_job_evaluation(job_id: str) -> dict[str, Any]:
        try:
            job = store.repair_truncated_evaluation(job_id)
            runner.wake()
            return _job_with_usage(job, ledger)
        except (JobNotFound, InvalidTransition, JobConflict) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/jobs/{job_id}/devin", status_code=202, dependencies=protected)
    def create_devin_session(job_id: str) -> dict[str, Any]:
        if devin is None:
            raise HTTPException(status_code=503, detail="Devin is not configured")
        try:
            job = store.get_job(job_id)
            if job["status"] != "integrating":
                raise InvalidTransition("job must be in integrating")
            contract = {
                name: store.get_artifact(job_id, f"{name}.json")
                for name in ("skill_spec", "output_schema", "evaluation_plan")
            }
            with span("agent.devin.integrate", "Devin skill integration", **{"dapta.job_id": job_id}):
                session = devin.create_integration_session(
                    job_id=job_id,
                    workflow=job["workflow"],
                    skill_contract=contract,
                )
            store.set_artifact(job_id, "devin_session.json", session.__dict__)
            return {"job": store.get_job(job_id), "devin": session.__dict__}
        except (JobNotFound, InvalidTransition, JobConflict, DevinError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/v1/jobs/{job_id}/devin/sync", dependencies=protected)
    def sync_devin_session(job_id: str) -> dict[str, Any]:
        if devin is None:
            raise HTTPException(status_code=503, detail="Devin is not configured")
        try:
            saved = store.get_artifact(job_id, "devin_session.json")
            session = devin.get_session(saved["session_id"])
            if session.status in {"finished", "completed"} and session.pull_request_url:
                store.set_artifact(job_id, "devin_result.json", session.__dict__)
                job = orchestrator.complete_integration(
                    job_id,
                    commit_sha=session.pull_request_url,
                    test_summary={"status": "reported_by_devin", "session_id": session.session_id},
                )
            else:
                job = store.get_job(job_id)
            return {"job": job, "devin": session.__dict__}
        except (JobNotFound, JobConflict, DevinError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    _add_compatibility_routes(app, store, ledger, comparison_store, orchestrator)
    return app


def _add_compatibility_routes(
    app: FastAPI,
    store: JobStore,
    ledger: UsageLedger,
    comparisons: ComparisonStore,
    orchestrator: AdapterJobOrchestrator,
) -> None:
    @app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    def create_job(
        request: CreateJobRequest,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=200),
    ) -> dict[str, Any]:
        return _create_job(orchestrator, request, idempotency_key)

    @app.get("/jobs")
    def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
        return store.list_jobs(limit=limit)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return _get_job(store, job_id)

    @app.get("/adapters")
    def list_adapters() -> list[dict[str, Any]]:
        return store.list_adapters()

    @app.get("/benchmarks")
    def list_benchmarks() -> list[dict[str, Any]]:
        return comparisons.list_benchmarks() or ledger.list_benchmarks()

    @app.post("/compare")
    def legacy_compare() -> None:
        raise HTTPException(status_code=501, detail="use POST /v1/comparisons")


def _job_with_usage(job: dict[str, Any], ledger: UsageLedger) -> dict[str, Any]:
    """Attach the durable cost ledger without making job polling brittle."""
    result = dict(job)
    try:
        result["usage"] = ledger.run_summary(str(job["job_id"]))
    except ValueError:
        result["usage"] = {
            "total_cost_usd": "0",
            "model_cost_usd": "0",
            "compute_cost_usd": "0",
            "conditions": [],
        }
    return result


def _create_job(
    orchestrator: AdapterJobOrchestrator,
    request: CreateJobRequest,
    key: str,
    runner: FactoryRunner | None = None,
) -> dict[str, Any]:
    try:
        job = orchestrator.create_job(
            workflow=request.workflow,
            skill_id=request.skill_id,
            idempotency_key=key,
            route_decision=request.route,
        )
        if runner:
            runner.wake()
        return job
    except JobConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _get_job(store: JobStore, job_id: str) -> dict[str, Any]:
    try:
        return store.get_job(job_id)
    except JobNotFound as error:
        raise HTTPException(status_code=404, detail="job not found") from error


def _factory_agents(ledger: UsageLedger) -> FactoryAgents | None:
    required = ("OPENROUTER_API_KEY", "OPENROUTER_FACTORY_MODEL", "OPENROUTER_CRITIC_MODEL")
    return FactoryAgents.from_env(ledger) if all(os.environ.get(key) for key in required) else None


def _openrouter(ledger: UsageLedger) -> OpenRouterClient | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    return (
        OpenRouterClient(
            api_key=key,
            ledger=ledger,
            app_url=os.environ.get("OPENROUTER_APP_URL", "https://github.com/adiv-1/Dapta"),
        )
        if key
        else None
    )


def _devin_client() -> DevinClient | None:
    if not os.environ.get("DEVIN_API_KEY"):
        return None
    if not (os.environ.get("DEVIN_ORGANIZATION_ID") or os.environ.get("DEVIN_ORG_ID")):
        return None
    return DevinClient.from_env()


def _supabase_store() -> SupabaseFactoryStore | None:
    try:
        return SupabaseFactoryStore.from_env()
    except Exception as error:
        capture_exception(error, stage="supabase_initialize")
        return None


def _allowed_origins() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get("DAPTA_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
        if item.strip()
    ]


def _extract_text(contents: bytes, suffix: str) -> str:
    if suffix in {".txt", ".md", ".csv", ".json"}:
        return contents.decode("utf-8", errors="replace").strip()
    if suffix == ".pdf":
        from io import BytesIO

        from pypdf import PdfReader

        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(contents)).pages)
        if not text.strip():
            raise ValueError("PDF contains no extractable text; OCR is required")
        return text.strip()
    raise ValueError("supported files are PDF, TXT, Markdown, CSV, and JSON")


def _validate_candidates(candidates: dict[str, Any]) -> dict[str, Any]:
    required = ("train", "dev", "test", "ood")
    missing = [name for name in required if not isinstance(candidates.get(name), list)]
    if missing:
        raise ValueError(f"dataset is missing list splits: {missing}")
    fingerprints: dict[str, str] = {}
    seen: set[str] = set()
    duplicates = 0
    for split in required:
        encoded = json.dumps(candidates[split], sort_keys=True, separators=(",", ":"))
        fingerprints[split] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        for row in candidates[split]:
            fingerprint = hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            duplicates += int(fingerprint in seen)
            seen.add(fingerprint)
    if duplicates:
        raise ValueError(f"dataset contains {duplicates} cross-split duplicates")
    return {
        "valid": True,
        "split_counts": {name: len(candidates[name]) for name in required},
        "split_sha256": fingerprints,
        "frozen_test_sha256": fingerprints["test"],
        "cross_split_duplicates": 0,
    }


def _seed_demo_data(store: JobStore, comparisons: ComparisonStore) -> None:
    adapter_id = "nda-intake/qwen3-vl-8b-v1"
    skill_path = Path(__file__).parent / "skills" / "nda-intake" / "SKILL.md"
    skill_prompt = skill_path.read_text(encoding="utf-8").strip()
    metadata = {
        "name": "NDA Intake",
        "description": "Long NDA to normalized legal intake JSON",
        "keywords": ["nda", "non-disclosure", "legal", "agreement", "contract"],
        "required_fields": ["effective_date", "jurisdiction", "parties", "term"],
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["effective_date", "jurisdiction", "parties", "term"],
            "properties": {
                "effective_date": {"type": ["string", "null"]},
                "jurisdiction": {"type": ["string", "null"]},
                "parties": {"type": "array", "items": {"type": "string"}},
                "term": {"type": ["string", "null"]},
            },
        },
        "status": "ready",
        "benchmark_accuracy": 0.7074626865671643,
        "training_minutes": 13.7073,
        "training_cost_usd": "0",
        "skill_prompt": skill_prompt,
        "skill_path": "backend/dapta/skills/nda-intake/SKILL.md",
        "skill_sha256": hashlib.sha256(skill_prompt.encode("utf-8")).hexdigest(),
    }
    try:
        existing = store.get_adapter(adapter_id)
    except JobNotFound:
        store.register_adapter(
            adapter_id=adapter_id,
            skill_id="nda-intake",
            version="v1",
            base_model="Qwen/Qwen3-VL-8B-Instruct",
            artifact_uri="hf://adityave/dapta-lora-adapters/nda-intake/qwen3-vl-8b-v1",
            metadata=metadata,
        )
    else:
        store.update_adapter(
            adapter_id=adapter_id,
            artifact_uri=str(existing["artifact_uri"]),
            metadata={**dict(existing.get("metadata") or {}), **metadata},
        )
    comparisons.put_benchmark("nda-kleister-v1", _nda_benchmark())


def _nda_benchmark() -> dict[str, Any]:
    return {
        "name": "NDA Intake — Kleister frozen test",
        "view": "adapter_lift",
        "test_documents": 83,
        "test_sha256": "b5a42b7d8c4d9c634bf17eff469b18f4207750c4a028b150b9517564d6ebba65",
        "conditions": {
            "base_no_skill": {
                "label": "Qwen Base", "accuracy": 57.57997218358831,
                "schema_validity": 98.79518072289156, "exact_form_rate": 6.024096385542169,
                "latency_seconds": 1.82796287110845, "input_tokens": 291462, "output_tokens": 4947,
            },
            "base_with_skill": {
                "label": "Qwen + Skill", "accuracy": 61.28133704735376,
                "schema_validity": 98.79518072289156, "exact_form_rate": 8.433734939759036,
                "latency_seconds": 1.7702271022048388, "input_tokens": 303331, "output_tokens": 4738,
            },
            "lora_short_prompt": {
                "label": "Qwen LoRA", "accuracy": 70.74626865671643,
                "schema_validity": 85.54216867469879, "exact_form_rate": 33.734939759036145,
                "latency_seconds": 2.1004295580602177, "input_tokens": 291379, "output_tokens": 3282,
            },
        },
    }


def _agent_tool_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": "list_adapters",
            "description": "List registered Dapta LoRA adapters and benchmark metadata.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "run_comparison",
            "description": "Run Qwen LoRA and configured frontier models on one document.",
            "input_schema": {
                "type": "object",
                "required": ["workflow", "document_text", "adapter_id", "idempotency_key"],
                "properties": {
                    "workflow": {"type": "string"},
                    "document_text": {"type": "string"},
                    "adapter_id": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
            },
        },
        {
            "name": "create_adapter",
            "description": "Create and run a measured adapter factory job through publishing.",
            "input_schema": {
                "type": "object",
                "required": ["workflow", "idempotency_key"],
                "properties": {
                    "workflow": {"type": "string"},
                    "skill_id": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
            },
        },
        {
            "name": "get_job_status",
            "description": "Read stages, artifacts, and evidence for an adapter factory job.",
            "input_schema": {
                "type": "object",
                "required": ["job_id"],
                "properties": {"job_id": {"type": "string"}},
            },
        },
    ]


app = create_app()
