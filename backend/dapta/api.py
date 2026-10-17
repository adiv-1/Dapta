from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from .jobs import AdapterJobOrchestrator, JobConflict, JobNotFound, JobStore
from .ledger import UsageLedger


class CreateJobRequest(BaseModel):
    workflow: str = Field(min_length=10, max_length=20_000)
    skill_id: Optional[str] = Field(default=None, max_length=200)


def create_app(database_path: Path | str | None = None) -> FastAPI:
    path = Path(database_path or os.environ.get("DAPTA_LEDGER_PATH", "artifacts/dapta.sqlite3"))
    store = JobStore(path)
    ledger = UsageLedger(path)
    orchestrator = AdapterJobOrchestrator(store, ledger)
    app = FastAPI(title="Dapta", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    def create_job(
        request: CreateJobRequest,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=200),
    ) -> dict:
        try:
            return orchestrator.create_job(
                workflow=request.workflow,
                skill_id=request.skill_id,
                idempotency_key=idempotency_key,
            )
        except JobConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/jobs")
    def list_jobs(limit: int = 100) -> list[dict]:
        try:
            return store.list_jobs(limit=limit)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return store.get_job(job_id)
        except JobNotFound as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.get("/adapters")
    def list_adapters() -> list[dict]:
        return store.list_adapters()

    @app.get("/benchmarks")
    def list_benchmarks() -> list[dict]:
        return ledger.list_benchmarks()

    @app.post("/compare")
    def compare() -> None:
        raise HTTPException(
            status_code=501,
            detail="comparison inference is connected after the Qwen serving stage",
        )

    return app


app = create_app()
