from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .devin import DevinClient, DevinError
from .jobs import (
    AdapterJobOrchestrator,
    JobBlocked,
    JobConflict,
    JobStore,
)
from .ledger import UsageLedger
from .observability import (
    ai_agent, capture_exception, current_factory_stage, error_category, log_event, mark_failed,
    root_transaction, span,
)


def training_allocation_id(job_id: str, round: int) -> str:
    """Return the stable logical GPU lease for one job and data round."""
    return f"runpod:{job_id}:round-{round}:lease-v2"


def training_instance_id(physical_instance_id: str, job_id: str, round: int) -> str:
    """Scope a persistent RunPod pod to one billable adapter workload."""
    return f"{physical_instance_id}:job-{job_id}:round-{round}"


def evaluation_allocation_id(job_id: str, round: int) -> str:
    return f"runpod:{job_id}:round-{round}:benchmark-repair-v1"


class AgentDatasetSource:
    max_rounds = 3

    def __init__(self, agents: Any):
        self.agents = agents

    def build(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        skill_contract: Mapping[str, Any],
        integration: Mapping[str, Any],
        feedback: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        del integration
        result = self.agents.build_dataset(
            run_id=str(job["job_id"]),
            workflow=str(job["workflow"]),
            skill_contract=skill_contract,
            feedback=feedback,
        )
        return {
            split: [_normalize_case(case, split, index) for index, case in enumerate(result[split])]
            for split in ("train", "dev", "test", "ood")
        }

    def validate(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
        skill_contract: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del job, integration
        import jsonschema

        minimums = {
            "train": int(os.environ.get("DAPTA_MIN_TRAIN_EXAMPLES", "16")),
            "dev": int(os.environ.get("DAPTA_MIN_DEV_EXAMPLES", "4")),
            "test": int(os.environ.get("DAPTA_MIN_TEST_EXAMPLES", "4")),
            "ood": int(os.environ.get("DAPTA_MIN_OOD_EXAMPLES", "2")),
        }
        errors: list[str] = []
        counts: dict[str, int] = {}
        seen_inputs: dict[str, str] = {}
        seen_ids: dict[str, str] = {}
        split_hashes: dict[str, str] = {}
        schema = skill_contract.get("output_schema") or {}
        for split, minimum in minimums.items():
            cases = dataset.get(split)
            if not isinstance(cases, list):
                errors.append(f"{split} must be a list")
                continue
            counts[split] = len(cases)
            if len(cases) < minimum:
                errors.append(f"{split} has {len(cases)} cases; requires at least {minimum}")
            encoded = json.dumps(cases, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            split_hashes[split] = hashlib.sha256(encoded.encode()).hexdigest()
            for index, case in enumerate(cases):
                if not isinstance(case, dict):
                    errors.append(f"{split}[{index}] is not an object")
                    continue
                text = case.get("input")
                if not isinstance(text, str) or len(text.strip()) < 10:
                    errors.append(f"{split}[{index}].input is too short")
                if "output" not in case:
                    errors.append(f"{split}[{index}] is missing output")
                else:
                    try:
                        jsonschema.validate(case["output"], schema)
                    except jsonschema.ValidationError as error:
                        errors.append(f"{split}[{index}].output violates schema: {error.message}")
                case_id = str(case.get("id") or "")
                if not case_id:
                    errors.append(f"{split}[{index}] is missing id")
                elif case_id in seen_ids:
                    errors.append(f"duplicate id across {seen_ids[case_id]} and {split}[{index}]")
                seen_ids[case_id] = f"{split}[{index}]"
                input_fingerprint = hashlib.sha256(
                    str(text or "").strip().encode("utf-8")
                ).hexdigest()
                if input_fingerprint in seen_inputs:
                    errors.append(
                        f"input leakage across {seen_inputs[input_fingerprint]} and {split}[{index}]"
                    )
                seen_inputs[input_fingerprint] = f"{split}[{index}]"
        return {
            "passed": not errors,
            "round": round,
            "counts": counts,
            "split_hashes": split_hashes,
            "frozen_test_sha256": split_hashes.get("test"),
            "errors": errors[:100],
        }


class DevinIntegrator:
    def __init__(self, client: DevinClient, store: JobStore):
        self.client = client
        self.store = store
        self.poll_seconds = float(os.environ.get("DAPTA_DEVIN_POLL_SECONDS", "10"))
        self.timeout_seconds = float(os.environ.get("DAPTA_DEVIN_TIMEOUT_SECONDS", "1800"))

    def integrate(
        self, *, job: Mapping[str, Any], skill_contract: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        job_id = str(job["job_id"])
        started = time.monotonic()
        polls = 0
        stage = current_factory_stage()
        try:
            saved = self.store.get_artifact(job_id, "devin_session.json")
            session_id = saved["session_id"]
        except Exception:
            with ai_agent("devin_skill_integrator", system="devin", model="devin", **{
                "dapta.job_id": job_id, "dapta.retry_count": 0,
            }) as telemetry:
                try:
                    created = self.client.create_integration_session(
                        job_id=job_id,
                        workflow=str(job["workflow"]),
                        skill_contract=skill_contract,
                    )
                except Exception as error:
                    mark_failed(telemetry, error, retry_count=0)
                    raise
                if telemetry is not None:
                    telemetry.set_data("dapta.devin.status", created.status)
            session_id = created.session_id
            self.store.set_artifact(job_id, "devin_session.json", created.__dict__)
            log_event("devin_session_started", job_id=job_id, session_id=session_id, status=created.status)
        if stage is not None:
            stage.suspend("devin_session")
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            polls += 1
            try:
                session = self.client.get_session(session_id)
            except Exception as error:
                if stage is not None:
                    stage.resume("devin_failed")
                log_event("devin_session_failed", level="error", job_id=job_id, session_id=session_id,
                          retry_count=max(0, polls - 1), duration_ms=round((time.monotonic() - started) * 1000),
                          error_type=type(error).__name__, error_category=error_category(error))
                raise
            self.store.set_artifact(job_id, "devin_session.json", session.__dict__)
            self.store.record_progress(job_id, {
                "stage": "integrating",
                "message": f"Devin session: {session.status}",
            })
            normalized = session.status.lower()
            if normalized in {"finished", "completed", "done"}:
                if stage is not None:
                    stage.resume("devin_completed")
                    if stage.active is not None:
                        stage.active.set_data("dapta.devin.status", session.status)
                        stage.active.set_data("dapta.devin.poll_count", polls)
                        if session.acus_consumed is not None:
                            stage.active.set_data("dapta.devin.acus_consumed", session.acus_consumed)
                log_event("devin_session_completed", job_id=job_id, session_id=session_id,
                          status=session.status, retry_count=max(0, polls - 1),
                          duration_ms=round((time.monotonic() - started) * 1000))
                return {
                    **session.__dict__,
                    "evidence_type": "huggingface_skill_bundle",
                    "repository_write_required": False,
                }
            if (session.status_detail or "").lower() in {"waiting_for_user", "inactivity"}:
                if stage is not None:
                    stage.resume("devin_evidence_ready")
                log_event("devin_session_completed", job_id=job_id, session_id=session_id,
                          status=session.status_detail, retry_count=max(0, polls - 1),
                          duration_ms=round((time.monotonic() - started) * 1000))
                return {
                    **session.__dict__,
                    "evidence_type": "huggingface_skill_bundle",
                    "repository_write_required": False,
                    "note": "Devin preserved its scoped integration evidence; no GitHub action is required.",
                }
            if normalized in {"failed", "error", "cancelled", "stopped"}:
                if stage is not None:
                    stage.resume("devin_failed")
                error = DevinError(f"Devin session ended with status {session.status}")
                log_event("devin_session_failed", level="error", job_id=job_id, session_id=session_id,
                          status=session.status, retry_count=max(0, polls - 1),
                          duration_ms=round((time.monotonic() - started) * 1000),
                          error_category=error_category(error))
                raise error
            time.sleep(self.poll_seconds)
        if stage is not None:
            stage.resume("devin_timeout")
        error = DevinError("Devin integration timed out before producing tested evidence")
        log_event("devin_session_failed", level="error", job_id=job_id, session_id=session_id,
                  retry_count=max(0, polls - 1), duration_ms=round((time.monotonic() - started) * 1000),
                  error_category=error_category(error))
        raise error


class QwenLifecycle:
    def __init__(self, url: str, token: str | None):
        self.url = url.rstrip("/")
        self.token = token

    def unload(self) -> None:
        self._post("/admin/unload", timeout=180)

    def load(self) -> None:
        self._post("/admin/load", timeout=600)

    def _post(self, path: str, *, timeout: int) -> None:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            f"{self.url}{path}", data=b"{}", method="POST", headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            if payload.get("status") != "ok":
                raise RuntimeError(f"Qwen lifecycle returned {payload}")
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Qwen lifecycle request failed: {error}") from error


class RunPodTrainingBackend:
    def __init__(self, ledger: UsageLedger, lifecycle: QwenLifecycle):
        self.ledger = ledger
        self.lifecycle = lifecycle
        self.work_dir = Path(os.environ.get("DAPTA_FACTORY_WORK_DIR", "artifacts/factory"))
        self.hourly_cost = Decimal(os.environ.get("RUNPOD_GPU_HOURLY_COST_USD", "0.59"))
        self.maximum_cost = Decimal(os.environ.get("DAPTA_FACTORY_MAX_JOB_COST_USD", "5"))
        self.maximum_hours = Decimal(os.environ.get("DAPTA_FACTORY_MAX_TRAIN_HOURS", "2"))

    def preflight(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
    ) -> None:
        del job, round, integration
        if not dataset.get("train") or not dataset.get("test"):
            raise JobBlocked("awaiting_compute: training and frozen test data are required")
        estimate = self.hourly_cost * self.maximum_hours
        if estimate > self.maximum_cost:
            raise JobBlocked(
                f"awaiting_compute: ${estimate} worst-case estimate exceeds ${self.maximum_cost} cap"
            )

    def train(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
        on_progress: Any,
    ) -> Mapping[str, Any]:
        del integration
        job_id = str(job["job_id"])
        run_dir = self.work_dir / job_id / f"round-{round}"
        output_dir = run_dir / "adapter"
        run_dir.mkdir(parents=True, exist_ok=True)
        contract = {
            name: job.get(name) for name in ("skill_spec", "output_schema", "evaluation_plan")
        }
        bundle = {
            "job_id": job_id,
            "workflow": job["workflow"],
            "skill_contract": contract,
            "dataset": dataset,
            "output_dir": str(output_dir.resolve()),
            "base_model": os.environ.get("QWEN_BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            "epochs": float(os.environ.get("DAPTA_FACTORY_EPOCHS", "2")),
            "max_length": int(os.environ.get("DAPTA_FACTORY_MAX_LENGTH", "4096")),
            "max_train_examples": int(os.environ.get("DAPTA_FACTORY_MAX_TRAIN_EXAMPLES", "64")),
            "max_test_examples": int(os.environ.get("DAPTA_FACTORY_MAX_TEST_EXAMPLES", "12")),
        }
        for name in ("skill_spec", "output_schema", "evaluation_plan"):
            bundle["skill_contract"][name] = self._artifact(job_id, name)
        bundle_path = run_dir / "bundle.json"
        bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
        allocation_id = training_allocation_id(job_id, round)
        physical_instance_id = os.environ.get("RUNPOD_POD_ID", "existing-pod")
        started = _now()
        self.ledger.register_compute(
            run_id=job_id,
            allocation_id=allocation_id,
            provider="runpod",
            instance_id=training_instance_id(physical_instance_id, job_id, round),
            plan_id=os.environ.get("RUNPOD_GPU_PLAN", "RTX-A40-48GB"),
            region=os.environ.get("RUNPOD_REGION", "CA-MTL-1"),
            gpu_type=os.environ.get("RUNPOD_GPU_TYPE", "RTX A40"),
            gpu_count=1,
            hourly_cost_usd=self.hourly_cost,
            pricing_source="runpod_console",
            registered_at=str(job["created_at"]),
            metadata={
                "provider_instance_id": physical_instance_id,
                "billing_scope": "adapter_job_round",
            },
        )
        self.ledger.record_compute_event(allocation_id=allocation_id, event_type="workload_started")
        training_clock = time.monotonic()
        try:
            result = self._execute_gpu_worker(
                bundle_path=bundle_path,
                result_path=run_dir / "result.json",
                training_clock=training_clock,
                on_progress=on_progress,
            )
        finally:
            completed = _now()
            elapsed = Decimal(str((_parse(completed) - _parse(started)).total_seconds()))
            cost = (elapsed / Decimal(3600) * self.hourly_cost).quantize(Decimal("0.000001"))
            self.ledger.record_compute_event(
                allocation_id=allocation_id, event_type="workload_finished"
            )
            self.ledger.record_compute_cost(
                allocation_id=allocation_id,
                amount_usd=cost,
                source="calculated",
                period_start=started,
                period_end=completed,
                metadata={"billing_unit": "millisecond", "active_seconds": float(elapsed)},
            )
        result["training_cost_usd"] = str(cost)
        result["adapter_dir"] = str(output_dir.resolve())
        return result

    def reevaluate(
        self, *, job: Mapping[str, Any], round: int, training: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Regenerate frozen benchmark outputs for saved weights without retraining."""
        job_id = str(job["job_id"])
        run_dir = self.work_dir / job_id / f"round-{round}" / "benchmark-repair"
        run_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "job_id": job_id,
            "workflow": job["workflow"],
            "skill_contract": {
                name: self._artifact(job_id, name) for name in (
                    "skill_spec", "output_schema", "evaluation_plan",
                )
            },
            "dataset": self._artifact(job_id, "dataset", round=round),
            "adapter_dir": str(training["adapter_dir"]),
            "base_model": str(training["base_model"]),
            "max_test_examples": int(os.environ.get("DAPTA_FACTORY_MAX_TEST_EXAMPLES", "12")),
        }
        bundle_path = run_dir / "bundle.json"
        bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
        store = JobStore(os.environ.get("DAPTA_LEDGER_PATH", "artifacts/dapta.sqlite3"))
        allocation_id = evaluation_allocation_id(job_id, round)
        started = _now()
        self.ledger.register_compute(
            run_id=job_id,
            allocation_id=allocation_id,
            provider="runpod",
            instance_id=training_instance_id(
                os.environ.get("RUNPOD_POD_ID", "existing-pod"), job_id, round,
            ) + ":benchmark-repair",
            plan_id=os.environ.get("RUNPOD_GPU_PLAN", "RTX-A40-48GB"),
            region=os.environ.get("RUNPOD_REGION", "CA-MTL-1"),
            gpu_type=os.environ.get("RUNPOD_GPU_TYPE", "RTX A40"),
            gpu_count=1,
            hourly_cost_usd=self.hourly_cost,
            pricing_source="runpod_console",
            registered_at=started,
            metadata={"billing_scope": "benchmark_only_repair", "training_skipped": True},
        )
        self.ledger.record_compute_event(allocation_id=allocation_id, event_type="workload_started")
        clock = time.monotonic()
        try:
            report = self._execute_gpu_worker(
                bundle_path=bundle_path,
                result_path=run_dir / "result.json",
                training_clock=clock,
                on_progress=lambda item: store.record_progress(job_id, item),
                evaluate_only=True,
            )
        finally:
            completed = _now()
            elapsed = Decimal(str((_parse(completed) - _parse(started)).total_seconds()))
            cost = (elapsed / Decimal(3600) * self.hourly_cost).quantize(Decimal("0.000001"))
            self.ledger.record_compute_event(allocation_id=allocation_id, event_type="workload_finished")
            self.ledger.record_compute_cost(
                allocation_id=allocation_id,
                amount_usd=cost,
                source="calculated",
                period_start=started,
                period_end=completed,
                metadata={"billing_unit": "millisecond", "active_seconds": float(elapsed)},
            )
        report["evaluation_cost_usd"] = str(cost)
        report["evaluation_only"] = True
        return report

    def _execute_gpu_worker(
        self,
        *,
        bundle_path: Path,
        result_path: Path,
        training_clock: float,
        on_progress: Any,
        evaluate_only: bool = False,
    ) -> dict[str, Any]:
        progress_stage = "evaluation" if evaluate_only else "training"
        on_progress({"stage": progress_stage, "message": "Unloading inference model"})
        self.lifecycle.unload()
        try:
            command = [sys.executable, "-m", "dapta.factory_gpu", "--bundle", str(bundle_path)]
            if evaluate_only:
                command.append("--evaluate-only")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            telemetry_stage = current_factory_stage()
            if telemetry_stage is not None:
                telemetry_stage.suspend("gpu_evaluation" if evaluate_only else "gpu_training")
            timed_out = threading.Event()

            def stop_over_budget() -> None:
                timed_out.set()
                process.terminate()

            timer = threading.Timer(float(self.maximum_hours * Decimal(3600)), stop_over_budget)
            timer.daemon = True
            timer.start()
            assert process.stdout is not None
            tail: list[str] = []
            try:
                for line in process.stdout:
                    line = line.strip()
                    tail.append(line)
                    tail = tail[-30:]
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("type") == "progress":
                        active_seconds = Decimal(str(time.monotonic() - training_clock))
                        event["cost_usd"] = str(
                            (active_seconds / Decimal(3600) * self.hourly_cost).quantize(
                                Decimal("0.000001")
                            )
                        )
                        on_progress(event)
                code = process.wait()
            finally:
                timer.cancel()
            if timed_out.is_set():
                raise RuntimeError(
                    f"GPU worker exceeded the {self.maximum_hours}-hour training cap"
                )
            if code != 0:
                raise RuntimeError("GPU worker failed: " + " | ".join(tail[-8:]))
            if not result_path.exists():
                raise RuntimeError("GPU worker did not write result.json")
            return json.loads(result_path.read_text(encoding="utf-8"))
        finally:
            telemetry_stage = current_factory_stage()
            if telemetry_stage is not None:
                telemetry_stage.resume("gpu_worker_result")
            on_progress({"stage": progress_stage, "message": "Reloading inference model"})
            self.lifecycle.load()

    def _artifact(self, job_id: str, name: str, *, round: int = 1) -> Any:
        database = os.environ.get("DAPTA_LEDGER_PATH", "artifacts/dapta.sqlite3")
        path = f"{name}.json" if round == 1 else f"round-{round}/{name}.json"
        return JobStore(database).get_artifact(job_id, path)


class SavedEvaluation:
    def __init__(self, regenerator: Any | None = None):
        self.regenerator = regenerator

    def evaluate(
        self, *, job: Mapping[str, Any], round: int, training: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        report = dict(training["evaluation"])
        existing_conditions = report.get("conditions") or {}
        was_truncated = any(
            isinstance(value, Mapping) and value.get("truncated_cases")
            for value in existing_conditions.values()
        )
        previous_cap = max(
            [int(value.get("generation_max_new_tokens", 0))
             for value in existing_conditions.values() if isinstance(value, Mapping)] or [0]
        )
        configured_cap = int(os.environ.get("DAPTA_FACTORY_MAX_NEW_TOKENS", "4096"))
        if was_truncated and self.regenerator is not None and configured_cap > previous_cap:
            report = dict(self.regenerator(
                job=job,
                round=int(training.get("_source_round", round)),
                training=training,
            ))
        conditions = report.get("conditions") or {}
        base_skill = float(conditions.get("base_with_skill", {}).get("field_f1", 0))
        lora = float(conditions.get("lora_short_prompt", {}).get("field_f1", 0))
        schema = float(conditions.get("lora_short_prompt", {}).get("schema_pass_rate", 0))
        minimum_lift = float(os.environ.get("DAPTA_MIN_LORA_LIFT", "0.02"))
        minimum_schema = float(os.environ.get("DAPTA_MIN_SCHEMA_PASS_RATE", "0.8"))
        truncated = {
            name: list(value.get("truncated_cases") or [])
            for name, value in conditions.items()
            if isinstance(value, Mapping) and value.get("truncated_cases")
        }
        report["passed"] = lora >= base_skill + minimum_lift and schema >= minimum_schema
        report["repairable"] = not bool(truncated)
        report["thresholds"] = {
            "minimum_lift": minimum_lift,
            "minimum_schema_pass_rate": minimum_schema,
        }
        if truncated:
            report["failure_reason"] = "evaluation_output_truncated"
            report["truncated_cases"] = truncated
        if not report["passed"]:
            report["dev_feedback"] = {
                "observed_lift": lora - base_skill,
                "schema_pass_rate": schema,
                "instruction": (
                    "Increase the evaluator output budget and rerun the frozen benchmark; "
                    "do not generate repair data."
                    if truncated
                    else "Add targeted valid examples for failing DEV fields; do not inspect TEST."
                ),
            }
        return report


class HuggingFacePublisher:
    def __init__(self, repo_id: str, token: str):
        self.repo_id = repo_id
        self.token = token

    def promote(
        self,
        *,
        job: Mapping[str, Any],
        training: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        skill_contract: Mapping[str, Any],
    ) -> str:
        from huggingface_hub import HfApi

        adapter_dir = Path(str(training["adapter_dir"]))
        if not adapter_dir.is_dir():
            raise RuntimeError("adapter directory no longer exists")
        skill_id = _slug(
            str(job.get("skill_id") or skill_contract["skill_spec"].get("name") or job["job_id"])
        )
        path = f"generated/{skill_id}/{training['version']}"
        skill_dir = adapter_dir / "skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(_render_skill(skill_contract), encoding="utf-8")
        (skill_dir / "skill_spec.json").write_text(
            json.dumps(skill_contract["skill_spec"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (skill_dir / "output_schema.json").write_text(
            json.dumps(skill_contract["output_schema"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (skill_dir / "evaluation_plan.json").write_text(
            json.dumps(skill_contract["evaluation_plan"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if skill_contract.get("integration") is not None:
            (skill_dir / "integration_evidence.json").write_text(
                json.dumps(skill_contract["integration"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        (adapter_dir / "metrics.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (adapter_dir / "manifest.json").write_text(
            json.dumps({
                "job_id": job["job_id"],
                "workflow_hash": job["workflow_hash"],
                "base_model": training["base_model"],
                "version": training["version"],
            }, indent=2),
            encoding="utf-8",
        )
        HfApi(token=self.token).upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=str(adapter_dir),
            path_in_repo=path,
            commit_message=f"Publish Dapta adapter {skill_id} {training['version']}",
        )
        training["adapter_id"] = path
        return f"hf://{self.repo_id}/{path}"


class FactoryRunner:
    def __init__(
        self,
        orchestrator: AdapterJobOrchestrator,
        store: JobStore,
        mirror: Any | None = None,
    ):
        self.orchestrator = orchestrator
        self.store = store
        self.mirror = mirror
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.store.recover_interrupted_stages()
        for job in self.store.list_jobs():
            self._mirror(job)
        self.thread = threading.Thread(target=self._loop, name="dapta-factory", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        if self.thread:
            self.thread.join(timeout=10)

    def wake(self) -> None:
        self.wake_event.set()

    def run_once(self) -> bool:
        active = [job for job in self.store.list_jobs() if job["status"] not in {"ready", "failed"}]
        if not active:
            return False
        job = active[-1]
        job_id = job["job_id"]
        self._mirror(job)
        try:
            with root_transaction("dapta.adapter_build", job_id=job_id, status=job["status"]):
                self.orchestrator.advance(job_id)
        except JobBlocked as blocked:
            self.store.record_progress(job_id, {
                "stage": job["status"], "message": str(blocked),
            })
            return False
        except JobConflict:
            return False
        except Exception as error:
            capture_exception(error, job_id=job_id, stage=job["status"])
        finally:
            self._mirror(self.store.get_job(job_id))
        return True

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            progressed = self.run_once()
            if not progressed:
                self.wake_event.wait(timeout=5)
                self.wake_event.clear()

    def _mirror(self, job: Mapping[str, Any]) -> None:
        if self.mirror is None:
            return
        try:
            self.mirror.sync_job(job, self.store)
        except Exception as error:
            capture_exception(error, job_id=job["job_id"], stage="supabase_mirror")


def _normalize_case(case: Any, split: str, index: int) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        return {"id": f"{split}-{index}", "input": str(case), "output": None}
    source = case.get("input") or case.get("document") or case.get("prompt") or case.get("text")
    output = case.get("output", case.get("answer", case.get("expected")))
    return {"id": str(case.get("id") or f"{split}-{index}"), "input": str(source or ""), "output": output}


def _render_skill(contract: Mapping[str, Any]) -> str:
    spec = contract["skill_spec"]
    public_contract = {
        key: contract[key]
        for key in ("skill_spec", "output_schema", "evaluation_plan")
        if key in contract
    }
    return "# " + str(spec.get("name") or "Dapta Skill") + "\n\n" + json.dumps(
        public_contract, indent=2, ensure_ascii=False
    ) + "\n"


def _slug(value: str) -> str:
    return "-".join("".join(character if character.isalnum() else " " for character in value.lower()).split())[:80]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
