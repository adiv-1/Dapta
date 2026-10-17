from __future__ import annotations

import argparse
import fcntl
import ipaddress
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlencode

import httpx

from .ledger import UsageLedger
from .preflight import load_env


API_URL = "https://api.vultr.com/v2"
MANAGED_TAG = "dapta-managed"
CONFIRMATION = "PROVISION"
SAFE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class VultrError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerConfig:
    api_key: str
    region: str
    plan_id: str
    os_id: int
    ssh_key_id: str
    ssh_private_key: Path
    hourly_cost_usd: Decimal
    max_job_cost_usd: Decimal
    max_job_seconds: int
    ready_timeout_seconds: int
    poll_seconds: int
    ssh_user: str = "root"

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        config = cls(
            api_key=_env("VULTR_API_KEY"),
            region=_safe_env("VULTR_REGION"),
            plan_id=_safe_env("VULTR_GPU_PLAN_ID"),
            os_id=int(_env("VULTR_OS_ID")),
            ssh_key_id=_safe_env("VULTR_SSH_KEY_ID"),
            ssh_private_key=Path(_env("VULTR_SSH_PRIVATE_KEY_PATH")).expanduser().resolve(),
            hourly_cost_usd=Decimal(_env("VULTR_HOURLY_COST_USD")),
            max_job_cost_usd=Decimal(os.environ.get("VULTR_MAX_JOB_COST_USD", "5.00")),
            max_job_seconds=int(os.environ.get("VULTR_MAX_JOB_SECONDS", "7200")),
            ready_timeout_seconds=int(os.environ.get("VULTR_READY_TIMEOUT_SECONDS", "1800")),
            poll_seconds=int(os.environ.get("VULTR_POLL_SECONDS", "15")),
            ssh_user=os.environ.get("VULTR_SSH_USER", "root"),
        )
        if not config.ssh_private_key.is_file():
            raise ValueError("VULTR_SSH_PRIVATE_KEY_PATH is not a file")
        if config.hourly_cost_usd <= 0 or config.max_job_cost_usd <= 0:
            raise ValueError("Vultr costs must be positive")
        if min(config.max_job_seconds, config.ready_timeout_seconds, config.poll_seconds) <= 0:
            raise ValueError("Vultr timeouts must be positive")
        if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", config.ssh_user):
            raise ValueError("invalid VULTR_SSH_USER")
        if config.projected_cost_usd > config.max_job_cost_usd:
            raise ValueError(
                f"projected cost ${config.projected_cost_usd} exceeds the "
                f"${config.max_job_cost_usd} per-job limit"
            )
        return config

    @property
    def projected_cost_usd(self) -> Decimal:
        return self.hourly_cost_usd * Decimal(self.max_job_seconds) / Decimal(3600)


class VultrClient:
    def __init__(self, api_key: str, *, base_url: str = API_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def plans(self) -> list[dict[str, Any]]:
        return self._request("GET", "/plans", query={"per_page": 500}).get(
            "plans", []
        )

    def available_plans(self, region: str) -> set[str]:
        value = self._request("GET", f"/regions/{region}/availability")
        return set(value.get("available_plans", []))

    def instances(self) -> list[dict[str, Any]]:
        return self._request("GET", "/instances", query={"per_page": 500}).get("instances", [])

    def create_instance(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/instances", payload=payload)["instance"]

    def get_instance(self, instance_id: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", f"/instances/{instance_id}")["instance"]
        except VultrError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def delete_instance(self, instance_id: str) -> None:
        try:
            self._request("DELETE", f"/instances/{instance_id}")
        except VultrError as exc:
            if "HTTP 404" not in str(exc):
                raise

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        try:
            transport = httpx.HTTPTransport(local_address="0.0.0.0")
            with httpx.Client(transport=transport, timeout=30) as client:
                response = client.request(
                    method,
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "User-Agent": "dapta-worker/1",
                    },
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                message = exc.response.json().get("error", "")
                if isinstance(message, str) and message:
                    detail = f": {message[:200]}"
            except (AttributeError, TypeError, ValueError):
                pass
            raise VultrError(
                f"Vultr API HTTP {exc.response.status_code} for {method} {path}{detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VultrError(f"Vultr API request failed for {method} {path}: {type(exc).__name__}") from exc
        return response.json() if response.content else {}


class SshRunner:
    def __init__(self, config: WorkerConfig, repo_dir: Path):
        self.config = config
        self.repo_dir = repo_dir.resolve()
        self._temporary = tempfile.TemporaryDirectory(prefix="dapta-ssh-")
        self.known_hosts = Path(self._temporary.name) / "known_hosts"

    def close(self) -> None:
        self._temporary.cleanup()

    def verify_commit(self, commit: str) -> None:
        commit = _safe_commit(commit)
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=self.repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise ValueError("commit does not exist in the local repository")

    def wait_ready(self, host: str, timeout: int, poll_seconds: int) -> None:
        deadline = time.monotonic() + timeout
        while True:
            result = subprocess.run(
                self._ssh_args(host) + ["true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("SSH did not become ready")
            time.sleep(poll_seconds)

    def upload_commit(self, host: str, commit: str) -> None:
        _safe_commit(commit)
        archive = subprocess.Popen(
            ["git", "archive", "--format=tar", commit],
            cwd=self.repo_dir,
            stdout=subprocess.PIPE,
        )
        remote = subprocess.run(
            self._ssh_args(host) + ["mkdir -p /opt/dapta && tar -x -C /opt/dapta"],
            stdin=archive.stdout,
            check=False,
        )
        if archive.stdout:
            archive.stdout.close()
        archive_code = archive.wait()
        if archive_code or remote.returncode:
            raise RuntimeError("failed to upload the exact Git commit")

    def upload_environment(self, host: str, values: Mapping[str, str]) -> None:
        allowed = {"HF_TOKEN", "HF_ADAPTER_REPO", "DAPTA_REMOTE_WORKER"}
        if set(values) - allowed:
            raise ValueError("worker environment contains an unapproved key")
        content = "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items())
        subprocess.run(
            self._ssh_args(host) + ["umask 077; cat > /opt/dapta/.env.worker"],
            input=content.encode(),
            check=True,
        )

    def prepare(
        self,
        host: str,
        job_id: str,
        *,
        skill_id: str,
        limit: int | None,
        timeout: int,
    ) -> None:
        job_id = _safe_value(job_id, "job_id")
        skill_id = _skill_id(skill_id)
        limit_arg = f" --limit {limit}" if limit is not None else ""
        if skill_id == "receipt-to-expense":
            dataset_command = (
                f"DAPTA_REMOTE_WORKER=1 PYTHONPATH=backend python -m dapta.receipts "
                f"--job-dir /opt/dapta-jobs/{job_id}{limit_arg}\n"
                f"PYTHONPATH=backend python -m dapta.validation "
                f"--dataset /opt/dapta-jobs/{job_id}/receipt-dataset "
                f"--report /opt/dapta-jobs/{job_id}/validation.json"
            )
        else:
            dataset_command = (
                f"DAPTA_REMOTE_WORKER=1 PYTHONPATH=backend python -m dapta.dashboards "
                f"--job-dir /opt/dapta-jobs/{job_id}{limit_arg} "
                f"--report /opt/dapta-jobs/{job_id}/validation.json"
            )
        script = f"""set -euo pipefail
cd /opt/dapta
nvidia-smi
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-data.txt
set -a
. .env.worker
set +a
mkdir -p /opt/dapta-jobs/{job_id}
{dataset_command}
"""
        self._run(host, script, timeout)

    def read_validation(self, host: str, job_id: str, *, timeout: int) -> dict[str, Any]:
        job_id = _safe_value(job_id, "job_id")
        result = subprocess.run(
            self._ssh_args(host) + [f"cat /opt/dapta-jobs/{job_id}/validation.json"],
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError("validation report is not a JSON object")
        return value

    def train(
        self,
        host: str,
        job_id: str,
        version: str,
        *,
        skill_id: str,
        smoke: bool,
        timeout: int,
    ) -> None:
        job_id = _safe_value(job_id, "job_id")
        version = _safe_value(version, "version")
        skill_id = _skill_id(skill_id)
        dataset_name = "receipt-dataset" if skill_id == "receipt-to-expense" else "dashboard-dataset"
        contract = f"research/adapter-assets/skills/{skill_id}"
        smoke_args = " --epochs 1 --max-steps 2 --smoke-inference" if smoke else ""
        script = f"""set -euo pipefail
cd /opt/dapta
. .venv/bin/activate
set -a
. .env.worker
set +a
python -m pip install -r backend/requirements.txt
PYTHONPATH=backend python -m dapta.train --dataset /opt/dapta-jobs/{job_id}/{dataset_name} --output /opt/dapta-jobs/{job_id}/adapters --skill {skill_id} --version {version} --contract {contract}{smoke_args}
"""
        self._run(host, script, timeout)

    def _run(self, host: str, script: str, timeout: int) -> None:
        subprocess.run(
            self._ssh_args(host) + ["bash -s"],
            input=script.encode(),
            check=True,
            timeout=timeout,
        )

    def _ssh_args(self, host: str) -> list[str]:
        return [
            "ssh",
            "-i",
            str(self.config.ssh_private_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            f"{self.config.ssh_user}@{host}",
        ]


class VultrWorker:
    def __init__(
        self,
        config: WorkerConfig,
        client: VultrClient,
        ledger: UsageLedger,
        runner: SshRunner,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.client = client
        self.ledger = ledger
        self.runner = runner
        self.sleep = sleep
        self.monotonic = monotonic

    def plan(
        self,
        job_id: str,
        commit: str,
        *,
        skill_id: str = "receipt-to-expense",
        limit: int | None,
    ) -> dict[str, Any]:
        job_id = _safe_value(job_id, "job_id")
        commit = _safe_commit(commit)
        skill_id = _skill_id(skill_id)
        self.runner.verify_commit(commit)
        plan = self._preflight()
        return {
            "job_id": job_id,
            "skill_id": skill_id,
            "commit": commit,
            "region": self.config.region,
            "plan_id": self.config.plan_id,
            "gpu": plan.get("gpu_type", "NVIDIA A40"),
            "hourly_cost_usd": str(self.config.hourly_cost_usd),
            "max_job_seconds": self.config.max_job_seconds,
            "projected_max_cost_usd": str(self.config.projected_cost_usd),
            "budget_limit_usd": str(self.config.max_job_cost_usd),
            "dataset_limit_per_source_split": limit,
            "block_storage": False,
            "will_provision": False,
        }

    def run(
        self,
        *,
        run_id: str,
        job_id: str,
        commit: str,
        version: str,
        skill_id: str = "receipt-to-expense",
        smoke: bool = False,
        limit: int | None,
        confirmation: str,
        validation_report_path: Path | None = None,
    ) -> dict[str, Any]:
        if confirmation != CONFIRMATION:
            raise ValueError(f"paid execution requires --confirm-spend {CONFIRMATION}")
        run_id = _safe_value(run_id, "run_id")
        job_id = _safe_value(job_id, "job_id")
        commit = _safe_commit(commit)
        version = _safe_value(version, "version")
        skill_id = _skill_id(skill_id)
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")

        self.runner.verify_commit(commit)
        self._preflight()
        deadline = self.monotonic() + self.config.max_job_seconds
        label = f"dapta-{job_id[:48]}-{uuid.uuid4().hex[:8]}"
        payload = self._create_payload(job_id, label)
        requested_at = _utcnow()
        instance: dict[str, Any] | None = None
        allocation_id: str | None = None
        error: BaseException | None = None
        try:
            try:
                instance = self.client.create_instance(payload)
            except Exception:
                matches = [item for item in self.client.instances() if item.get("label") == label]
                if len(matches) != 1:
                    for match in matches:
                        self.client.delete_instance(str(match["id"]))
                    raise
                instance = matches[0]

            instance_id = str(instance["id"])
            allocation_id = f"vultr:{instance_id}"
            self.ledger.register_compute(
                run_id=run_id,
                allocation_id=allocation_id,
                provider="vultr",
                instance_id=instance_id,
                plan_id=self.config.plan_id,
                region=self.config.region,
                gpu_type="NVIDIA A40",
                gpu_count=1,
                hourly_cost_usd=self.config.hourly_cost_usd,
                pricing_source="vultr-plan-api",
                registered_at=requested_at,
                metadata={"code_commit": commit, "job_id": job_id},
            )
            self._event(allocation_id, "provision_requested", requested_at)
            self._event(allocation_id, "created", _provider_time(instance) or _utcnow())
            instance = self._wait_for_instance(instance_id, deadline)
            host = str(instance.get("main_ip", ""))
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                raise VultrError("ready instance has no public IPv4 address")
            if address.version != 4 or not address.is_global:
                raise VultrError("ready instance has no public IPv4 address")
            self._event(allocation_id, "ready")

            self.runner.wait_ready(
                host,
                timeout=min(self.config.ready_timeout_seconds, self._remaining(deadline)),
                poll_seconds=self.config.poll_seconds,
            )
            self.runner.upload_commit(host, commit)
            self.runner.upload_environment(
                host,
                {
                    "HF_TOKEN": _env("HF_TOKEN"),
                    "HF_ADAPTER_REPO": _env("HF_ADAPTER_REPO"),
                    "DAPTA_REMOTE_WORKER": "1",
                },
            )
            self._event(allocation_id, "workload_started")
            try:
                self.runner.prepare(
                    host,
                    job_id,
                    skill_id=skill_id,
                    limit=limit,
                    timeout=self._remaining(deadline),
                )
            except BaseException:
                self._save_validation(host, job_id, validation_report_path, deadline, required=False)
                raise
            validation_report = self._save_validation(
                host, job_id, validation_report_path, deadline, required=True
            )
            if not validation_report.get("passed"):
                raise ValueError("remote dataset validation did not pass")
            self.ledger.record_run_event(
                run_id=run_id,
                stage="dataset_validation",
                status="completed",
                metadata={
                    "validator_version": validation_report.get("validator_version"),
                    "counts": validation_report.get("counts", {}),
                    "hashes": validation_report.get("hashes", {}),
                },
            )
            self.runner.train(
                host,
                job_id,
                version,
                skill_id=skill_id,
                smoke=smoke,
                timeout=self._remaining(deadline),
            )
            self._event(allocation_id, "workload_finished")
            return {
                "instance_id": instance_id,
                "allocation_id": allocation_id,
                "status": "completed",
                "validation_passed": True,
            }
        except BaseException as exc:
            error = exc
            if allocation_id:
                self._event(allocation_id, "failed", metadata={"error_type": type(exc).__name__})
            raise
        finally:
            try:
                if instance is not None:
                    instance_id = str(instance["id"])
                    if allocation_id:
                        self._event(allocation_id, "destroy_requested")
                    self.client.delete_instance(instance_id)
                    self._wait_for_deletion(instance_id)
                    if allocation_id:
                        self._event(allocation_id, "destroyed")
                        self.ledger.estimate_destroyed_compute_cost(allocation_id)
            except BaseException as destroy_error:
                if error is not None:
                    raise RuntimeError(
                        f"worker failed and automatic Vultr deletion also failed: {type(destroy_error).__name__}"
                    ) from destroy_error
                raise
            finally:
                self.runner.close()

    def _preflight(self) -> dict[str, Any]:
        if self.config.plan_id not in self.client.available_plans(self.config.region):
            raise VultrError("configured GPU plan is unavailable in the configured region")
        plan = next((item for item in self.client.plans() if item.get("id") == self.config.plan_id), None)
        if plan is None:
            raise VultrError("configured GPU plan was not returned by Vultr")
        if Decimal(str(plan.get("hourly_cost"))) != self.config.hourly_cost_usd:
            raise VultrError("configured Vultr hourly price no longer matches the plan API")
        active = [item for item in self.client.instances() if MANAGED_TAG in item.get("tags", [])]
        if active:
            raise VultrError("another Dapta-managed Vultr worker already exists")
        if self.config.projected_cost_usd > self.config.max_job_cost_usd:
            raise VultrError("projected worker cost exceeds the per-job budget")
        return plan

    def _create_payload(self, job_id: str, label: str) -> dict[str, Any]:
        return {
            "region": self.config.region,
            "plan": self.config.plan_id,
            "os_id": self.config.os_id,
            "label": label,
            "hostname": label,
            "sshkey_id": [self.config.ssh_key_id],
            "tags": [MANAGED_TAG, f"dapta-job:{job_id}"],
            "backups": "disabled",
            "activation_email": False,
            "enable_ipv6": False,
        }

    def _wait_for_instance(self, instance_id: str, deadline: float) -> dict[str, Any]:
        ready_deadline = min(deadline, self.monotonic() + self.config.ready_timeout_seconds)
        while True:
            instance = self.client.get_instance(instance_id)
            if instance is None:
                raise VultrError("Vultr instance disappeared while provisioning")
            if (
                instance.get("status") == "active"
                and instance.get("power_status") == "running"
                and instance.get("server_status") in {None, "ok"}
            ):
                return instance
            if self.monotonic() >= ready_deadline:
                raise TimeoutError("Vultr instance did not become ready")
            self.sleep(self.config.poll_seconds)

    def _wait_for_deletion(self, instance_id: str) -> None:
        deadline = self.monotonic() + 300
        while self.client.get_instance(instance_id) is not None:
            if self.monotonic() >= deadline:
                raise TimeoutError("Vultr instance deletion was not confirmed")
            self.sleep(self.config.poll_seconds)

    def _remaining(self, deadline: float) -> int:
        remaining = int(deadline - self.monotonic())
        if remaining < 1:
            raise TimeoutError("Vultr job reached its wall-clock budget")
        return remaining

    def _save_validation(
        self,
        host: str,
        job_id: str,
        path: Path | None,
        deadline: float,
        *,
        required: bool,
    ) -> dict[str, Any]:
        try:
            report = self.runner.read_validation(
                host, job_id, timeout=min(60, self._remaining(deadline))
            )
        except Exception:
            if required:
                raise
            return {}
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    def _event(
        self,
        allocation_id: str,
        event_type: str,
        occurred_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.ledger.record_compute_event(
            allocation_id=allocation_id,
            event_type=event_type,
            occurred_at=occurred_at,
            metadata=metadata,
        )


@contextmanager
def worker_lock(path: Path = Path("/tmp/dapta-vultr-worker.lock")) -> Iterator[None]:
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another local Dapta worker command is running") from exc
        yield


def _provider_time(instance: Mapping[str, Any]) -> str | None:
    value = instance.get("date_created")
    return str(value) if value else None


def _safe_value(value: str, name: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


def _safe_commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("commit must be a full 40-character Git SHA")
    return value


def _skill_id(value: str) -> str:
    if value not in {"receipt-to-expense", "csv-to-dashboard"}:
        raise ValueError(f"unsupported skill: {value}")
    return value


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing environment variable: {name}")
    return value


def _safe_env(name: str) -> str:
    return _safe_value(_env(name), name)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--version", default="v1-smoke")
    parser.add_argument(
        "--skill",
        choices=("receipt-to-expense", "csv-to-dashboard"),
        default="receipt-to-expense",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--confirm-spend")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--ledger", type=Path, default=Path("artifacts/dapta-ledger.sqlite3"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/jobs"))
    args = parser.parse_args()

    load_env(args.env)
    config = WorkerConfig.from_env()
    ledger = UsageLedger(args.ledger)
    client = VultrClient(config.api_key)
    runner = SshRunner(config, Path.cwd())
    worker = VultrWorker(config, client, ledger, runner)
    if args.command == "plan":
        print(
            json.dumps(
                worker.plan(
                    args.job_id,
                    args.commit,
                    skill_id=args.skill,
                    limit=args.limit,
                ),
                indent=2,
            )
        )
        runner.close()
        return

    run_id = args.run_id or f"vultr-{args.job_id}"
    ledger.create_run(
        run_id=run_id,
        run_type="adapter_training",
        job_id=args.job_id,
        skill_id=args.skill,
        metadata={"code_commit": args.commit, "dataset_limit": args.limit},
    )
    with worker_lock():
        result = worker.run(
            run_id=run_id,
            job_id=args.job_id,
            commit=args.commit,
            version=args.version,
            skill_id=args.skill,
            smoke=args.smoke,
            limit=args.limit,
            confirmation=args.confirm_spend or "",
            validation_report_path=args.artifacts / args.job_id / "validation.json",
        )
    print(json.dumps({**result, "usage": ledger.run_summary(run_id)}, indent=2))


if __name__ == "__main__":
    main()
