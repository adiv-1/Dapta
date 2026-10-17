"""Receipt-skill implementations of the orchestrator's stage interfaces."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .ledger import UsageLedger
from .receipts import PREPROCESSING_VERSION, load_receipt_contract, load_receipt_sources
from .train import CANDIDATE_PREFIX, MODEL_ID
from .vultr import CONFIRMATION, SshRunner, VultrClient, VultrWorker, WorkerConfig, worker_lock


RECEIPT_SKILL_ID = "receipt-to-expense"
SKILLS_PATH = Path("research/adapter-assets/skills")
SOURCE_MANIFEST_PATH = Path("research/adapter-assets/sources.lock.json")
CONTRACT_NAMES = ("skill_spec", "output_schema", "evaluation_plan")
CONTRACT_FILES = (*CONTRACT_NAMES, "review")
APPROVED = "approved_for_implementation"
SKILL_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")

WorkerFactory = Callable[[Callable[[str], None]], VultrWorker]


class ApprovedContracts:
    """Reuses the checked-in receipt contract once its review approves it, so the Architect is not re-run."""

    def __init__(self, skills_dir: Path = SKILLS_PATH):
        self.skills_dir = skills_dir

    def get(self, skill_id: str) -> Mapping[str, Any] | None:
        # Only the receipt contract has a loader; other skills go through the Skill Architect.
        if skill_id != RECEIPT_SKILL_ID or not SKILL_ID.fullmatch(skill_id):
            return None
        directory = self.skills_dir / skill_id
        review_path = directory / "review.json"
        if not review_path.is_file():
            return None
        if json.loads(review_path.read_text(encoding="utf-8")).get("status") != APPROVED:
            return None
        # An approved contract that fails its own hash checks is an error, not a fallback.
        artifacts = load_receipt_contract(directory)
        return {name: artifacts[name] for name in CONTRACT_NAMES}


class RemoteReceiptDataset:
    """Pins CORD v2 + SROIE at the integration commit; rows are materialized and validated on the worker."""

    # The pinned sources produce the same rows every round, so a second round cannot help.
    max_rounds = 1

    def __init__(
        self,
        *,
        repo_dir: Path | None = None,
        limit_per_source_split: int | None = None,
    ):
        if limit_per_source_split is not None and limit_per_source_split < 1:
            raise ValueError("limit_per_source_split must be positive")
        self.repo_dir = repo_dir or Path.cwd()
        self.limit_per_source_split = limit_per_source_split

    def build(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        skill_contract: Mapping[str, Any],
        integration: Mapping[str, Any],
        feedback: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        _require_receipt_job(job)
        if round > self.max_rounds:
            raise ValueError("the pinned CORD/SROIE dataset is identical every round")
        commit = integration["commit_sha"]
        with self._at_commit(commit) as (contract_dir, manifest):
            approved = load_receipt_contract(contract_dir)
            if any(dict(skill_contract[name]) != approved[name] for name in CONTRACT_NAMES):
                raise ValueError(f"job contract differs from the contract at commit {commit}")
            sources = _pinned_sources(manifest)
        return {
            "source": "receipt-cord-sroie-remote",
            "preprocessing_version": PREPROCESSING_VERSION,
            "commit_sha": commit,
            "sources": sources,
            "limit_per_source_split": self.limit_per_source_split,
            "materialized_on": "training_worker",
        }

    def validate(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        _require_receipt_job(job)
        commit = integration["commit_sha"]
        with self._at_commit(commit) as (contract_dir, manifest):
            load_receipt_contract(contract_dir)
            sources_match = dataset.get("sources") == _pinned_sources(manifest)
        commit_matches = dataset.get("commit_sha") == commit
        return {
            "passed": sources_match and commit_matches,
            "checks": {
                "contract_approved_at_commit": True,
                "sources_pinned_at_commit": True,
                "dataset_built_from_integration_commit": commit_matches,
                "sources_unchanged_since_build": sources_match,
            },
            # Row-level checks need the images, which only exist on the worker; it refuses
            # to train unless they pass.
            "deferred_to": "training_worker",
            "deferred_checks": [
                "schema",
                "arithmetic",
                "duplicates",
                "split_leakage",
                "frozen_test_hash",
            ],
        }

    @contextmanager
    def _at_commit(self, commit: str) -> Iterator[tuple[Path, Path]]:
        # The worker trains on `git archive <commit>`, so read the contract from that commit
        # rather than from the working tree.
        with tempfile.TemporaryDirectory(prefix="dapta-commit-") as temporary:
            root = Path(temporary)
            contract_dir = root / "contract"
            contract_dir.mkdir()
            for name in CONTRACT_FILES:
                path = SKILLS_PATH / RECEIPT_SKILL_ID / f"{name}.json"
                (contract_dir / f"{name}.json").write_bytes(self._show(commit, path))
            manifest = root / "sources.lock.json"
            manifest.write_bytes(self._show(commit, SOURCE_MANIFEST_PATH))
            yield contract_dir, manifest

    def _show(self, commit: str, path: Path) -> bytes:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path.as_posix()}"],
            cwd=self.repo_dir,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ValueError(f"{path} is not present at commit {commit}")
        return result.stdout


class VultrTrainer:
    """Runs the guarded Vultr lifecycle for one dataset round and reports the uploaded adapter."""

    def __init__(
        self,
        worker_factory: WorkerFactory,
        *,
        hf_repo: str | None = None,
        base_version: str = "v1",
        artifacts_dir: Path = Path("artifacts/jobs"),
        lock: Callable[[], AbstractContextManager[None]] = worker_lock,
    ):
        self.worker_factory = worker_factory
        self.hf_repo = hf_repo
        self.base_version = base_version
        self.artifacts_dir = artifacts_dir
        self.lock = lock

    def preflight(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
    ) -> None:
        _require_receipt_job(job)
        self._hf_settings()
        with self.lock():
            pass
        # plan() is read-only: it checks the commit, plan availability, price and any live worker.
        worker = self.worker_factory(lambda _: None)
        try:
            worker.plan(
                _worker_job_id(job, round),
                integration["commit_sha"],
                limit=dataset.get("limit_per_source_split"),
            )
        finally:
            worker.runner.close()

    def train(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
        on_progress: Callable[[str], None],
    ) -> Mapping[str, Any]:
        _require_receipt_job(job)
        hf_repo, hf_token = self._hf_settings()
        # Unique per job and round, so no job can overwrite another job's uploaded adapter.
        version = f"{self.base_version}-{job['job_id'].replace('-', '')[:12]}-r{round}"
        worker_job_id = _worker_job_id(job, round)
        report_path = self.artifacts_dir / worker_job_id / "validation.json"

        def on_event(event: str) -> None:
            if event == "dataset_validated":
                on_progress("training")

        worker = self.worker_factory(on_event)
        with self.lock():
            result = worker.run(
                run_id=job["job_id"],
                job_id=worker_job_id,
                commit=integration["commit_sha"],
                version=version,
                limit=dataset.get("limit_per_source_split"),
                # The orchestrator only calls train() after a persisted spend approval.
                confirmation=CONFIRMATION,
                validation_report_path=report_path,
                hf_repo=hf_repo,
                hf_token=hf_token,
            )
        validation = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "version": version,
            "base_model": MODEL_ID,
            # A candidate until publish promotes it; consumers resolve adapters via the registry.
            "artifact_uri": f"hf://{hf_repo}/{CANDIDATE_PREFIX}/{RECEIPT_SKILL_ID}/{version}",
            "worker_job_id": worker_job_id,
            "instance_id": result["instance_id"],
            "allocation_id": result["allocation_id"],
            "remote_validation": {
                "passed": validation.get("passed"),
                "validator_version": validation.get("validator_version"),
                "counts": validation.get("counts", {}),
                "hashes": validation.get("hashes", {}),
            },
        }

    def _hf_settings(self) -> tuple[str, str]:
        # One source of truth: the same repo is checked, uploaded to, and recorded.
        hf_repo = (self.hf_repo or os.environ.get("HF_ADAPTER_REPO", "")).strip()
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_repo:
            raise ValueError("HF_ADAPTER_REPO is required; the worker VM is deleted after training")
        if not hf_token:
            raise ValueError("HF_TOKEN is required to upload the adapter")
        return hf_repo, hf_token


class HuggingFacePromoter:
    """Copies a candidate adapter folder to its published path in the same Hugging Face repo."""

    def __init__(self, api: Any = None):
        self.api = api

    def promote(self, *, job: Mapping[str, Any], training: Mapping[str, Any]) -> str:
        repo_id, source = _parse_hf_uri(training["artifact_uri"])
        prefix = f"{CANDIDATE_PREFIX}/"
        if not source.startswith(prefix):
            raise ValueError(f"adapter is not a candidate upload: {training['artifact_uri']}")
        destination = source[len(prefix):]
        api = self.api or _hf_api()
        with tempfile.TemporaryDirectory(prefix="dapta-promote-") as temporary:
            local = Path(
                api.snapshot_download(
                    repo_id=repo_id, allow_patterns=[f"{source}/*"], local_dir=temporary
                )
            )
            folder = local / source
            if not folder.is_dir() or not any(folder.iterdir()):
                raise ValueError(f"candidate adapter has no files: {training['artifact_uri']}")
            # Re-uploading identical files is a no-op commit, so retries are safe.
            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(folder),
                path_in_repo=destination,
                commit_message=f"Publish {destination} from job {job['job_id']}",
            )
        return f"hf://{repo_id}/{destination}"


def vultr_worker_factory(ledger: UsageLedger, repo_dir: Path | None = None) -> WorkerFactory:
    def factory(on_event: Callable[[str], None]) -> VultrWorker:
        config = WorkerConfig.from_env()
        return VultrWorker(
            config,
            VultrClient(config.api_key),
            ledger,
            SshRunner(config, repo_dir or Path.cwd()),
            on_event=on_event,
        )

    return factory


def _parse_hf_uri(uri: str) -> tuple[str, str]:
    parts = uri.removeprefix("hf://").split("/", 2)
    if not uri.startswith("hf://") or len(parts) != 3 or not all(parts):
        raise ValueError(f"not an hf://owner/repo/path URI: {uri}")
    return f"{parts[0]}/{parts[1]}", parts[2]


class _HubClient:
    def __init__(self, token: str):
        from huggingface_hub import HfApi

        self.token = token
        self.api = HfApi(token=token)

    def snapshot_download(self, **kwargs: Any) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(token=self.token, **kwargs)

    def upload_folder(self, **kwargs: Any) -> Any:
        return self.api.upload_folder(**kwargs)


def _hf_api() -> _HubClient:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("HF_TOKEN is required to promote the adapter")
    return _HubClient(token)


def _pinned_sources(manifest: Path) -> dict[str, dict[str, str]]:
    return {
        source.id: {"source": source.source, "revision": source.revision}
        for source in load_receipt_sources(manifest).values()
    }


def _worker_job_id(job: Mapping[str, Any], round: int) -> str:
    return f"{job['job_id']}-r{round}"


def _require_receipt_job(job: Mapping[str, Any]) -> None:
    if job.get("skill_id") != RECEIPT_SKILL_ID:
        raise ValueError(f"this backend only handles {RECEIPT_SKILL_ID} jobs")
