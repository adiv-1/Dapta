"""Real job payloads at every factory stage, shared with the frontend rendering tests.

The frontend fixture (frontend/tests/fixtures/factory-stages.json) is produced by the real JobStore and
orchestrator driven through a full run with one DEV repair round. Regenerate it with:
    DAPTA_WRITE_FIXTURES=1 PYTHONPATH=backend python -m unittest backend.tests.test_factory_stage_shapes
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from dapta.jobs import AdapterJobOrchestrator, JobStore
from dapta.ledger import UsageLedger

try:
    from .test_jobs import (
        FakeAgents, FakeDatasetSource, FakeEvaluator, FakeIntegrator, FakePublisher, FakeTrainer,
    )
except ImportError:  # unittest discover loads test modules without a package
    from test_jobs import (  # type: ignore[no-redef]
        FakeAgents, FakeDatasetSource, FakeEvaluator, FakeIntegrator, FakePublisher, FakeTrainer,
    )

FIXTURE = Path(__file__).resolve().parents[2] / "frontend" / "tests" / "fixtures" / "factory-stages.json"


def factory_snapshots(path: Path) -> list[dict]:
    store = JobStore(path)
    orchestrator = AdapterJobOrchestrator(
        store, UsageLedger(path), FakeAgents(), integrator=FakeIntegrator(),
        dataset_source=FakeDatasetSource(), trainer=FakeTrainer(),
        evaluator=FakeEvaluator(fail_first=True), publisher=FakePublisher(), max_dataset_rounds=2,
    )
    job = orchestrator.create_job(
        workflow="Synthetic: turn receipt text into an expense report.",
        idempotency_key="fixture-factory-stages",
        skill_id="expense-report",
        route_decision={"decision": "create_adapter", "confidence": 0.9, "reason": "synthetic fixture"},
    )
    snapshots = [job]
    while job["status"] not in {"ready", "failed"}:
        job = orchestrator.advance(job["job_id"])
        snapshots.append(job)
    return snapshots


class FactoryStageShapeTests(unittest.TestCase):
    def test_frontend_fixture_matches_the_real_factory_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshots = factory_snapshots(Path(temp) / "dapta.sqlite3")
        counts = [(job["status"], len(job["agent_outputs"])) for job in snapshots]
        # Every stage, including a DEV repair round (evaluating -> building_dataset) and publishing.
        self.assertEqual([status for status, _ in counts].count("building_dataset"), 2)
        self.assertEqual(counts[-1][0], "ready")

        if os.environ.get("DAPTA_WRITE_FIXTURES"):
            FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE.write_text(json.dumps(snapshots, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual([(job["status"], len(job["agent_outputs"])) for job in stored], counts,
                         "frontend fixture is stale; regenerate it (see module docstring)")


if __name__ == "__main__":
    unittest.main()
