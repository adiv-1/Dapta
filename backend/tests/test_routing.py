import json
import unittest
from unittest.mock import MagicMock, patch

from dapta.routing import WorkflowRouter


class RoutingTests(unittest.TestCase):
    def test_jev_receives_registry_and_returns_its_actual_decision(self):
        adapters = [{"adapter_id": "nda-v1", "skill_id": "nda-intake", "metadata": {"name": "NDA Intake"}}]
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": json.dumps({
                "decision": "existing_adapter", "adapter_id": "nda-v1",
                "confidence": .98, "reason": "The NDA specialist matches.",
            })}}],
        }).encode()
        router = WorkflowRouter(api_key="test-key", base_url="https://jev.test", model="test-model")
        with patch("dapta.routing.urllib.request.urlopen", return_value=response) as request:
            decision = router.decide("Extract an NDA", iter(adapters))
        self.assertEqual(decision.source, "jev")
        self.assertEqual(decision.adapter_id, "nda-v1")
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(json.loads(payload["messages"][1]["content"])["adapters"], adapters)

    def test_fallback_does_not_claim_a_match_below_threshold(self):
        router = WorkflowRouter(api_key="test-key")
        with patch.object(router, "_jev", side_effect=RuntimeError("provider unavailable")):
            decision = router.decide("Sort parcels by size", [{"adapter_id": "nda-v1", "skill_id": "nda-intake"}])
        self.assertEqual(decision.decision, "create_adapter")
        self.assertEqual(decision.source, "registry_fallback")
        self.assertEqual(decision.reason, "No registered adapter met the match threshold")


if __name__ == "__main__":
    unittest.main()
