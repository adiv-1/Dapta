import json
import unittest
from unittest.mock import MagicMock, patch

from dapta.routing import WorkflowRouter


class RoutingTests(unittest.TestCase):
    def test_jev_receives_registry_and_returns_its_actual_decision(self):
        adapters = [{"adapter_id": "nda-v1", "skill_id": "nda-intake", "metadata": {"name": "NDA Intake"}}]
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "answers": {"route": {"type": "choice", "choice": "adapter_0", "confidence": .98}},
        }).encode()
        router = WorkflowRouter(api_key="test-key", base_url="https://jev.test", model="test-model")
        with patch("dapta.routing.urllib.request.urlopen", return_value=response) as request:
            decision = router.decide("Extract an NDA", iter(adapters))
        self.assertEqual(decision.source, "jev")
        self.assertEqual(decision.adapter_id, "nda-v1")
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(request.call_args.args[0].full_url, "https://jev.test/v1/systemone")
        self.assertEqual(payload["state"]["registry"]["adapter_0"]["adapter_id"], "nda-v1")
        self.assertEqual(payload["questions"]["route"]["type"], "choice")
        self.assertIn("create_adapter", payload["questions"]["route"]["criteria"])
        self.assertNotIn("messages", payload)

    def test_low_confidence_jev_match_requires_user_confirmation(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "answers": {"route": {"choice": "adapter_0", "confidence": .6}},
        }).encode()
        with patch("dapta.routing.urllib.request.urlopen", return_value=response):
            decision = WorkflowRouter(api_key="test-key").decide(
                "Extract an NDA", [{"adapter_id": "nda-v1", "skill_id": "nda-intake"}]
            )
        self.assertEqual(decision.source, "jev")
        self.assertTrue(decision.requires_confirmation)

    def test_jev_new_and_unsupported_choices_map_without_fabricating_an_adapter(self):
        for choice in ("create_adapter", "unsupported"):
            with self.subTest(choice=choice):
                response = MagicMock()
                response.__enter__.return_value.read.return_value = json.dumps({
                    "answers": {"route": {"choice": choice, "confidence": .99}},
                }).encode()
                with patch("dapta.routing.urllib.request.urlopen", return_value=response):
                    decision = WorkflowRouter(api_key="test-key").decide("A new workflow", [])
                self.assertEqual(decision.source, "jev")
                self.assertEqual(decision.decision, choice)
                self.assertIsNone(decision.adapter_id)

    def test_fallback_does_not_claim_a_match_below_threshold(self):
        router = WorkflowRouter(api_key="test-key")
        with patch.object(router, "_jev", side_effect=RuntimeError("provider unavailable")):
            decision = router.decide("Sort parcels by size", [{"adapter_id": "nda-v1", "skill_id": "nda-intake"}])
        self.assertEqual(decision.decision, "create_adapter")
        self.assertEqual(decision.source, "registry_fallback")
        self.assertEqual(decision.reason, "No registered adapter met the match threshold")


if __name__ == "__main__":
    unittest.main()
