import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dapta.preflight import _sanitize, check_configuration, load_env


class PreflightTests(unittest.TestCase):
    def test_remote_runtime_dependencies_are_exactly_pinned(self):
        root = Path(__file__).resolve().parents[2]
        for name in ("requirements-data.txt", "requirements.txt"):
            requirements = (root / "backend" / name).read_text(encoding="utf-8").splitlines()
            for requirement in requirements:
                if requirement and not requirement.startswith("#"):
                    self.assertIn("==", requirement, f"unpinned dependency: {requirement}")

    def test_env_file_does_not_override_process_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("EXISTING=file\nNEW_VALUE=loaded\n", encoding="utf-8")
            with patch.dict(os.environ, {"EXISTING": "process"}, clear=True):
                load_env(path)
                self.assertEqual(os.environ["EXISTING"], "process")
                self.assertEqual(os.environ["NEW_VALUE"], "loaded")

    def test_errors_redact_bearer_and_provider_tokens(self):
        value = "Bearer secret-value hf_abcdef sk-example jv_example tsf_example"
        sanitized = _sanitize(value)
        self.assertNotIn("secret-value", sanitized)
        self.assertNotIn("abcdef", sanitized)
        self.assertNotIn("sk-example", sanitized)
        self.assertEqual(sanitized.count("[redacted]"), 5)

    def test_configuration_requires_hourly_cost(self):
        complete = {
            "VULTR_API_KEY": "x",
            "VULTR_REGION": "ewr",
            "VULTR_GPU_PLAN_ID": "plan",
            "VULTR_OS_ID": "1743",
            "VULTR_SSH_KEY_ID": "key",
            "VULTR_SSH_PRIVATE_KEY_PATH": "/tmp/id_ed25519",
            "HF_TOKEN": "x",
            "HF_ADAPTER_REPO": "owner/repo",
            "OPENROUTER_API_KEY": "x",
            "OPENROUTER_FACTORY_MODEL": "factory",
            "OPENROUTER_CRITIC_MODEL": "critic",
            "JEV_API_KEY": "x",
            "JEV_BASE_URL": "https://example.com",
            "JEV_MODEL": "jev-latest",
        }
        with patch.dict(os.environ, complete, clear=True):
            result = check_configuration()
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.details["missing"], ["VULTR_HOURLY_COST_USD"])


if __name__ == "__main__":
    unittest.main()
