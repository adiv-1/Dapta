"""The Qwen worker on vLLM: adapters, streaming metrics, sleep/wake for training, worker API.

A local fake implements the vLLM OpenAI-server endpoints the runtime uses, so no GPU is needed.
"""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from dapta.model_server import GenerateRequest, generation_messages
from dapta.vllm_backend import VllmError, VllmRuntime, adapter_alias

NDA = "nda-intake/qwen3-vl-8b-v1"
SECRET = "CONFIDENTIAL-NDA-TEXT-91c2"


class FakeVllm:
    def __init__(self, *, reply='{"effective_date": "2026-01-01"}', usage=True, fail_chat=None):
        self.requests: list[tuple[str, str, dict]] = []
        self.sleeping = False
        self.adapters: dict[str, str] = {}
        self.reply, self.usage, self.fail_chat = reply, usage, fail_chat
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _body(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                return json.loads(raw) if raw else {}

            def _json(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.requests.append(("GET", self.path, {}))
                if self.path == "/health":
                    return self._json(200, {})
                if self.path == "/is_sleeping":
                    return self._json(200, {"is_sleeping": fake.sleeping})
                self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                body = self._body()
                fake.requests.append(("POST", self.path, body))
                if self.path.startswith("/sleep"):
                    fake.sleeping = True
                    return self._json(200, {})
                if self.path == "/wake_up":
                    fake.sleeping = False
                    return self._json(200, {})
                if self.path == "/collective_rpc":
                    return self._json(200, {"results": [None]})
                if self.path == "/v1/load_lora_adapter":
                    fake.adapters[body["lora_name"]] = body["lora_path"]
                    return self._json(200, {})
                if self.path == "/v1/chat/completions":
                    if fake.fail_chat:
                        return self._json(fake.fail_chat, {"error": {"message": f"echo: {SECRET}"}})
                    if body["model"] not in fake.adapters:
                        return self._json(404, {"error": {"message": "model not found"}})
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    half = len(fake.reply) // 2
                    for piece in (fake.reply[:half], fake.reply[half:]):
                        chunk = {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                    if fake.usage:
                        usage = {"choices": [], "usage": {"prompt_tokens": 321, "completion_tokens": 17}}
                        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                self._json(404, {"error": {"message": "not found"}})

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def paths(self, method=None):
        return [path for verb, path, _ in self.requests if method in (None, verb)]

    def close(self):
        self.server.shutdown()


def fake_download(repo_id, adapter_id, root):
    path = Path(root) / adapter_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text('{"r": 16}')
    return path


def request(adapter_id=NDA):
    return GenerateRequest(run_id="run-1", workflow="Extract the NDA", document_text=SECRET, adapter_id=adapter_id)


class VllmRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.downloads: list[str] = []
        self.env = patch.dict(os.environ, {"QWEN_ADAPTER_SUBFOLDER": "", "QWEN_MAX_NEW_TOKENS": "1200",
                                           "VLLM_READY_TIMEOUT_SECONDS": "5"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def downloader(self, repo_id, adapter_id, root):
        self.downloads.append(f"{repo_id}:{adapter_id}")
        path = Path(root) / adapter_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text('{"r": 16}')
        return path

    def runtime(self, fake, **kwargs):
        runtime = VllmRuntime(build_messages=generation_messages, base_url=fake.url, api_key="",
                              adapter_repo="adityave/dapta-lora-adapters", adapter_root=self.temp.name,
                              downloader=self.downloader, timeout=10, **kwargs)
        runtime.load()
        return runtime

    def test_generation_uses_the_adapter_prompt_greedy_decoding_and_reported_usage(self):
        fake = FakeVllm()
        try:
            result = self.runtime(fake).generate(request())
        finally:
            fake.close()
        chat = next(body for verb, path, body in fake.requests if path == "/v1/chat/completions")
        self.assertEqual(chat["model"], adapter_alias(NDA))
        self.assertEqual(chat["messages"], generation_messages(request()), "same prompt as the transformers path")
        self.assertEqual((chat["temperature"], chat["max_tokens"], chat["stream"]), (0, 1200, True))
        self.assertEqual(result["output"], {"effective_date": "2026-01-01"})
        self.assertEqual((result["input_tokens"], result["output_tokens"]), (321, 17))
        self.assertIsInstance(result["ttft_ms"], int, "TTFT is measured from the first streamed token")
        self.assertEqual(result["finish_reason"], "stop")

    def test_adapters_are_downloaded_and_registered_once(self):
        fake = FakeVllm()
        try:
            runtime = self.runtime(fake)
            runtime.generate(request())
            runtime.generate(request())
        finally:
            fake.close()
        self.assertEqual(self.downloads, [f"adityave/dapta-lora-adapters:{NDA}"])
        loads = [body for verb, path, body in fake.requests if path == "/v1/load_lora_adapter"]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0]["lora_path"], str(Path(self.temp.name).resolve() / NDA))
        self.assertTrue(loads[0]["load_inplace"])

    def test_missing_usage_stays_missing_instead_of_zero(self):
        fake = FakeVllm(usage=False)
        try:
            result = self.runtime(fake).generate(request())
        finally:
            fake.close()
        self.assertIsNone(result["input_tokens"])
        self.assertIsNone(result["output_tokens"])

    def test_unload_sleeps_vllm_for_training_and_load_wakes_it_and_reregisters_adapters(self):
        fake = FakeVllm()
        try:
            runtime = self.runtime(fake)
            runtime.generate(request())
            runtime.unload()
            self.assertTrue(fake.sleeping)
            self.assertIn("/sleep?level=1", fake.paths("POST"))
            with self.assertRaisesRegex(RuntimeError, "model is not loaded"):
                runtime.generate(request())
            runtime.load()
            self.assertFalse(fake.sleeping)
            self.assertIn("/wake_up", fake.paths("POST"))
            runtime.generate(request())
        finally:
            fake.close()
        loads = [path for path in fake.paths("POST") if path == "/v1/load_lora_adapter"]
        self.assertEqual(len(loads), 2, "re-registered after waking, not assumed to survive sleep")
        self.assertEqual(len(self.downloads), 1, "the download is reused")

    def test_level_two_sleep_reloads_weights_after_waking(self):
        fake = FakeVllm()
        try:
            with patch.dict(os.environ, {"VLLM_SLEEP_LEVEL": "2"}):
                runtime = self.runtime(fake)
                runtime.unload()
                runtime.load()
        finally:
            fake.close()
        posts = fake.paths("POST")
        self.assertIn("/sleep?level=2", posts)
        self.assertEqual(posts[posts.index("/wake_up") + 1], "/collective_rpc")
        reload = next(body for verb, path, body in fake.requests if path == "/collective_rpc")
        self.assertEqual(reload, {"method": "reload_weights"})

    def test_default_adapter_is_registered_at_load(self):
        fake = FakeVllm()
        try:
            with patch.dict(os.environ, {"QWEN_ADAPTER_SUBFOLDER": NDA}):
                runtime = self.runtime(fake)
        finally:
            fake.close()
        self.assertEqual(runtime.loaded_adapters, {NDA: adapter_alias(NDA)})

    def test_invalid_adapter_ids_are_rejected_before_any_request(self):
        fake = FakeVllm()
        try:
            runtime = self.runtime(fake)
            for bad in ("../etc", "a", "x/../../y"):
                with self.assertRaises(ValueError):
                    runtime.generate(request(bad))
        finally:
            fake.close()
        self.assertNotIn("/v1/load_lora_adapter", fake.paths("POST"))

    def test_vllm_errors_are_reported_without_request_content_in_sentry(self):
        fake = FakeVllm(fail_chat=400)
        try:
            runtime = self.runtime(fake)
            with self.assertRaises(VllmError) as caught:
                runtime.generate(request())
        finally:
            fake.close()
        self.assertIn("HTTP 400", str(caught.exception))
        self.assertEqual(caught.exception.dapta_sentry_message, "vLLM /v1/chat/completions failed with HTTP 400")
        self.assertNotIn(SECRET, caught.exception.dapta_sentry_message)

    def test_unreachable_vllm_fails_clearly(self):
        runtime = VllmRuntime(build_messages=generation_messages, base_url="http://127.0.0.1:9", api_key="",
                              adapter_repo="repo", adapter_root=self.temp.name, timeout=1)
        with patch.dict(os.environ, {"VLLM_READY_TIMEOUT_SECONDS": "0"}):
            with self.assertRaisesRegex(VllmError, "unreachable"):
                runtime.load()


class WorkerOnVllmTests(unittest.TestCase):
    """The worker API the controller uses is unchanged on the vLLM backend."""

    def test_generate_health_and_lifecycle_through_the_worker_app(self):
        from fastapi.testclient import TestClient

        from dapta import model_server

        fake = FakeVllm()
        temp = tempfile.TemporaryDirectory()
        runtime = VllmRuntime(build_messages=generation_messages, base_url=fake.url, api_key="",
                              adapter_repo="adityave/dapta-lora-adapters", adapter_root=temp.name,
                              downloader=fake_download, timeout=10)
        try:
            with patch.object(model_server, "runtime", runtime), \
                    patch.dict(os.environ, {"DAPTA_QWEN_TOKEN": "worker-token", "QWEN_ADAPTER_SUBFOLDER": "",
                                            "VLLM_READY_TIMEOUT_SECONDS": "5"}):
                with TestClient(model_server.app) as client:  # lifespan loads the runtime
                    auth = {"Authorization": "Bearer worker-token"}
                    body = {"run_id": "r1", "workflow": "Extract the NDA", "document_text": SECRET, "adapter_id": NDA}
                    health = client.get("/health").json()
                    self.assertEqual((health["backend"], health["model_loaded"]), ("vllm", True))
                    self.assertEqual(client.post("/generate", json=body).status_code, 401, "auth unchanged")
                    result = client.post("/generate", json=body, headers=auth).json()
                    self.assertEqual(result["output"], {"effective_date": "2026-01-01"})
                    self.assertEqual(client.post("/admin/unload", headers=auth).json()["model_loaded"], False)
                    self.assertTrue(fake.sleeping)
                    self.assertEqual(client.post("/admin/load", headers=auth).json()["model_loaded"], True)
                    self.assertFalse(fake.sleeping)
        finally:
            fake.close()
            temp.cleanup()

    def test_backend_selection(self):
        from dapta import model_server

        with patch.dict(os.environ, {"DAPTA_QWEN_BACKEND": "transformers"}):
            self.assertEqual(model_server.create_runtime().backend, "transformers")
        with patch.dict(os.environ, {"DAPTA_QWEN_BACKEND": "vllm"}):
            self.assertEqual(model_server.create_runtime().backend, "vllm")
        with patch.dict(os.environ, {"DAPTA_QWEN_BACKEND": "tgi"}):
            with self.assertRaises(RuntimeError):
                model_server.create_runtime()


if __name__ == "__main__":
    unittest.main()
