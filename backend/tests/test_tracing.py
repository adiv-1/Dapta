"""Distributed tracing and privacy: React/proxy -> controller -> background comparison -> Qwen worker.

Sentry runs for real, with a transport that keeps envelopes in memory instead of sending them.
"""

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import sentry_sdk
from sentry_sdk.transport import Transport

from dapta import observability

TRACE_ID = "0123456789abcdef0123456789abcdef"
PARENT_SPAN = "89abcdef01234567"
INCOMING = {"sentry-trace": f"{TRACE_ID}-{PARENT_SPAN}-1",
            "baggage": f"sentry-trace_id={TRACE_ID},sentry-sample_rate=1.0,sentry-sampled=true"}
SECRET = "CONFIDENTIAL-DOCUMENT-CONTENT-7f3a"
DSN = "https://public@o0.ingest.sentry.io/0"


class CaptureTransport(Transport):
    def __init__(self, options=None):
        super().__init__(options)
        self.envelopes = []

    def capture_envelope(self, envelope):
        self.envelopes.append(envelope)

    def items(self, kind):
        found = []
        for envelope in self.envelopes:
            for item in envelope.items:
                if item.type == kind:
                    found.append(item.payload.json)
        return found

    def raw(self):
        return "".join(item.payload.get_bytes().decode("utf-8", "replace")
                       for envelope in self.envelopes for item in envelope.items)


def start_sentry(service="controller"):
    transport = CaptureTransport()
    sentry_sdk.get_global_scope().set_client(None)
    observability.initialize_sentry(service=service, dsn=DSN, transport=transport, profiles_sample_rate=0)
    return transport


def stop_sentry():
    sentry_sdk.flush()
    sentry_sdk.get_global_scope().set_client(None)


class QwenStub:
    """Local stand-in for the Qwen worker that records the headers it receives."""

    def __init__(self):
        self.headers = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                stub.headers.append({k.lower(): v for k, v in self.headers.items()})
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.dumps({"output": {"ok": True}, "input_tokens": 3, "output_tokens": 2}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class HelperTests(unittest.TestCase):
    def tearDown(self):
        stop_sentry()

    def test_inactive_sentry_propagates_nothing(self):
        sentry_sdk.get_global_scope().set_client(None)
        self.assertEqual(observability.trace_headers(), {})
        with observability.continued_transaction(INCOMING, op="x", name="y") as active:
            self.assertIsNone(active)

    def test_initialization_is_idempotent_and_tags_the_service(self):
        transport = start_sentry(service="qwen-worker")
        self.assertTrue(observability.initialize_sentry(service="controller"))
        sentry_sdk.capture_message("probe")
        sentry_sdk.flush()
        [event] = transport.items("event")
        self.assertEqual(event["tags"]["dapta.service"], "qwen-worker")

    def test_representative_cpu_workload_emits_an_interpretable_profile(self):
        transport = CaptureTransport()
        sentry_sdk.get_global_scope().set_client(None)
        observability.initialize_sentry(
            dsn=DSN, transport=transport, traces_sample_rate=1.0, profiles_sample_rate=1.0,
        )
        with sentry_sdk.start_transaction(op="dapta.profile", name="JSON response construction"):
            deadline = time.perf_counter() + 0.25
            while time.perf_counter() < deadline:
                json.dumps({"runs": [{"id": index, "status": "completed"} for index in range(500)]})
        sentry_sdk.flush()

        [profile] = [item for envelope in transport.envelopes for item in envelope.items
                     if item.type == "profile"]
        payload = profile.payload.json["profile"]
        self.assertGreater(len(payload["samples"]), 0)
        self.assertGreater(len(payload["frames"]), 0)

    def test_continued_transaction_keeps_the_trace_in_another_thread(self):
        transport = start_sentry()
        with sentry_sdk.start_transaction(sentry_sdk.continue_trace(INCOMING, op="http.server", name="request")):
            captured = observability.trace_headers()
        self.assertTrue(captured["sentry-trace"].startswith(TRACE_ID))

        def background():
            with observability.continued_transaction(captured, op="dapta.comparison", name="background"):
                with observability.span("gen_ai.invoke_agent", "child"):
                    self.child_headers = observability.trace_headers()

        worker = threading.Thread(target=background)
        worker.start()
        worker.join()
        sentry_sdk.flush()
        background_tx = next(t for t in transport.items("transaction") if t["transaction"] == "background")
        request_tx = next(t for t in transport.items("transaction") if t["transaction"] == "request")
        self.assertEqual(background_tx["contexts"]["trace"]["trace_id"], TRACE_ID)
        self.assertEqual(background_tx["contexts"]["trace"]["parent_span_id"], request_tx["contexts"]["trace"]["span_id"])
        [child] = background_tx["spans"]
        self.assertEqual(self.child_headers["sentry-trace"].split("-")[:2], [TRACE_ID, child["span_id"]])

    def test_factory_waits_are_separate_transactions_in_one_job_trace(self):
        transport = start_sentry()
        job_id = "job-segment-test"
        observability.emit_job_root(job_id, skill_id="synthetic")
        with observability.FactoryStage(
            job_id, "skill_integration", round=1, op="dapta.factory.stage",
            name="factory skill_integration",
        ) as stage:
            with observability.span("agent.devin.integrate", "Create Devin integration"):
                pass
            stage.suspend("devin_session")
            stage.resume("devin_completed")
            with observability.span("db.sql.transaction", "complete stage"):
                pass
        sentry_sdk.flush()

        transactions = transport.items("transaction")
        root = next(item for item in transactions if item["transaction"] == "factory job")
        segments = [item for item in transactions if item["transaction"].startswith("factory skill_integration")]
        self.assertEqual(len(segments), 2)
        self.assertEqual({item["contexts"]["trace"]["trace_id"] for item in segments},
                         {root["contexts"]["trace"]["trace_id"]})
        self.assertTrue(all(item["contexts"]["trace"]["parent_span_id"]
                            == root["contexts"]["trace"]["span_id"] for item in segments))

    def test_openrouter_agent_and_call_spans_include_reported_usage(self):
        from dapta.ledger import UsageLedger
        from dapta.openrouter import FactoryAgents, OpenRouterClient

        class Transport:
            def request(self, method, url, *, headers, payload=None):
                del method, url, headers, payload
                return {
                    "id": None,
                    "model": "moonshotai/kimi-k3",
                    "choices": [{"message": {"content": '{"result":"ok"}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15, "cost": "0.001"},
                }

        transport = start_sentry()
        with tempfile.TemporaryDirectory() as temp:
            ledger = UsageLedger(Path(temp) / "ledger.sqlite3")
            ledger.create_run(run_id="job-ai", run_type="adapter_build", job_id="job-ai",
                              started_at="2026-10-18T00:00:00Z")
            client = OpenRouterClient(api_key="test", ledger=ledger, transport=Transport())
            agents = FactoryAgents(client, factory_model="moonshotai/kimi-k3", critic_model="critic")
            with sentry_sdk.start_transaction(op="test", name="factory AI test"):
                agents.architect(run_id="job-ai", workflow="synthetic")
        sentry_sdk.flush()

        transaction = next(item for item in transport.items("transaction")
                           if item["transaction"] == "factory AI test")
        agent = next(item for item in transaction["spans"] if item["op"] == "gen_ai.invoke_agent")
        call = next(item for item in transaction["spans"] if item["op"] == "gen_ai.chat")
        self.assertEqual(call["parent_span_id"], agent["span_id"])
        self.assertEqual(call["data"]["gen_ai.usage.input_tokens"], 10)
        self.assertEqual(call["data"]["gen_ai.usage.output_tokens"], 5)
        self.assertEqual(call["data"]["dapta.cost_usd"], 0.001)


class ComparisonTracingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.qwen = QwenStub()

    def tearDown(self):
        self.qwen.close()
        self.temp.cleanup()
        stop_sentry()

    def service(self):
        from dapta.comparisons import ComparisonService, ComparisonStore, QwenGateway
        from dapta.config import Settings
        from dapta.ledger import UsageLedger

        path = Path(self.temp.name) / "dapta.sqlite3"
        with patch.dict(os.environ, {"DAPTA_ENV": "test"}, clear=False):
            settings = Settings.from_env()
        return ComparisonService(store=ComparisonStore(path), ledger=UsageLedger(path), settings=settings,
                                 qwen=QwenGateway(url=self.qwen.url, token="worker-token"))

    def test_trace_survives_the_background_thread_pool_and_reaches_the_worker(self):
        transport = start_sentry()
        service = self.service()
        threads = []

        def background(function, *args):
            thread = threading.Thread(target=function, args=args)
            threads.append(thread)
            thread.start()

        adapter = {"adapter_id": "nda-intake/qwen3-vl-8b-v1", "skill_id": "nda-intake", "metadata": {}}
        with sentry_sdk.start_transaction(sentry_sdk.continue_trace(INCOMING, op="http.server", name="POST /v1/comparisons")):
            run = service.start(workflow="Summarise this NDA", document_text=SECRET, adapter=adapter,
                                idempotency_key="trace-test-0001", background=background)
        for thread in threads:
            thread.join(timeout=30)
        sentry_sdk.flush()

        transactions = {t["transaction"]: t for t in transport.items("transaction")}
        request_tx, comparison_tx = transactions["POST /v1/comparisons"], transactions["Dapta model comparison"]
        self.assertEqual(comparison_tx["contexts"]["trace"]["trace_id"], TRACE_ID)
        self.assertEqual(comparison_tx["contexts"]["trace"]["parent_span_id"], request_tx["contexts"]["trace"]["span_id"])
        qwen_agent = next(s for s in comparison_tx["spans"] if s["op"] == "gen_ai.invoke_agent")
        qwen_call = next(s for s in comparison_tx["spans"] if s["op"] == "gen_ai.chat")

        [received] = self.qwen.headers
        self.assertEqual(qwen_call["parent_span_id"], qwen_agent["span_id"])
        self.assertEqual(received["sentry-trace"].split("-")[:2], [TRACE_ID, qwen_call["span_id"]])
        self.assertIn(f"sentry-trace_id={TRACE_ID}", received["baggage"])
        self.assertEqual(received["authorization"], "Bearer worker-token", "worker auth unchanged")
        self.assertEqual(service.store.get(run["run_id"])["status"], "completed")
        self.assertNotIn(SECRET, transport.raw())

    def test_no_sentry_means_no_trace_headers_to_the_worker(self):
        sentry_sdk.get_global_scope().set_client(None)
        service = self.service()
        done = []
        adapter = {"adapter_id": "nda-intake/qwen3-vl-8b-v1", "metadata": {}}
        service.start(workflow="Summarise this NDA", document_text="text", adapter=adapter,
                      idempotency_key="trace-test-0002", background=lambda f, *a: done.append(f(*a)))
        self.assertNotIn("sentry-trace", self.qwen.headers[0])


class ServiceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()
        stop_sentry()

    def test_controller_continues_the_incoming_trace_without_sending_request_bodies(self):
        from fastapi.testclient import TestClient

        transport = start_sentry()
        environment = {"DAPTA_ENV": "test", "DAPTA_CONTROL_TOKEN": "control-token", "DAPTA_QWEN_URL": "",
                       "OPENROUTER_API_KEY": "", "JEV_API_KEY": "", "DEVIN_API_KEY": "", "SUPABASE_URL": "",
                       "DAPTA_FACTORY_AUTOSTART": "false"}
        with patch.dict(os.environ, environment, clear=False):
            from dapta.api import create_app

            client = TestClient(create_app(Path(self.temp.name) / "dapta.sqlite3"))
            response = client.post("/v1/comparisons", headers={**INCOMING, "X-Dapta-Token": "control-token",
                                                               "Idempotency-Key": "trace-test-0003"},
                                   json={"workflow": "Summarise this NDA", "document_text": SECRET,
                                         "adapter_id": "missing/adapter"})
            unauthorized = client.get("/v1/runs", headers=INCOMING)
        sentry_sdk.flush()
        self.assertEqual(response.status_code, 404, "API contract unchanged for an unknown adapter")
        self.assertEqual(unauthorized.status_code, 401, "authentication still enforced")
        traced = [t for t in transport.items("transaction") if "/v1/comparisons" in t["transaction"]]
        self.assertTrue(traced)
        self.assertEqual(traced[0]["contexts"]["trace"]["trace_id"], TRACE_ID)
        self.assertEqual(traced[0]["contexts"]["trace"]["parent_span_id"], PARENT_SPAN)
        self.assertEqual(traced[0]["tags"]["dapta.service"], "controller")
        raw = transport.raw()
        self.assertNotIn(SECRET, raw, "document text must never reach Sentry")
        self.assertNotIn("control-token", raw, "control token must be filtered")
        self.assertFalse(any(s["op"].startswith("middleware.") for s in traced[0]["spans"]), "no header-dumping middleware spans")

    def test_secret_header_values_are_scrubbed_from_any_text(self):
        event = {"spans": [{"description": "Headers({'x-dapta-token': 'abc', 'authorization': 'Bearer xyz.123'})"}],
                 "message": 'call with Authorization: "Bearer tok-9" done'}
        scrubbed = json.dumps(observability._scrub_event(event, {}))
        for leaked in ("abc", "xyz.123", "tok-9"):
            self.assertNotIn(leaked, scrubbed)

    def test_library_log_records_reach_sentry_logs_scrubbed(self):
        import logging

        transport = start_sentry()
        logging.getLogger("httpx").warning("retrying with x-dapta-token: abc123 and Authorization: Bearer xyz.987")
        logging.getLogger("httpx").warning("payload %s", {"document_text": SECRET})
        stop_sentry()
        raw = transport.raw()
        self.assertIn("retrying with", raw, "logs are still sent")
        for leaked in ("abc123", "xyz.987", SECRET):
            self.assertNotIn(leaked, raw)

    def test_full_chain_browser_to_controller_to_background_comparison_to_worker(self):
        """One trace, correct parents at every hop, through the real controller and worker apps."""
        from fastapi.testclient import TestClient

        from dapta.jobs import JobStore

        transport = start_sentry()
        qwen = QwenStub()
        database = Path(self.temp.name) / "dapta.sqlite3"
        JobStore(database).register_adapter(adapter_id="nda-intake/qwen3-vl-8b-v1", skill_id="nda-intake",
                                            version="v1", base_model="Qwen/Qwen3-VL-8B-Instruct",
                                            artifact_uri="hf://example/nda-intake/v1")
        environment = {"DAPTA_ENV": "test", "DAPTA_CONTROL_TOKEN": "control-token", "DAPTA_QWEN_URL": qwen.url,
                       "DAPTA_QWEN_TOKEN": "worker-token", "OPENROUTER_API_KEY": "", "JEV_API_KEY": "",
                       "DEVIN_API_KEY": "", "SUPABASE_URL": "", "DAPTA_FACTORY_AUTOSTART": "false"}
        try:
            with patch.dict(os.environ, environment, clear=False):
                from dapta.api import create_app

                client = TestClient(create_app(database))
                auth = {"X-Dapta-Token": "control-token"}
                started = client.post("/v1/comparisons", headers={**INCOMING, **auth, "Idempotency-Key": "trace-chain-01"},
                                      json={"workflow": "Summarise this NDA", "document_text": SECRET,
                                            "adapter_id": "nda-intake/qwen3-vl-8b-v1"})
                self.assertEqual(started.status_code, 202)
                run_id = started.json()["run_id"]
                deadline = time.monotonic() + 30
                while client.get(f"/v1/runs/{run_id}", headers=auth).json()["status"] not in ("completed", "failed"):
                    self.assertLess(time.monotonic(), deadline, "comparison did not finish")
                    time.sleep(0.05)
                # The run is marked completed inside the comparison transaction, which finishes just after.
                while not any(t["transaction"] == "Dapta model comparison" for t in transport.items("transaction")):
                    self.assertLess(time.monotonic(), deadline, "comparison transaction was not sent")
                    time.sleep(0.05)
        finally:
            qwen.close()

        # Hop 4: the worker app receives exactly the headers the controller sent.
        [sent_to_worker] = qwen.headers
        stop_sentry()
        worker_transport = start_sentry(service="qwen-worker")
        from dapta import model_server

        reply = {"output": {}, "input_tokens": 1, "output_tokens": 1, "latency_ms": 1, "ttft_ms": None}
        with patch.dict(os.environ, {"DAPTA_QWEN_TOKEN": "worker-token"}, clear=False), \
                patch.object(model_server.runtime, "generate", return_value=reply):
            TestClient(model_server.app).post(
                "/generate", json={"run_id": run_id, "workflow": "w", "document_text": SECRET, "adapter_id": "a/b"},
                headers={"sentry-trace": sent_to_worker["sentry-trace"], "baggage": sent_to_worker["baggage"],
                         "Authorization": sent_to_worker["authorization"]})
        sentry_sdk.flush()

        controller = {t["transaction"]: t for t in transport.items("transaction")}
        request_tx = next(t for name, t in controller.items() if name == "/v1/comparisons")
        comparison_tx = controller["Dapta model comparison"]
        qwen_agent = next(s for s in comparison_tx["spans"] if s["op"] == "gen_ai.invoke_agent")
        qwen_call = next(s for s in comparison_tx["spans"] if s["op"] == "gen_ai.chat")
        worker_tx = next(t for t in worker_transport.items("transaction") if t["transaction"] == "/generate")

        chain = [(request_tx["contexts"]["trace"], PARENT_SPAN),                                # browser -> controller
                 (comparison_tx["contexts"]["trace"], request_tx["contexts"]["trace"]["span_id"]),  # request -> thread
                 (qwen_agent, comparison_tx["contexts"]["trace"]["span_id"]),                   # thread -> agent
                 (qwen_call, qwen_agent["span_id"]),                                               # agent -> model call
                 (worker_tx["contexts"]["trace"], qwen_call["span_id"])]                         # model call -> worker
        for context, parent in chain:
            self.assertEqual(context["trace_id"], TRACE_ID)
            self.assertEqual(context["parent_span_id"], parent)
        self.assertNotIn("request", comparison_tx, "background work does not inherit the HTTP request's metadata")
        self.assertEqual(comparison_tx["tags"]["dapta.service"], "controller")
        self.assertEqual(worker_tx["tags"]["dapta.service"], "qwen-worker")
        self.assertNotIn(SECRET, transport.raw() + worker_transport.raw())

    def test_worker_continues_the_controller_trace(self):
        from fastapi.testclient import TestClient

        transport = start_sentry(service="qwen-worker")
        with patch.dict(os.environ, {"DAPTA_QWEN_TOKEN": "worker-token"}, clear=False):
            from dapta import model_server

            reply = {"output": {"ok": True}, "input_tokens": 3, "output_tokens": 2, "latency_ms": 1, "ttft_ms": None}
            with patch.object(model_server.runtime, "generate", return_value=reply):
                client = TestClient(model_server.app)  # no lifespan: the model is never loaded
                response = client.post("/generate", headers={**INCOMING, "Authorization": "Bearer worker-token"},
                                       json={"run_id": "run-1", "workflow": "Summarise", "document_text": SECRET,
                                             "adapter_id": "nda-intake/qwen3-vl-8b-v1"})
                rejected = client.post("/generate", json={})
        sentry_sdk.flush()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(rejected.status_code, 401, "worker auth unchanged")
        [tx] = [t for t in transport.items("transaction") if t["transaction"] == "/generate"
                and t["contexts"]["trace"].get("parent_span_id") == PARENT_SPAN]
        self.assertEqual(tx["contexts"]["trace"]["trace_id"], TRACE_ID)
        self.assertEqual(tx["tags"]["dapta.service"], "qwen-worker")
        self.assertNotIn(SECRET, transport.raw())

    def test_errors_carry_no_document_text_in_locals_or_messages(self):
        transport = start_sentry()

        def failing(text):
            raise RuntimeError("generation failed")

        try:
            failing(SECRET)
        except RuntimeError as error:
            observability.capture_exception(error, stage="compare", document_text=SECRET)
        sentry_sdk.flush()
        [event] = transport.items("event")
        self.assertEqual(event["tags"]["dapta.stage"], "compare")
        self.assertNotIn(SECRET, transport.raw())

    def test_provider_error_details_are_replaced_before_capture(self):
        from dapta.openrouter import OpenRouterError

        transport = start_sentry()
        private = "provider echoed private prompt 91f8"
        observability.capture_exception(OpenRouterError(private, status=403), stage="compare")
        sentry_sdk.flush()
        [event] = transport.items("event")
        self.assertEqual(event["tags"]["dapta.error_category"], "content_policy")
        self.assertNotIn(private, transport.raw())


if __name__ == "__main__":
    unittest.main()
