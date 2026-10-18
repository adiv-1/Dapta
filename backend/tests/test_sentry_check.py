"""sentry_check reports queued, accepted-by-ingest and stored-in-Sentry as different outcomes.

A local fake Sentry ingest endpoint and a fake Sentry API stand in for sentry.io; the real SDK HTTP
transport sends to them.
"""

import contextlib
import io
import os
import re
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import sentry_sdk

from dapta import sentry_check


class FakeSentry:
    """Ingest (POST /api/1/envelope/) answering `ingest_status`, and the events API (GET) answering
    404 for the first `not_found_reads` reads of each event and then `api_status`."""

    def __init__(self, ingest_status=200, api_status=200, not_found_reads=0):
        self.ingest_status, self.api_status, self.not_found_reads = ingest_status, api_status, not_found_reads
        self.envelopes = []
        self.reads = {}
        self.auth = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                fake.envelopes.append(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                self._reply(fake.ingest_status, b'{"id":"x"}')

            def do_GET(self):
                fake.auth.append(self.headers.get("Authorization"))
                match = re.fullmatch(r"/api/0/projects/dapta-org/(?:dapta-backend|1)/events/([0-9a-f]{32})/", self.path)
                if not match:
                    return self._reply(404, b"{}")
                count = fake.reads[match[1]] = fake.reads.get(match[1], 0) + 1
                self._reply(404 if count <= fake.not_found_reads else fake.api_status, b'{"id":"x"}')

            def _reply(self, status, body):
                self.send_response(status)
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


def unused_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class SentryCheckTests(unittest.TestCase):
    def setUp(self):
        self.fake = None

    def tearDown(self):
        sentry_sdk.get_global_scope().set_client(None)
        if self.fake:
            self.fake.close()

    def run_check(self, *argv, dsn=None, confirm_env=True, env=None):
        environment = {"SENTRY_DSN": dsn or f"http://public@127.0.0.1:{self.fake.server.server_port}/1",
                       "SENTRY_ENVIRONMENT": "test", "SENTRY_RELEASE": "check-release",
                       "SENTRY_TRACES_SAMPLE_RATE": "1.0", "SENTRY_PROFILES_SAMPLE_RATE": "0",
                       "SENTRY_AUTH_TOKEN": "", "SENTRY_ORG": "", "SENTRY_PROJECT": "", "SENTRY_API_URL": ""}
        if confirm_env and self.fake:
            environment.update({"SENTRY_AUTH_TOKEN": "sntryu_test", "SENTRY_ORG": "dapta-org",
                                "SENTRY_PROJECT": "dapta-backend", "SENTRY_API_URL": self.fake.url})
        environment.update(env or {})
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, environment), patch.object(sentry_check, "POLL_INTERVAL", 0.05), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = sentry_check.main(list(argv))
        return code, out.getvalue() + err.getvalue()

    def test_accepted_by_ingest_is_not_reported_as_stored(self):
        self.fake = FakeSentry(ingest_status=200)
        code, output = self.run_check("--service", "controller")
        self.assertEqual(code, 0)
        self.assertEqual(output.count("accepted by Sentry ingest (HTTP 200)"), 3, output)
        self.assertNotIn("stored:", output)
        self.assertIn("rerun with --confirm", output)
        self.assertEqual(len(self.fake.envelopes), 3, "one message and two transactions were really sent")

    def test_confirm_waits_until_each_event_can_be_read_back(self):
        self.fake = FakeSentry(ingest_status=200, not_found_reads=2)
        code, output = self.run_check("--confirm", "--timeout", "10")
        self.assertEqual(code, 0, output)
        self.assertEqual(output.count("stored: read back through the Sentry API"), 3, output)
        self.assertEqual(sorted(self.fake.reads.values()), [3, 3, 3], "polled through the 404s")
        self.assertEqual(set(self.fake.auth), {"Bearer sntryu_test"})
        self.assertNotIn("sntryu_test", output, "the auth token is never printed")

    def test_accepted_but_never_stored_fails_the_check(self):
        self.fake = FakeSentry(ingest_status=200, not_found_reads=10_000)
        code, output = self.run_check("--confirm", "--timeout", "0.3")
        self.assertEqual(code, 1)
        self.assertEqual(output.count("unconfirmed: not found after"), 3, output)
        self.assertIn("NOT CONFIRMED", output)

    def test_token_without_access_is_reported_not_retried(self):
        self.fake = FakeSentry(ingest_status=200, api_status=403)
        code, output = self.run_check("--confirm", "--timeout", "10")
        self.assertEqual(code, 1)
        self.assertIn("refused the token (HTTP 403)", output)

    def test_ingest_rejection_is_reported_as_rejected(self):
        self.fake = FakeSentry(ingest_status=403)
        code, output = self.run_check("--confirm")
        self.assertEqual(code, 1)
        self.assertEqual(output.count("rejected by Sentry ingest (HTTP 403)"), 3, output)
        self.assertEqual(self.fake.reads, {}, "nothing rejected is looked up as stored")

    def test_unreachable_ingest_is_not_delivered(self):
        code, output = self.run_check(dsn=f"http://public@127.0.0.1:{unused_port()}/1", confirm_env=False)
        self.assertEqual(code, 1)
        self.assertNotIn("accepted", output.replace("Accepted by ingest", ""))
        self.assertIn("NOT CONFIRMED", output)

    def test_confirm_requires_api_credentials(self):
        self.fake = FakeSentry()
        code, output = self.run_check("--confirm", confirm_env=False)
        self.assertEqual(code, 2)
        self.assertIn("SENTRY_AUTH_TOKEN, SENTRY_ORG", output)
        self.assertEqual(self.fake.envelopes, [], "nothing is sent when the check cannot complete")

    def test_confirm_takes_the_project_and_api_from_the_dsn(self):
        self.fake = FakeSentry()
        code, output = self.run_check("--confirm", "--timeout", "10", confirm_env=False,
                                      env={"SENTRY_AUTH_TOKEN": "sntryu_test", "SENTRY_ORG": "dapta-org"})
        self.assertEqual(code, 0, output)
        self.assertEqual(output.count("stored: read back through the Sentry API"), 3, output)

    def test_a_project_that_does_not_own_the_dsn_is_refused(self):
        self.fake = FakeSentry()
        code, output = self.run_check("--confirm", confirm_env=False,
                                      env={"SENTRY_AUTH_TOKEN": "sntryu_test", "SENTRY_ORG": "dapta-org",
                                           "SENTRY_PROJECT": "4512275570819072"})
        self.assertEqual(code, 2)
        self.assertIn("not the project that owns SENTRY_DSN (1)", output)
        self.assertEqual(self.fake.envelopes, [])

    def test_dsn_regions_map_to_their_api(self):
        defaults = sentry_check._dsn_api_defaults
        self.assertEqual(defaults("https://key@o450.ingest.us.sentry.io/4512275570819072"),
                         ("4512275570819072", "https://sentry.io"))
        self.assertEqual(defaults("https://key@o450.ingest.de.sentry.io/77"), ("77", "https://de.sentry.io"))
        self.assertEqual(defaults("https://key@sentry.example.com/sentry/5"), ("5", "https://sentry.example.com"))

    def test_without_dsn_nothing_is_sent(self):
        with patch.dict(os.environ, {"SENTRY_DSN": ""}), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(sentry_check.main([]), 2)
        self.assertIn("Sentry is not active", err.getvalue())


if __name__ == "__main__":
    unittest.main()
