import assert from "node:assert/strict";
import { test } from "node:test";

import { stringMatchesSomePattern } from "@sentry/core";

import { propagationTargets, sentryBaggage, toProxyTrace } from "../src/tracing.ts";

const TRACE = "0123456789abcdef0123456789abcdef-89abcdef01234567-1";

test("carries a valid sentry-trace and only sentry- baggage members", () => {
  assert.deepEqual(
    toProxyTrace({ "sentry-trace": TRACE, baggage: "sentry-trace_id=0123,other=secret, sentry-sampled=true" }),
    { sentry_trace: TRACE, baggage: "sentry-trace_id=0123,sentry-sampled=true" },
  );
});

test("drops malformed or missing trace context", () => {
  assert.equal(toProxyTrace({}), undefined);
  assert.equal(toProxyTrace({ "sentry-trace": "not-a-trace" }), undefined);
  assert.equal(toProxyTrace({ "sentry-trace": `${TRACE}\nX-Injected: 1` }), undefined);
  assert.deepEqual(toProxyTrace({ "sentry-trace": TRACE }), { sentry_trace: TRACE });
});

test("rejects oversized baggage instead of truncating it", () => {
  assert.equal(sentryBaggage(`sentry-x=${"a".repeat(9000)}`), undefined);
  assert.equal(sentryBaggage("third-party=1"), undefined);
});

// The same matching @sentry/browser 10.x applies (shouldAttachHeaders in tracing/request.js).
function sdkWouldAttach(targetUrl: string, pageHref: string, targets: RegExp[]): boolean {
  const resolved = new URL(targetUrl, pageHref);
  const sameOrigin = resolved.origin === new URL(pageHref).origin;
  return stringMatchesSomePattern(resolved.toString(), targets)
    || (sameOrigin && stringMatchesSomePattern(resolved.pathname, targets));
}

test("local dev propagates only to the Dapta API proxy", () => {
  const targets = propagationTargets("/api");
  const page = "http://127.0.0.1:5173/";
  assert.equal(sdkWouldAttach("/api/v1/runs", page, targets), true);
  assert.equal(sdkWouldAttach("/api/v1/documents/extract", page, targets), true);
  assert.equal(sdkWouldAttach("/apiary", page, targets), false);
  assert.equal(sdkWouldAttach("https://openrouter.ai/api/v1/chat", page, targets), false);
});

test("an absolute dev API URL still matches only that API", () => {
  const targets = propagationTargets("http://localhost:8000/");
  assert.equal(sdkWouldAttach("http://localhost:8000/v1/runs", "http://localhost:5173/", targets), true);
  assert.equal(sdkWouldAttach("http://localhost:8001/v1/runs", "http://localhost:5173/", targets), false);
});

test("production propagates nothing automatically, wherever Base44 serves the app", () => {
  const targets = propagationTargets(undefined);
  assert.deepEqual(targets, []);
  for (const page of ["https://dapta.base44.app/", "https://base44.app/apps/APP"]) {
    assert.equal(sdkWouldAttach("https://base44.app/api/apps/APP/functions/dapta-proxy", page, targets), false);
  }
});
