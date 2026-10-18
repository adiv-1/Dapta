import assert from "node:assert/strict";
import { test } from "node:test";

import * as Sentry from "@sentry/browser";

import { toProxyTrace } from "../src/tracing.ts";

// The installed SDK, not a mock: what api.ts sends in the dapta-proxy payload must be the
// active client span's own trace context.
const sent: unknown[] = [];
Sentry.init({
  dsn: "https://public@o0.ingest.sentry.io/1",
  defaultIntegrations: false,
  tracesSampleRate: 1,
  transport: () => ({ send: async (envelope: unknown) => { sent.push(envelope); return {}; }, flush: async () => true }),
});

test("the payload trace is the active api span, sampled, with matching baggage", async () => {
  let span: { traceId: string; spanId: string } | undefined;
  let payload: ReturnType<typeof toProxyTrace>;
  await Sentry.startSpan({ name: "POST /v1/comparisons", op: "dapta.api" }, async (active) => {
    span = active.spanContext();
    payload = toProxyTrace(Sentry.getTraceData());
  });
  assert.ok(span && payload);
  assert.equal(payload.sentry_trace, `${span.traceId}-${span.spanId}-1`);
  assert.match(payload.baggage ?? "", new RegExp(`sentry-trace_id=${span.traceId}`));
  assert.match(payload.baggage ?? "", /sentry-sampled=true/);
});

test("outside any span the trace context is still valid, so the backend joins one trace", () => {
  const payload = toProxyTrace(Sentry.getTraceData());
  assert.ok(payload);
  assert.match(payload.sentry_trace, /^[0-9a-f]{32}-[0-9a-f]{16}/);
});
