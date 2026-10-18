import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";

// Runs the real Base44 dapta-proxy handler with stubbed SDK/secrets and a recording fetch.
const TRACE = "0123456789abcdef0123456789abcdef-89abcdef01234567-1";
type Call = { url: string; method?: string; headers: Headers };
const calls: Call[] = [];

globalThis.fetch = (async (url: string, init: RequestInit = {}) => {
  calls.push({ url: String(url), method: init.method, headers: new Headers(init.headers) });
  return new Response(JSON.stringify([]), { status: 200, headers: { "Content-Type": "application/json" } });
}) as typeof fetch;

const { default: handler } = await import("../base44/functions/dapta-proxy/entry.ts");

function invoke(payload: unknown, headers: Record<string, string> = {}) {
  return handler(new Request("https://app.base44.test/functions/dapta-proxy", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(payload),
  }));
}

beforeEach(() => {
  calls.length = 0;
  (globalThis as any).__base44User = { id: "user-1" };
  (globalThis as any).__base44Secrets = {
    DAPTA_CONTROLLER_URL: "https://controller.test/",
    DAPTA_CONTROL_TOKEN: "control-token",
  };
});

test("forwards the browser trace to the controller alongside the control token", async () => {
  const response = await invoke({
    path: "/v1/runs",
    trace: { sentry_trace: TRACE, baggage: "sentry-trace_id=0123,third-party=leak,sentry-sampled=true" },
  });
  assert.equal(response.status, 200);
  const [call] = calls;
  assert.equal(call.url, "https://controller.test/v1/runs");
  assert.equal(call.headers.get("sentry-trace"), TRACE);
  assert.equal(call.headers.get("baggage"), "sentry-trace_id=0123,sentry-sampled=true");
  assert.equal(call.headers.get("x-dapta-token"), "control-token");
});

test("falls back to trace headers on the function request itself", async () => {
  await invoke({ path: "/v1/adapters" }, { "sentry-trace": TRACE, baggage: "sentry-sampled=true" });
  assert.equal(calls[0].headers.get("sentry-trace"), TRACE);
  assert.equal(calls[0].headers.get("baggage"), "sentry-sampled=true");
});

test("drops malformed trace context without blocking the request", async () => {
  const response = await invoke({ path: "/v1/jobs", trace: { sentry_trace: "abc\r\nX-Evil: 1", baggage: "sentry-a=1" } });
  assert.equal(response.status, 200);
  assert.equal(calls[0].headers.get("sentry-trace"), null);
  assert.equal(calls[0].headers.get("baggage"), null);
});

test("keeps POST bodies, idempotency keys and authentication unchanged", async () => {
  await invoke({ path: "/v1/comparisons", method: "POST", idempotencyKey: "key-12345678",
                 body: { workflow: "w" }, trace: { sentry_trace: TRACE } });
  assert.equal(calls[0].method, "POST");
  assert.equal(calls[0].headers.get("idempotency-key"), "key-12345678");
  assert.equal(calls[0].headers.get("sentry-trace"), TRACE);

  (globalThis as any).__base44User = null;
  const unauthenticated = await invoke({ path: "/v1/runs", trace: { sentry_trace: TRACE } });
  assert.equal(unauthenticated.status, 401);
  assert.equal(calls.length, 1, "no controller call without a Base44 user");
});
