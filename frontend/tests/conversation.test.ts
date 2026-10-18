import "./dom.ts";
import assert from "node:assert/strict";
import { test } from "node:test";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { Conversation } from "../src/Conversation.tsx";
import { ModelGrid } from "../src/App.tsx";
import { api } from "../src/api.ts";
import type { Job, Run, RouteDecision } from "../src/types.ts";
import { container } from "./dom.ts";

const newRoute: RouteDecision = { decision: "create_adapter", adapter_id: null, confidence: .9, reason: "A new repeatable workflow.", source: "jev" };
const existingRoute: RouteDecision = { ...newRoute, decision: "existing_adapter", adapter_id: "test-adapter", reason: "A suitable specialist exists." };

async function mountConversation() {
  const host = container(); const root = createRoot(host);
  await act(async () => root.render(createElement(Conversation, {
    adapters: [], jobs: [], runs: [], onRun: () => undefined, onJob: () => undefined,
    renderRun: () => createElement("div", null, "Comparison started"), renderJob: () => null,
    openLibrary: () => undefined, clearAdapter: () => undefined, clearJob: () => undefined,
  })));
  async function send(value: string) {
    const textarea = host.querySelector("textarea")!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")!.set!.call(textarea, value);
      textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    await act(async () => (host.querySelector('[aria-label="Send message"]') as HTMLButtonElement).click());
  }
  return { host, send, async close() { await act(async () => root.unmount()); host.remove(); } };
}

test("new workflow routes each reply and waits for requirements and confirmation before paid work", async () => {
  const original = { route: api.route, createJob: api.createJob };
  const routed: string[] = []; const built: string[] = [];
  api.route = async (workflow) => { routed.push(workflow); return newRoute; };
  api.createJob = async (workflow) => { built.push(workflow); return { job_id: "new-job" } as Job; };
  const view = await mountConversation();
  try {
    await view.send("Create a support ticket specialist");
    assert.match(view.host.textContent!, /Before I build it/);
    assert.equal(built.length, 0);
    await view.send("Return queue and urgency as JSON, with explicit keyword rules.");
    assert.match(view.host.textContent!, /typical input/);
    await view.send("Generate short synthetic examples, including ambiguous messages.");
    assert.equal(routed.length, 3);
    assert.match(routed[2], /Return queue and urgency/);
    assert.match(routed[2], /Generate short synthetic examples/);
    assert.equal(built.length, 0);
    const build = [...view.host.querySelectorAll("button")].find((button) => button.textContent?.includes("Build specialist"))!;
    await act(async () => build.click());
    assert.equal(built.length, 1);
    assert.match(built[0], /explicit keyword rules/);
  } finally { await view.close(); Object.assign(api, original); }
});

test("matched specialist collects input conversationally and calls the real comparison endpoint", async () => {
  const original = { route: api.route, compare: api.compare, createJob: api.createJob };
  const calls: unknown[][] = []; let builds = 0; let routes = 0;
  api.route = async () => { routes++; return existingRoute; };
  api.compare = async (...args) => { calls.push(args); return { run_id: "run-test" } as Run; };
  api.createJob = async () => { builds++; return {} as Job; };
  const view = await mountConversation();
  try {
    await view.send("Extract an NDA");
    assert.equal(calls.length, 0);
    assert.match(view.host.textContent!, /Send the text/);
    await view.send("Mutual NDA between A and B, effective 2026-10-18.");
    assert.equal(routes, 2);
    assert.equal(calls.length, 1);
    assert.equal(calls[0][1], "Mutual NDA between A and B, effective 2026-10-18.");
    assert.equal(calls[0][2], "test-adapter");
    assert.equal(builds, 0);
  } finally { await view.close(); Object.assign(api, original); }
});

test("unsupported routes never create a job or comparison", async () => {
  const original = { route: api.route, compare: api.compare, createJob: api.createJob };
  api.route = async () => ({ ...newRoute, decision: "unsupported", reason: "This needs live information." });
  api.compare = async () => { throw new Error("must not compare"); };
  api.createJob = async () => { throw new Error("must not build"); };
  const view = await mountConversation();
  try { await view.send("What is the weather right now?"); assert.match(view.host.textContent!, /This needs live information/); }
  finally { await view.close(); Object.assign(api, original); }
});

test("fallback matches require confirmation and actual input before any comparison", async () => {
  const original = { route: api.route, compare: api.compare };
  let comparisons = 0;
  api.route = async () => ({ ...existingRoute, requires_confirmation: true, source: "registry_fallback" });
  api.compare = async () => { comparisons++; return { run_id: "r" } as Run; };
  const view = await mountConversation();
  try {
    await view.send("Extract an NDA");
    assert.match(view.host.textContent!, /Registry fallback/);
    const confirm = [...view.host.querySelectorAll("button")].find((button) => button.textContent?.includes("Confirm specialist"))!;
    await act(async () => confirm.click());
    assert.equal(comparisons, 0);
    assert.match(view.host.textContent!, /Send the text or attach/);
  } finally { await view.close(); Object.assign(api, original); }
});

test("comparison shows each measurement once and never awards missing-cost or partial-run winners", async () => {
  const host = container(); const root = createRoot(host);
  const condition = { status: "completed", label: "Specialist", model: "model", output: { result: "ok" }, metrics: { latency_ms: 1230, input_tokens: 123, output_tokens: 45, cost_usd: null } };
  const run = { run_id: "r", created_at: new Date().toISOString(), status: "running", results: { qwen_lora: condition } } as Run;
  try {
    await act(async () => root.render(createElement(ModelGrid, { run, running: true })));
    assert.equal(host.textContent!.split("1.23").length - 1, 1);
    assert.equal(host.textContent!.split("168").length - 1, 1);
    assert.ok(!host.textContent!.includes("$0.00000"));
    assert.ok(!host.textContent!.includes("Lowest"));
    assert.match(host.textContent!, /Working/);
  } finally { await act(async () => root.unmount()); host.remove(); }
});
