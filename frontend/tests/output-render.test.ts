import "./dom.ts";
import assert from "node:assert/strict";
import { test } from "node:test";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { ModelOutput, OutputPreview } from "../src/ModelOutput.tsx";
import { ModelGrid } from "../src/App.tsx";
import { chartCandidate, csvExport, parseCsv, parseOutput, prepareChartSpec, recordsTable } from "../src/outputData.ts";
import { container } from "./dom.ts";
import type { Run } from "../src/types.ts";

const chart = { mark: "bar", encoding: { x: { field: "region", type: "nominal" }, y: { field: "sales", type: "quantitative", aggregate: "sum" } } };

async function mount(output: unknown) {
  const host = container(); const root = createRoot(host);
  await act(async () => root.render(createElement(ModelOutput, { output, modelLabel: "Test model" })));
  return { host, root, async close() { await act(async () => root.unmount()); host.remove(); } };
}

test("JSON strings and fences can be previewed without changing the original answer", () => {
  assert.deepEqual(parseOutput('```json\n{"queue":"billing"}\n```'), { queue: "billing" });
  assert.equal(parseOutput("<script>alert(1)</script>"), "<script>alert(1)</script>");
  assert.equal(parseOutput("{truncated"), "{truncated");
});

test("CSV parser preserves multiline quoted fields, duplicates, blanks and numeric strings", () => {
  assert.deepEqual(parseCsv('\uFEFFname,id,note\r\n"A, B",001,"two\nlines"\r\n"A ""B""",,\r\n'), [
    ["name", "id", "note"], ["A, B", "001", "two\nlines"], ['A "B"', "", ""],
  ]);
  assert.throws(() => parseCsv('name\n"unclosed'), /unclosed/);
  assert.match(csvExport(["name", "number"], [["=HYPERLINK(1)", -5], ["  +cmd", "001"]]), /"'=HYPERLINK\(1\)","-5"/);
  assert.match(csvExport(["value"], [['x"y']]), /"x""y"/);
});

test("table columns use the union of row keys without inventing missing fields", () => {
  assert.deepEqual(recordsTable([{ a: 1 }, { b: null }]), { columns: ["a", "b"], cells: [[1, undefined], [undefined, null]] });
  assert.equal(recordsTable(["text", {}]), undefined);
});

test("chart preview uses returned rows only when no explicit data exists and does not mutate answers", () => {
  const rows = [{ region: "West", sales: 40, url: "ordinary data, not a resource" }];
  const candidate = { ...chart, $schema: "https://vega.github.io/schema/vega-lite/v5.json" };
  const prepared = prepareChartSpec(candidate, rows);
  assert.deepEqual(prepared.data, { values: rows });
  assert.ok(!("data" in candidate));
  const explicit = { ...chart, data: { values: [{ region: "East", sales: 2 }] } };
  assert.deepEqual(prepareChartSpec(explicit, rows).data, explicit.data);
  const layered = { layer: [explicit] };
  assert.ok(!("data" in prepareChartSpec(layered, rows)));
  assert.deepEqual(chartCandidate({ dashboard_spec: JSON.stringify(candidate) }), candidate);
});

test("unsafe, oversized and non-Vega-Lite chart specs are rejected at every nesting level", () => {
  for (const unsafe of [
    { ...chart, data: { url: "https://private.example/data.csv" } },
    { layer: [{ ...chart, encoding: { href: { value: "javascript:alert(1)" } } }] },
    { ...chart, usermeta: { embedOptions: { loader: "unsafe" } } },
    { ...chart, params: [{ name: "x", bind: { input: "text" } }] },
    { ...chart, data: { sequence: { start: 0, stop: 1e12 } } },
    { ...chart, mark: { type: "image" } },
    { ...chart, $schema: "https://vega.github.io/schema/vega/v6.json" },
    { ...chart, data: { values: Array.from({ length: 5001 }, () => ({ a: 1 })) } },
    { ...chart, description: "a".repeat(1_000_001) },
    { dashboard_code: "print('not a spec')" }, null,
  ]) assert.throws(() => prepareChartSpec(unsafe));
});

test("extracted fields and support routing badges are readable, with original JSON one click away", async () => {
  const output = { queue: "billing", priority: "urgent", response_code: "BILL-02", other: null };
  const view = await mount(output);
  try {
    assert.equal(view.host.querySelector(".output-badge.urgent")?.textContent, "urgent");
    assert.match(view.host.textContent!, /Not provided/);
    assert.ok(!view.host.querySelector("pre"));
    await act(async () => ([...view.host.querySelectorAll("button")].find((button) => button.textContent === "JSON")!).click());
    assert.equal(view.host.querySelector("pre")?.textContent, JSON.stringify(output, null, 2));
  } finally { await view.close(); }
});

test("receipt reports preserve mixed currencies, unknown amounts, zero totals and line items", async () => {
  const view = await mount({ receipts: [
    { merchant: "Cafe", currency: "SGD", total: 12.5, tax: null, line_items: [{ description: "Tea", quantity: 1, amount: 12.5 }] },
    { merchant: "Store", currency: "EUR", total: 0, transaction_date: "2026-10-18", line_items: [] },
    { merchant: "Unknown", currency: null, total: null },
  ] });
  try {
    assert.equal(view.host.querySelectorAll(".output-receipt").length, 3);
    assert.match(view.host.textContent!, /12\.50SGD/);
    assert.match(view.host.textContent!, /0\.00EUR/);
    assert.match(view.host.textContent!, /Currency not provided/);
    assert.match(view.host.textContent!, /Not provided/);
    assert.ok(!view.host.textContent!.includes("$"));
    assert.equal(view.host.querySelectorAll(".output-table tbody tr").length, 1);
  } finally { await view.close(); }
});

test("cleaned data tables paginate without hiding rows or inheriting global comparison table columns", async () => {
  const view = await mount({ cleaned_rows: Array.from({ length: 21 }, (_, index) => ({ id: index, value: "Row " + index })) });
  try {
    assert.equal(view.host.querySelectorAll("tbody tr").length, 10);
    assert.match(view.host.textContent!, /21 rows · 2 columns/);
    const next = [...view.host.querySelectorAll("button")].find((button) => button.textContent === "Next")!;
    await act(async () => next.click());
    await act(async () => next.click());
    assert.equal(view.host.querySelectorAll("tbody tr").length, 1);
    assert.match(view.host.textContent!, /Row 20/);
    assert.equal(next.disabled, true);
  } finally { await view.close(); }
});

test("legacy code artifacts and model HTML are shown only as text", async () => {
  const source = '<script>globalThis.modelCodeExecuted = true</script>';
  const view = await mount({ cleaned_csv: 'id,name\n001,"<img src=x onerror=alert(1)>"', dashboard_code: source });
  try {
    assert.match(view.host.textContent!, /001/);
    assert.ok(!view.host.querySelector("script, img"));
    assert.match(view.host.textContent!, /not a Vega-Lite chart specification/);
    assert.equal(view.host.querySelector(".output-code pre")?.textContent, source);
    assert.equal((globalThis as Record<string, unknown>).modelCodeExecuted, undefined);
  } finally { await view.close(); }
});

test("bad chart specs remain local and leave their cleaned data and other fields visible", async () => {
  const view = await mount({ dashboard_spec: { ...chart, data: { url: "https://example.com" } }, cleaned_rows: [{ region: "West" }], validation_summary: { warnings: ["Missing date"] } });
  try {
    assert.match(view.host.textContent!, /inline data/);
    assert.match(view.host.textContent!, /West/);
    assert.match(view.host.textContent!, /Missing date/);
  } finally { await view.close(); }
});

test("expanded output is a labelled focus-trapped dialog and Escape restores focus", async () => {
  const view = await mount({ parties: ["Acme", "Beta"], effective_date: "2026-10-18" });
  try {
    const expand = view.host.querySelector<HTMLButtonElement>('[aria-label="Expand Test model output"]')!;
    expand.focus();
    await act(async () => expand.click());
    const dialog = document.querySelector('[role="dialog"]')!;
    assert.ok(dialog.textContent!.includes("Test model"));
    assert.equal(document.body.style.overflow, "hidden");
    assert.equal(document.activeElement?.getAttribute("aria-label"), "Close expanded output");
    await act(async () => document.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, cancelable: true })));
    assert.equal(document.activeElement?.getAttribute("aria-label"), "Download original JSON");
    await act(async () => document.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
    assert.ok(!document.querySelector('[role="dialog"]'));
    assert.equal(document.body.style.overflow, "");
    assert.equal(document.activeElement?.getAttribute("aria-label"), "Expand Test model output");
  } finally { await view.close(); }
});

test("worker charts render as isolated SVG images and dispose worker/blob resources", async () => {
  const original = globalThis.Worker; const revoke = URL.revokeObjectURL;
  let terminated = 0; let revoked = 0;
  class FakeWorker {
    onmessage?: (event: { data: { svg: string } }) => void;
    postMessage(message: { spec: unknown }) {
      assert.deepEqual((message.spec as Record<string, unknown>).data, { values: [{ region: "West", sales: 2 }] });
      queueMicrotask(() => this.onmessage?.({ data: { svg: '<svg xmlns="http://www.w3.org/2000/svg"><text>West</text></svg>' } }));
    }
    terminate() { terminated++; }
  }
  globalThis.Worker = FakeWorker as unknown as typeof Worker;
  URL.revokeObjectURL = (url) => { revoked++; revoke(url); };
  const view = await mount({ dashboard_spec: chart, cleaned_rows: [{ region: "West", sales: 2 }] });
  try {
    assert.match(view.host.querySelector("img")?.getAttribute("src") || "", /^blob:/);
    assert.ok(!view.host.querySelector(".output-chart-scroll svg"));
    assert.ok(terminated > 0);
  } finally { await view.close(); globalThis.Worker = original; URL.revokeObjectURL = revoke; }
  assert.equal(revoked, 1);
});

test("all three compared models use the same visual output renderer without altering metrics", async () => {
  const host = container(); const root = createRoot(host);
  const result = { status: "completed", model: "test", output: { queue: "identity", priority: "normal", response_code: "ID-01" }, metrics: { latency_ms: 1230, input_tokens: 12, output_tokens: 34, cost_usd: "0.001" } };
  const run = { run_id: "visual", created_at: new Date().toISOString(), status: "completed", results: { qwen_lora: result, frontier_a: result, frontier_b: result } } as Run;
  try {
    await act(async () => root.render(createElement(ModelGrid, { run, running: false })));
    assert.equal(host.querySelectorAll(".model-output").length, 3);
    assert.equal(host.querySelectorAll(".output-badge").length, 9);
    assert.equal(host.querySelectorAll(".comparison-matrix").length, 1);
    assert.equal(host.textContent!.split("1.23").length - 1, 3);
  } finally { await act(async () => root.unmount()); host.remove(); }
});
