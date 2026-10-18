// Local-only visual regression fixtures; not an entry point in production builds.
import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ModelGrid } from "../../src/App";
import type { Run } from "../../src/types";
import "../../src/styles.css";
import "../../src/conversation.css";
import "../../src/output.css";

const rows = [{ region: "West", revenue: 425 }, { region: "East", revenue: 310 }, { region: "North", revenue: 190 }, { region: "South", revenue: 260 }];
const dashboard = { cleaned_rows: rows, dashboard_spec: { $schema: "https://vega.github.io/schema/vega-lite/v6.json", title: "Sales by region", mark: { type: "bar", cornerRadiusTopLeft: 3, cornerRadiusTopRight: 3 }, encoding: { x: { field: "region", type: "nominal", title: null }, y: { field: "revenue", type: "quantitative", title: "Revenue" }, color: { field: "region", type: "nominal", legend: null } } }, validation_summary: { duplicates_removed: 2, warnings: ["One invalid date was excluded."] }, cleaning_code: "# Model-generated source is never executed by this preview.\ndf = df.drop_duplicates()" };
const fixtures: Record<string, unknown> = {
  Dashboard: dashboard,
  Receipts: { receipts: [{ merchant: "Northstar Cafe", transaction_date: "2026-10-18", currency: "USD", subtotal: 24, tax: 1.92, total: 25.92, line_items: [{ description: "Client lunch", quantity: 2, amount: 24 }] }] },
  "Support routing": { queue: "billing", priority: "urgent", response_code: "BILL-02" },
  "Extracted fields": { effective_date: "2026-10-18", jurisdiction: "Delaware", parties: ["Acme Corporation", "Beta Analytics LLC"], term: "3_years" },
  "Invalid chart": { ...dashboard, dashboard_spec: { ...dashboard.dashboard_spec, data: { url: "https://example.invalid/data.csv" } } },
};

function Preview() {
  const [fixture, setFixture] = useState("Dashboard");
  const result = (label: string, latency: number, tokens: number, cost: string) => ({ status: "completed", label, model: "fixture-only", output: fixtures[fixture], metrics: { latency_ms: latency, input_tokens: tokens, output_tokens: 72, cost_usd: cost, accuracy: 100 } });
  const run = { run_id: fixture, status: "completed", created_at: "2026-10-18T12:00:00Z", results: { qwen_lora: result("Qwen LoRA", 1250, 195, ".00047"), frontier_a: result("Claude", 6190, 2234, ".03696"), frontier_b: result("GPT", 7470, 1988, ".00751") } } as unknown as Run;
  return <main style={{ maxWidth: 1440, margin: "30px auto", padding: "0 20px" }}><p style={{ color: "#65745d", fontSize: 12 }}>Synthetic UI fixtures · No model calls · Not benchmark evidence</p><div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>{Object.keys(fixtures).map((name) => <button key={name} onClick={() => setFixture(name)} aria-pressed={name === fixture}>{name}</button>)}</div><ModelGrid run={run} running={false} /></main>;
}

createRoot(document.getElementById("root")!).render(<Preview />);
