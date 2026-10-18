import assert from "node:assert/strict";
import { test } from "node:test";
import { renderChartSvg } from "../src/chartRuntime.ts";

test("real Vega compiler/interpreter renders bar, line, pie, and compound dashboard SVGs", async () => {
  const data = { values: [{ region: "West", sales: 40 }, { region: "East", sales: 20 }] };
  const encoding = { x: { field: "region", type: "nominal" }, y: { field: "sales", type: "quantitative" } };
  for (const spec of [
    { mark: "bar", data, encoding },
    { mark: "line", data, encoding },
    { mark: "arc", data, encoding: { theta: { field: "sales", type: "quantitative" }, color: { field: "region", type: "nominal" } } },
    { data, vconcat: [{ mark: "bar", encoding }, { mark: "point", encoding }] },
    { mark: "bar", data, transform: [{ calculate: "datum.sales * 2", as: "doubled" }], encoding: { ...encoding, y: { field: "doubled", type: "quantitative", aggregate: "sum" } } },
  ]) {
    const svg = await renderChartSvg(spec, 320);
    assert.match(svg, /^<svg/);
    assert.match(svg, /West/);
    assert.match(svg, /<path/);
    assert.ok(!svg.includes("NaN"));
  }
});

test("headless chart runtime rejects external resources and escapes model text", async () => {
  await assert.rejects(renderChartSvg({ mark: "bar", data: { url: "https://example.com/secret" } }, 320), /inline data/);
  const svg = await renderChartSvg({ mark: "text", data: { values: [{ text: '<script>alert("x")</script>' }] }, encoding: { text: { field: "text" } } }, 320);
  assert.ok(!svg.includes("<script>"));
  assert.match(svg, /&lt;script&gt;/);
});
