import { renderChartSvg } from "./chartRuntime";

// Model-controlled compilation/dataflow never runs on the UI thread.
self.onmessage = async (event: MessageEvent<{ spec: unknown; width: number }>) => {
  try {
    const svg = await renderChartSvg(event.data.spec, event.data.width);
    self.postMessage({ svg });
  } catch {
    // Do not leak model/customer content through exception messages or telemetry.
    self.postMessage({ error: "This chart could not be rendered. Inspect the model’s JSON for the specification." });
  }
};
