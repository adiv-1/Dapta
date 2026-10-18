// Loaded only in the chart worker (and in its integration tests).
import { compile, type TopLevelSpec } from "vega-lite";
import { View, parse, logger, type Loader } from "vega";
import { expressionInterpreter } from "vega-interpreter";
import { prepareChartSpec } from "./outputData";

const blocked = async (): Promise<never> => { throw new Error("External chart resources are disabled."); };
const inlineOnlyLoader: Loader = { load: blocked, sanitize: blocked, http: blocked, file: blocked };

export async function renderChartSvg(candidate: unknown, width: number): Promise<string> {
  const spec = prepareChartSpec(candidate);
  const composed = ["layer", "vconcat", "hconcat", "concat", "facet", "repeat"].some((key) => key in spec);
  const chartWidth = Math.max(220, Math.min(1000, width));
  // Preserve the model's data, encodings, titles and transformations. Only bound
  // display dimensions; compound charts can scroll rather than becoming illegible.
  const bounded = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(bounded);
    if (value !== null && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key,
        (key === "width" || key === "height") && typeof item === "number"
          ? Math.max(40, Math.min(1000, item))
          : key === "data" || key === "datasets" ? item : bounded(item)]));
    }
    return value;
  };
  const presentation = { ...(bounded(spec) as object),
    ...(!composed ? { width: chartWidth, height: 230 } : {}),
    config: { font: "Inter, sans-serif", background: "#ffffff", view: { stroke: null },
      axis: { labelColor: "#61725b", titleColor: "#384d35", gridColor: "#edf1e8" },
      range: { category: ["#487e3b", "#9cb886", "#d4b871", "#8473ad", "#6594b8", "#b67e76"] },
      ...(typeof spec.config === "object" && spec.config !== null ? spec.config : {}) },
  };
  const quietLogger = logger(0);
  const compiled = compile(presentation as TopLevelSpec, { logger: quietLogger }).spec;
  const view = new View(parse(compiled, undefined, { ast: true }), {
    renderer: "none", loader: inlineOnlyLoader, expr: expressionInterpreter,
    logger: quietLogger,
  });
  try { await view.runAsync(); return await view.toSVG(); }
  finally { view.finalize(); }
}
