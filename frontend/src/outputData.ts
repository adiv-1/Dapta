// Presentation only: never repair a model's answer or manufacture missing values.
export type OutputRecord = Record<string, unknown>;
export const isRecord = (value: unknown): value is OutputRecord => value !== null && typeof value === "object" && !Array.isArray(value);
export const fieldLabel = (key: string) => key.replaceAll("_", " ").replace(/([a-z])([A-Z])/g, "$1 $2");

export function parseOutput(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const text = value.trim().replace(/^```(?:json)?\s*\n?([\s\S]*?)\n?```$/, "$1");
  try { return JSON.parse(text); } catch { return value; }
}

export function chartCandidate(output: unknown): unknown {
  if (!isRecord(output)) return undefined;
  for (const key of ["dashboard_spec", "vega_lite_spec", "chart_spec"]) {
    if (output[key] != null) return parseOutput(output[key]);
  }
  if (output.mark || output.layer || output.vconcat || output.hconcat || output.concat || output.facet || output.repeat) return output;
  return undefined;
}

const MAX_CHART_BYTES = 1_000_000;
const MAX_ROWS = 5_000;

export function hasChartData(value: unknown): boolean {
  return isRecord(value) && ("data" in value || ["layer", "vconcat", "hconcat", "concat", "spec"].some((key) =>
    Array.isArray(value[key]) ? (value[key] as unknown[]).some(hasChartData) : hasChartData(value[key])));
}

/** Accept declarative, self-contained Vega-Lite only. The worker additionally has
 * a deny-all loader, an expression interpreter (no eval), and a hard timeout. */
export function prepareChartSpec(candidate: unknown, fallbackRows?: unknown): OutputRecord {
  if (!isRecord(candidate)) throw new Error("The model did not return a valid Vega-Lite specification.");
  const json = JSON.stringify(candidate);
  if (new TextEncoder().encode(json).length > MAX_CHART_BYTES) throw new Error("This chart is too large to preview. Download the JSON to inspect it.");
  const spec: OutputRecord = JSON.parse(json);
  if (!(spec.mark || spec.layer || spec.vconcat || spec.hconcat || spec.concat || spec.facet || spec.repeat)) {
    throw new Error("The specification has no supported Vega-Lite chart view.");
  }
  if (spec.$schema && (typeof spec.$schema !== "string" || !/^https:\/\/vega\.github\.io\/schema\/vega-lite\/v[4-6](?:\.\d+)*\.json$/.test(spec.$schema))) {
    throw new Error("Only Vega-Lite chart specifications are supported.");
  }
  // Only supply cleaned_rows when the chart has no data anywhere. Never replace
  // explicit model data or guess which named dataset a model intended to use.
  if (!hasChartData(spec) && Array.isArray(fallbackRows)) spec.data = { values: fallbackRows };

  let nodes = 0; let rows = 0;
  function walk(value: unknown, depth: number) {
    if (++nodes > 30_000 || depth > 35) throw new Error("This chart is too complex to preview.");
    if (Array.isArray(value)) {
      if (value.length > MAX_ROWS) throw new Error("Chart previews support up to 5,000 data rows.");
      value.forEach((item) => walk(item, depth + 1));
    } else if (isRecord(value)) {
      for (const [key, item] of Object.entries(value)) {
        if (["url", "href", "usermeta", "bind", "sequence", "graticule", "__proto__", "constructor", "prototype"].includes(key)
          || key === "mark" && (item === "image" || isRecord(item) && item.type === "image")) {
          throw new Error("For safety, previews require inline data and cannot contain external resources, image marks, or browser controls.");
        }
        if (key === "values" && Array.isArray(item)) {
          rows += item.length; // Data fields named url / href are ordinary values, not configuration.
          if (rows > MAX_ROWS) throw new Error("Chart previews support up to 5,000 data rows.");
        } else if (key === "datasets" && isRecord(item)) {
          for (const dataset of Object.values(item)) {
            if (!Array.isArray(dataset)) throw new Error("Named chart datasets must contain inline rows.");
            rows += dataset.length;
          }
          if (rows > MAX_ROWS) throw new Error("Chart previews support up to 5,000 data rows.");
        } else walk(item, depth + 1);
      }
    }
  }
  // Include fallback data in the byte limit too.
  if (new TextEncoder().encode(JSON.stringify(spec)).length > MAX_CHART_BYTES) throw new Error("This chart is too large to preview.");
  walk(spec, 0);
  return spec;
}

// RFC 4180 parsing, including commas, escaped quotes and newlines inside cells.
// Keep values as strings: do not silently coerce account numbers/dates/currencies.
export function parseCsv(text: string): string[][] {
  if (!text) return [];
  const rows: string[][] = []; let row: string[] = []; let cell = ""; let quoted = false;
  const input = text.replace(/^\uFEFF/, "");
  for (let index = 0; index < input.length; index++) {
    const char = input[index];
    if (char === '"') {
      if (quoted && input[index + 1] === '"') { cell += '"'; index++; }
      else if (quoted || !cell) quoted = !quoted;
      else cell += char;
    } else if (!quoted && char === ",") { row.push(cell); cell = ""; }
    else if (!quoted && (char === "\n" || char === "\r")) {
      if (char === "\r" && input[index + 1] === "\n") index++;
      row.push(cell); rows.push(row); row = []; cell = "";
    } else cell += char;
  }
  if (quoted) throw new Error("The returned CSV has an unclosed quoted cell.");
  if (cell || row.length || !/[\r\n]$/.test(input)) { row.push(cell); rows.push(row); }
  return rows;
}

export function recordsTable(rows: unknown[]): { columns: string[]; cells: unknown[][] } | undefined {
  if (!rows.length || !rows.every(isRecord)) return undefined;
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  return { columns, cells: rows.map((row) => columns.map((key) => row[key])) };
}

export function displayValue(value: unknown): string {
  if (value === null) return "Not provided";
  if (value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

export function csvExport(columns: string[], cells: unknown[][]): string {
  const escape = (value: unknown) => {
    let text = value == null ? "" : displayValue(value);
    // Downloads are safe to open in spreadsheets; raw JSON retains exact values.
    if (/^\s*[=+@-]/.test(text) && !(typeof value === "number" && Number.isFinite(value))) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  return "\uFEFF" + [columns, ...cells].map((row) => row.map(escape).join(",")).join("\r\n");
}
