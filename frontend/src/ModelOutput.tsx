import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { BarChart3, Check, Code2, Copy, Download, FileJson, Maximize2, Table2, X } from "lucide-react";
import { chartCandidate, csvExport, displayValue, fieldLabel, hasChartData, isRecord, parseCsv, parseOutput,
  prepareChartSpec, recordsTable, type OutputRecord } from "./outputData";

function download(text: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a"); link.href = url; link.download = filename; link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function ModelOutput({ output, modelLabel }: { output: unknown; modelLabel: string }) {
  const [mode, setMode] = useState<"preview" | "json">("preview");
  const [expanded, setExpanded] = useState(false);
  const [copyState, setCopyState] = useState("");
  const opener = useRef<HTMLElement | null>(null);
  const raw = JSON.stringify(output, null, 2) ?? "null";
  const parsed = useMemo(() => parseOutput(output), [raw]);
  const close = useCallback(() => setExpanded(false), []);
  useEffect(() => { if (copyState) { const timer = window.setTimeout(() => setCopyState(""), 2500); return () => window.clearTimeout(timer); } }, [copyState]);
  const controls = <div className="output-toolbar">
    <div className="output-tabs" role="group" aria-label="Output format">
      <button type="button" aria-pressed={mode === "preview"} onClick={() => setMode("preview")}>Preview</button>
      <button type="button" aria-pressed={mode === "json"} onClick={() => setMode("json")}><Code2 size={12} />JSON</button>
    </div>
    <div className="output-actions">
      <button type="button" title="Copy original JSON" aria-label="Copy original JSON" onClick={async () => {
        try { await navigator.clipboard.writeText(raw); setCopyState("Copied"); }
        catch { setCopyState("Copy unavailable. Use Download JSON."); }
      }}>{copyState === "Copied" ? <Check size={14} /> : <Copy size={14} />}</button>
      <button type="button" title="Download original JSON" aria-label="Download original JSON" onClick={() => download(raw, "model-output.json", "application/json")}><Download size={14} /></button>
      <button type="button" hidden={expanded} title="Expand output" aria-label={`Expand ${modelLabel} output`} onClick={(event) => { opener.current = event.currentTarget; setExpanded(true); }}><Maximize2 size={14} /></button>
    </div>
  </div>;
  const content = mode === "json"
    ? <pre className="output-json">{raw.length > 250_000 ? raw.slice(0, 250_000) + "\n… Preview shortened. Download JSON for the complete output." : raw}</pre>
    : <OutputPreview output={parsed} />;
  return <div className="model-output"><div hidden={expanded}>{controls}{!expanded && <>{copyState && <p className="output-note" role="status">{copyState}</p>}{content}</>}</div>
    {expanded && <><p className="output-note">Expanded view is open.</p><OutputDialog modelLabel={modelLabel} onClose={close} returnFocus={opener.current}>{controls}{copyState && <p className="output-note" role="status">{copyState}</p>}{content}</OutputDialog></>}
  </div>;
}

function OutputDialog({ modelLabel, onClose, returnFocus, children }: { modelLabel: string; onClose: () => void; returnFocus: HTMLElement | null; children: React.ReactNode }) {
  const ref = useRef<HTMLElement>(null); const heading = useId();
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow; document.body.style.overflow = "hidden";
    ref.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.stopPropagation(); onClose(); }
      if (event.key === "Tab") {
        const elements = Array.from(ref.current?.querySelectorAll<HTMLElement>('button:not([hidden]):not(:disabled), a[href], select, [tabindex="0"]') || []);
        const first = elements[0]; const last = elements[elements.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener("keydown", keydown);
    return () => { document.body.style.overflow = overflow; document.removeEventListener("keydown", keydown); (returnFocus || previous)?.focus(); };
  }, [onClose, returnFocus]);
  return createPortal(<div className="output-overlay" onClick={onClose}><section ref={ref} className="output-dialog" role="dialog" aria-modal="true" aria-labelledby={heading} onClick={(event) => event.stopPropagation()}>
    <header><div><span className="welcome-kicker">Model response</span><h2 id={heading}>{modelLabel}</h2></div><button type="button" onClick={onClose} aria-label="Close expanded output"><X size={20} /></button></header>
    <div className="model-output output-dialog-body">{children}</div>
  </section></div>, document.body);
}

export function OutputPreview({ output }: { output: unknown }) {
  if (!isRecord(output)) return <div className="output-preview"><StructuredValue value={output} /></div>;
  const candidate = chartCandidate(output);
  const directChart = candidate === output;
  const handled = new Set(["dashboard_spec", "vega_lite_spec", "chart_spec", "receipts", "expenses"]);
  const sourceKeys = new Set(["cleaning_code", "dashboard_code", "python_code", "html_code", "sql_code"]);
  if (Array.isArray(output.cleaned_rows)) handled.add("cleaned_rows");
  if (typeof output.cleaned_csv === "string") handled.add("cleaned_csv");
  if (Array.isArray(output.line_items)) handled.add("line_items");
  const receipt = !candidate && ("merchant" in output || "vendor" in output) && ("total" in output || "total_amount" in output);
  if (receipt) return <div className="output-preview"><ReceiptReport receipt={output} /></div>;
  return <div className="output-preview">
    {candidate !== undefined && <ChartPreview candidate={candidate} fallbackRows={output.cleaned_rows} />}
    {Array.isArray(output.cleaned_rows) && <OutputSection title="Cleaned data" icon="table"><ArrayValue value={output.cleaned_rows} /></OutputSection>}
    {typeof output.cleaned_csv === "string" && !Array.isArray(output.cleaned_rows) && <CsvPreview csv={output.cleaned_csv} />}
    {(["receipts", "expenses"] as const).map((key) => Array.isArray(output[key]) ? <OutputSection key={key} title={`${output[key].length} ${key}`}>
      {output[key].length === 0 ? <p className="output-note">No {key} returned.</p> : output[key].slice(0, 30).map((item, index) => isRecord(item)
        ? <ReceiptReport key={index} receipt={item} /> : <StructuredValue key={index} value={item} />)}
      {output[key].length > 30 && <p className="output-note">Showing the first 30. The complete list is in JSON.</p>}
    </OutputSection> : output[key] !== undefined ? <OutputSection key={key} title={fieldLabel(key)}><StructuredValue value={output[key]} /></OutputSection> : null)}
    {Array.isArray(output.line_items) && <OutputSection title="Line items" icon="table"><ArrayValue value={output.line_items} /></OutputSection>}
    {!directChart && <FieldList value={Object.fromEntries(Object.entries(output).filter(([key]) => !handled.has(key) && !sourceKeys.has(key)))} />}
    {Object.entries(output).filter(([key]) => sourceKeys.has(key)).map(([key, value]) => <details className="output-code" key={key}><summary><FileJson size={13} />{fieldLabel(key)}<span>Source only</span></summary><p className="output-note">Generated code is shown as text, never executed in your browser.</p><pre>{displayValue(value)}</pre></details>)}
    {typeof output.dashboard_code === "string" && candidate === undefined && <p className="output-notice">This response includes dashboard code, not a Vega-Lite chart specification. Code is not executed; ask the specialist to return dashboard_spec for a visual preview.</p>}
    {directChart && isRecord(output.data) && Array.isArray(output.data.values) && <OutputSection title="Chart data" icon="table"><ArrayValue value={output.data.values} /></OutputSection>}
    {Object.keys(output).length === 0 && <p className="output-note">The model returned an empty object.</p>}
  </div>;
}

function OutputSection({ title, icon, children }: { title: string; icon?: "table"; children: React.ReactNode }) {
  return <section className="output-section"><h4>{icon === "table" && <Table2 size={13} />}{title}</h4>{children}</section>;
}

function FieldList({ value, depth = 0 }: { value: OutputRecord; depth?: number }) {
  return <dl className="output-fields">{Object.entries(value).slice(0, 60).map(([key, item]) => <div key={key}><dt>{fieldLabel(key)}</dt><dd>
    {(key === "queue" || key === "priority" || key === "response_code") && typeof item === "string"
      ? <span className={`output-badge ${key === "priority" && item.toLowerCase() === "urgent" ? "urgent" : ""}`}>{item}</span>
      : <StructuredValue value={item} depth={depth + 1} />}
  </dd></div>)}{Object.keys(value).length > 60 && <div><dd className="output-note">More fields are available in JSON.</dd></div>}</dl>;
}

function StructuredValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (depth >= 4 && typeof value === "object" && value !== null) return <pre>{JSON.stringify(value, null, 2)}</pre>;
  if (Array.isArray(value)) return <ArrayValue value={value} depth={depth} />;
  if (isRecord(value)) return <FieldList value={value} depth={depth} />;
  return <span className={value == null ? "output-missing" : "output-value"}>{displayValue(value)}</span>;
}

function ArrayValue({ value, depth = 0 }: { value: unknown[]; depth?: number }) {
  const table = recordsTable(value);
  if (table) return <DataTable {...table} />;
  if (!value.length) return <span className="output-missing">Empty list</span>;
  return <ul className="output-list">{value.slice(0, 50).map((item, index) => <li key={index}><StructuredValue value={item} depth={depth + 1} /></li>)}{value.length > 50 && <li className="output-note">More items are available in JSON.</li>}</ul>;
}

function CsvPreview({ csv }: { csv: string }) {
  try {
    const [header, ...rows] = parseCsv(csv);
    const width = Math.max(header?.length || 0, ...rows.map((row) => row.length));
    const columns = Array.from({ length: width }, (_, index) => header?.[index] ?? `Column ${index + 1}`);
    return <OutputSection title="Cleaned data" icon="table"><DataTable columns={columns} cells={rows} /></OutputSection>;
  } catch { return <p className="output-notice">The returned CSV could not be parsed. Its original text is available in JSON.</p>; }
}

function DataTable({ columns, cells }: { columns: string[]; cells: unknown[][] }) {
  const [page, setPage] = useState(0); const size = 10;
  const last = Math.max(0, Math.ceil(cells.length / size) - 1); const current = Math.min(page, last);
  return <div className="output-data">
    <div className="output-table-meta"><span>{cells.length.toLocaleString()} rows · {columns.length} columns</span><button type="button" onClick={() => download(csvExport(columns, cells), "model-data.csv", "text/csv;charset=utf-8")}><Download size={12} />CSV</button></div>
    <div className="output-table-scroll" tabIndex={0} role="region" aria-label="Model data table"><table className="output-table"><thead><tr>{columns.slice(0, 40).map((key, index) => <th key={index} scope="col">{key}</th>)}</tr></thead><tbody>{cells.slice(current * size, (current + 1) * size).map((row, index) => <tr key={index}>{columns.slice(0, 40).map((_, col) => <td key={col}>{displayValue(row[col])}</td>)}</tr>)}</tbody></table></div>
    {columns.length > 40 && <p className="output-note">Showing 40 columns; download CSV for all columns.</p>}
    {last > 0 && <div className="output-pagination"><button type="button" disabled={current === 0} onClick={() => setPage(current - 1)}>Previous</button><span>{current * size + 1}–{Math.min((current + 1) * size, cells.length)} of {cells.length}</span><button type="button" disabled={current === last} onClick={() => setPage(current + 1)}>Next</button></div>}
    {!cells.length && <p className="output-note">No rows returned.</p>}
  </div>;
}

function ReceiptReport({ receipt }: { receipt: OutputRecord }) {
  const currency = typeof receipt.currency === "string" ? receipt.currency : null;
  const total = "total" in receipt ? receipt.total : receipt.total_amount;
  const amount = typeof total === "number" ? total.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 }) : displayValue(total);
  const rest = Object.fromEntries(Object.entries(receipt).filter(([key]) => !["merchant", "vendor", "total", "total_amount", "currency", "line_items"].includes(key)));
  return <article className="output-receipt"><header><span>Expense report</span><h4>{displayValue(receipt.merchant ?? receipt.vendor ?? "Receipt")}</h4></header>
    <div className="output-receipt-total"><span>Reported total</span><strong>{amount}{total != null && currency ? <small>{currency}</small> : null}</strong>{!currency && <small>Currency not provided</small>}</div>
    <FieldList value={rest} />
    {Array.isArray(receipt.line_items) ? <OutputSection title="Line items" icon="table"><ArrayValue value={receipt.line_items} /></OutputSection> : receipt.line_items !== undefined && <StructuredValue value={receipt.line_items} />}
  </article>;
}

function ChartPreview({ candidate, fallbackRows }: { candidate: unknown; fallbackRows?: unknown }) {
  const host = useRef<HTMLDivElement>(null);
  const [chart, setChart] = useState<{ url?: string; svg?: string; error?: string }>({});
  const signature = JSON.stringify([candidate, fallbackRows]);
  useEffect(() => {
    let spec: OutputRecord;
    setChart({});
    try { spec = prepareChartSpec(candidate, fallbackRows); }
    catch (error) { setChart({ error: error instanceof Error ? error.message : "Invalid chart specification." }); return; }
    if (typeof Worker === "undefined") { setChart({ error: "Chart previews require browser worker support. The full specification is available in JSON." }); return; }
    let disposed = false; let url: string | undefined;
    let worker: Worker;
    try { worker = new Worker(new URL("./chart.worker.ts", import.meta.url), { type: "module" }); }
    catch { setChart({ error: "Your browser blocked chart previews. The full specification is available in JSON." }); return; }
    const fail = (message: string) => { if (!disposed) setChart({ error: message }); worker.terminate(); };
    const timer = window.setTimeout(() => fail("Chart preview timed out. Your model response is preserved in JSON."), 8000);
    worker.onmessage = (event: MessageEvent<{ svg?: string; error?: string }>) => {
      window.clearTimeout(timer); worker.terminate();
      if (disposed) return;
      if (event.data.svg) {
        url = URL.createObjectURL(new Blob([event.data.svg], { type: "image/svg+xml" }));
        setChart({ url, svg: event.data.svg });
      } else fail(event.data.error || "Unable to render this chart.");
    };
    worker.onerror = () => { window.clearTimeout(timer); fail("Chart preview is unavailable. Inspect the original JSON instead."); };
    worker.postMessage({ spec, width: Math.max(220, (host.current?.clientWidth || 340) - 65) });
    return () => { disposed = true; window.clearTimeout(timer); worker.terminate(); if (url) URL.revokeObjectURL(url); };
  }, [signature]);
  return <section ref={host} className="output-section output-chart"><div className="output-chart-heading"><h4><BarChart3 size={14} />Dashboard</h4>{chart.svg && <button type="button" onClick={() => download(chart.svg!, "model-chart.svg", "image/svg+xml")}><Download size={12} />SVG</button>}</div>
    {chart.error ? <p role="status" className="output-notice">{chart.error}</p> : chart.url ? <div className="output-chart-scroll" tabIndex={0} role="region" aria-label="Rendered dashboard"><img src={chart.url} alt="Dashboard rendered from this model’s Vega-Lite specification. Exact values are available in the data table or original JSON." /></div> : <p className="output-note" role="status">Rendering chart…</p>}
    <p className="output-caption">This model’s specification{isRecord(candidate) && !hasChartData(candidate) && Array.isArray(fallbackRows) ? " · using returned cleaned rows where no chart data is specified" : ""}. Preview only, not a quality score.</p>
  </section>;
}
