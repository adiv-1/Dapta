import { useCallback, useEffect, useRef, useState } from "react";
import * as Sentry from "@sentry/react";
import {
  Activity, ArrowUp, BarChart3, BrainCircuit, Check, Clock3, Cpu,
  Database, FileText, Layers3, Loader2, PackageCheck, Plus, Router,
  ShieldCheck, Sparkles, Upload, UploadCloud, Wrench, X, ChevronDown,
} from "lucide-react";
import { api } from "./api";
import { Conversation, AdapterDrawer } from "./Conversation";
import { ModelOutput } from "./ModelOutput";
import { PollCoordinator, type PollTelemetry } from "./polling";
import type { Adapter, Benchmark, Condition, Job, Run } from "./types";

type Tab = "chat" | "performance";
type PerformanceView = "live" | "lift";

const labels: Record<string, string> = {
  qwen_lora: "Qwen LoRA",
  frontier_a: "Claude",
  frontier_b: "GPT",
};

function recordPoll(event: PollTelemetry) {
  Sentry.addBreadcrumb({
    category: "dapta.poll",
    message: `poll_${event.outcome}`,
    level: event.outcome === "failed" ? "warning" : "info",
    data: event,
  });
  // Capture the first and every third consecutive failure without attaching upstream response text.
  if (event.outcome === "failed"
      && (event.consecutive_failure_count === 1 || event.consecutive_failure_count % 3 === 0)) {
    Sentry.captureException(new Error("Dapta background polling failed"), {
      tags: { "dapta.error_category": "polling", "dapta.consecutive_failures": event.consecutive_failure_count },
    });
  }
}

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [view, setView] = useState<PerformanceView>("live");
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>();
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>();
  const [selectedJob, setSelectedJob] = useState<string>();
  const [suggestedAdapter, setSuggestedAdapter] = useState<Adapter>();
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [error, setError] = useState<string>();
  const polls = useRef(new PollCoordinator());
  const knownStages = useRef(new Map<string, string>());
  const closeLibrary = useCallback(() => setLibraryOpen(false), []);

  const refresh = useCallback(async () => polls.current.run(async () => {
    const results = await Promise.allSettled([api.adapters(), api.benchmarks(), api.runs(), api.jobs()]);
    const [adapterResult, benchmarkResult, runResult, jobResult] = results;
    if (adapterResult.status === "fulfilled") setAdapters(adapterResult.value);
    if (benchmarkResult.status === "fulfilled") {
      setBenchmarks(benchmarkResult.value);
      setSelectedBenchmark((current) => current || benchmarkResult.value[0]?.benchmark_id);
    }
    if (runResult.status === "fulfilled") {
      setRuns(runResult.value);
      setSelectedRun((current) => current || runResult.value[0]?.run_id);
    }
    if (jobResult.status === "fulfilled") setJobs(jobResult.value);
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
    setError(undefined);
  }, recordPoll), []);

  useEffect(() => { refresh().catch((reason) => setError(reason instanceof Error ? reason.message : "Could not load saved work")); }, [refresh]);
  useEffect(() => {
    const active = runs.some((run) => !["completed", "failed"].includes(run.status))
      || jobs.some((job) => !["ready", "failed"].includes(job.status));
    if (!active) return;
    const timer = window.setInterval(() => refresh().catch((reason) => {
      setError(reason instanceof Error ? reason.message : "Live updates temporarily failed");
    }), 1500);
    return () => window.clearInterval(timer);
  }, [runs, jobs, refresh]);
  useEffect(() => {
    for (const job of jobs) {
      const previous = knownStages.current.get(job.job_id);
      if (previous && previous !== job.status) Sentry.addBreadcrumb({
        category: "dapta.factory", message: "stage_transitioned", level: "info",
        data: { job_id: job.job_id, previous_stage: previous, next_stage: job.status },
      });
      knownStages.current.set(job.job_id, job.status);
    }
    Sentry.setTag("dapta.factory_stage", jobs.find((job) => job.job_id === selectedJob)?.status || "none");
  }, [jobs, selectedJob]);

  async function retry(jobId: string) {
    try {
      const job = jobs.find((item) => item.job_id === jobId);
      const retried = job?.failure_code === "EvaluationOutputTruncated"
        ? await api.repairEvaluation(jobId) : await api.retryJob(jobId);
      setJobs((current) => current.map((item) => item.job_id === jobId ? retried : item));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not retry this stage"); }
  }
  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#" aria-label="Dapta home" onClick={() => setTab("chat")}><span className="brand-lockup"><img src="/dapta-chameleon.png" alt="" /><span className="brand-word">Dapta</span></span></a>
      <nav aria-label="Main navigation"><TabButton active={tab === "chat"} onClick={() => setTab("chat")}>Chat</TabButton><TabButton active={tab === "performance"} onClick={() => setTab("performance")}>Performance</TabButton></nav>
      <span className="secure-pill"><span />Private workspace</span>
    </header>
    <main>
      {error && <div className="error-banner" role="alert">{error}<button onClick={() => refresh().catch(() => undefined)}>Try again</button></div>}
      <div hidden={tab !== "chat"}><Conversation adapters={adapters} jobs={jobs} runs={runs}
        onRun={(run) => { setRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]); setSelectedRun(run.run_id); }}
        onJob={(job) => setJobs((current) => [job, ...current.filter((item) => item.job_id !== job.job_id)])}
        renderRun={(run) => <ModelGrid run={run} running={!run || ["queued", "running"].includes(run.status)} />}
        renderJob={(job) => <FactoryTimeline job={job} onRetry={retry} />}
        openLibrary={() => setLibraryOpen(true)} selectedAdapter={suggestedAdapter} clearAdapter={() => setSuggestedAdapter(undefined)}
        openedJob={jobs.find((job) => job.job_id === selectedJob)} clearJob={() => setSelectedJob(undefined)}
      /></div>
      {tab === "performance" && <Performance view={view} setView={setView} runs={runs} selectedRun={selectedRun}
        setSelectedRun={setSelectedRun} run={runs.find((run) => run.run_id === selectedRun)} benchmarks={benchmarks}
        selectedBenchmark={selectedBenchmark} setSelectedBenchmark={setSelectedBenchmark}
        benchmark={benchmarks.find((item) => item.benchmark_id === selectedBenchmark)} />}
    </main>
    {libraryOpen && <AdapterDrawer adapters={adapters} jobs={jobs} onClose={closeLibrary}
      onSelect={(adapter) => { setSuggestedAdapter(adapter); setLibraryOpen(false); setTab("chat"); }}
      onOpenJob={(job) => { setSelectedJob(job.job_id); setLibraryOpen(false); setTab("chat"); }} />}
  </div>;
}

export function ModelGrid({ run, running }: { run?: Run; running: boolean }) {
  const names = ["qwen_lora", "frontier_a", "frontier_b"];
  const now = useLiveNow(running);
  const elapsed = run ? Math.max(0, (now - Date.parse(run.created_at)) / 1000) : 0;
  const complete = names.every((name) => run?.results[name]?.status === "completed");
  const metrics = [
    { key: "speed", label: "Time", unit: "seconds", value: (c?: Condition) => c?.status === "completed" && c.metrics.latency_ms != null ? c.metrics.latency_ms / 1000 : null, format: (v: number) => v.toFixed(2) },
    { key: "tokens", label: "Tokens", unit: "input + output", value: (c?: Condition) => c?.status === "completed" && c.metrics.input_tokens != null && c.metrics.output_tokens != null ? c.metrics.input_tokens + c.metrics.output_tokens : null, format: (v: number) => v.toLocaleString() },
    { key: "cost", label: "Cost", unit: "USD / request", value: (c?: Condition) => c?.status === "completed" && c.metrics.cost_usd != null && c.metrics.cost_usd !== "" ? Number(c.metrics.cost_usd) : null, format: (v: number) => "$" + v.toFixed(5) },
  ];
  return <section className="result-workspace" aria-label="Model comparison">
    <div className="result-heading"><div><span className="welcome-kicker">{running ? "Comparison in progress" : "The difference, measured"}</span><h2>Same task. Every model.</h2></div><span className="result-note">Lower time, tokens & cost is better</span></div>
    <div className="comparison-scroll"><table className="comparison-matrix">
      <thead><tr><th><span>One input.</span><small>Three approaches.</small></th>{names.map((name) => {
        const condition = run?.results[name];
        const status = condition?.status;
        return <th key={name} className={name}><div className="comparison-model"><i /><strong>{condition?.label || labels[name]}</strong></div><span className={status === "failed" ? "result-failed" : "result-status"}>{status === "completed" ? <><Check size={12} />Complete</> : status === "failed" ? <><X size={12} />Failed</> : running ? <><Loader2 size={12} className="spin" />Working{elapsed > 0 ? " · " + Math.floor(elapsed) + "s elapsed" : ""}</> : "No result"}</span></th>;
      })}</tr></thead><tbody>{metrics.map((metric) => {
        const values = names.map((name) => metric.value(run?.results[name]));
        const allKnown = complete && values.every((value) => value != null && Number.isFinite(value));
        const best = allKnown ? Math.min(...values as number[]) : null;
        const tied = allKnown && values.every((value) => value === best);
        const maximum = Math.max(...values.filter((v): v is number => v != null), 0);
        return <tr key={metric.key}><th scope="row"><strong>{metric.label}</strong><span>{metric.unit}</span></th>{names.map((name, index) => {
          const value = values[index];
          const winner = best != null && value === best && !tied;
          const c = run?.results[name];
          return <td key={name} className={winner ? "metric-winner" : ""}><div className="metric-number">{value != null && Number.isFinite(value) ? metric.format(value) : "—"}{winner && <span className="metric-best">Lowest</span>}{tied && <span className="metric-tie">Equal</span>}</div>{metric.key === "tokens" && c?.status === "completed" && value != null && <small className="token-detail">{c.metrics.input_tokens?.toLocaleString()} input · {c.metrics.output_tokens?.toLocaleString()} output</small>}<div className="metric-line"><span className={name} style={{ width: value != null && maximum ? Math.max(2, value / maximum * 100) + "%" : "0%" }} /></div></td>;
        })}</tr>;
      })}<tr className="quality-row"><th scope="row"><strong>Quality check</strong><span>Saved evaluation</span></th>{names.map((name) => {
        const condition = run?.results[name]; const score = condition?.metrics.accuracy;
        return <td key={name}>{condition?.status === "completed" && score != null ? <span className="quality-value"><ShieldCheck size={14} />{score.toFixed(1)}%</span> : <span className="quality-unknown">Not scored</span>}</td>;
      })}</tr></tbody></table></div>
    <details className="comparison-outputs" open><summary><span>Model responses</span><ChevronDown size={16} /></summary><div className="response-grid">{names.map((name) => {
      const condition = run?.results[name];
      return <article key={name} className="response-panel" aria-label={(condition?.label || labels[name]) + " response"}><span className="response-label">{condition?.label || labels[name]}</span>{condition?.status === "failed" ? <p role="alert" className="reply-error">{condition.error || "This model could not complete the request."}</p> : condition?.status === "completed" ? <ModelOutput key={run?.run_id} output={condition.output} modelLabel={condition.label || labels[name]} /> : <div className="response-wait"><Loader2 className={running ? "spin" : ""} size={18} /><span>{running ? "Waiting for the model’s response…" : "No response available."}</span></div>}</article>;
    })}</div></details>
    <details className="comparison-method"><summary>How this comparison is measured</summary><p>All models receive the same task and input. The LoRA specialist uses its compact skill prompt; frontier models receive the full skill. Time and usage come from completed calls. Cost is the recorded request cost; unavailable costs stay blank. Quality is a per-document evaluation, not a held-out benchmark. Provider tokenizers differ.</p><div>{names.map((name) => <span key={name}>{run?.results[name]?.model || labels[name]}{run?.results[name]?.metrics.accuracy_label ? " · " + run.results[name].metrics.accuracy_label : ""}</span>)}</div></details>
  </section>;
}

function Performance(props: {
  view: PerformanceView; setView: (view: PerformanceView) => void; runs: Run[];
  selectedRun?: string; setSelectedRun: (id: string) => void; run?: Run;
  benchmarks: Benchmark[]; selectedBenchmark?: string; setSelectedBenchmark: (id: string) => void;
  benchmark?: Benchmark;
}) {
  const conditions = props.view === "live" ? liveConditions(props.run) : benchmarkConditions(props.benchmark);
  return (
    <section className="page performance-page">
      <div className="page-title"><span><Activity size={18} /> Saved evidence</span><h1>Every run tells a story.</h1><p>Review the seconds, tokens, dollars, and quality behind your work.</p></div>
      <div className="segmented"><button className={props.view === "live" ? "active" : ""} onClick={() => props.setView("live")}>Comparison runs</button><button className={props.view === "lift" ? "active" : ""} onClick={() => props.setView("lift")}>Training benchmarks</button></div>
      <div className="run-pills">
        {props.view === "lift" && props.benchmarks.map((item) => <button className={props.selectedBenchmark === item.benchmark_id ? "active" : ""} onClick={() => props.setSelectedBenchmark(item.benchmark_id)} key={item.benchmark_id}>{item.name}<small>{item.test_documents} frozen cases</small></button>)}
      </div>
      {props.view === "live" ? props.run ? <ModelGrid run={props.run} running={["queued", "running"].includes(props.run.status)} /> : <EmptyState text="Your completed comparisons will appear here." /> : conditions.length ? <><Summary conditions={conditions} /><div className="chart-grid"><MetricChart title="Speed" suffix="s" lower conditions={conditions} field="time" /><MetricChart title="Cost" suffix="$" lower conditions={conditions} field="cost" /><MetricChart title="Accuracy" suffix="%" conditions={conditions} field="accuracy" /></div></> : <EmptyState text="No training benchmarks yet." />}
      {props.view === "lift" && props.benchmark && <p className="fine-print">Frozen test SHA-256: <code>{props.benchmark.test_sha256}</code>. Accuracy is labeled field F1; schema validity and exact-form rate remain available in the saved benchmark.</p>}
      {props.view === "live" && <RunHistoryTable runs={props.runs} selectedRun={props.selectedRun} setSelectedRun={props.setSelectedRun} />}
    </section>
  );
}

type ChartCondition = { name: string; label: string; time: number; cost: number; accuracy: number };

function Summary({ conditions }: { conditions: ChartCondition[] }) {
  return <div className="summary-strip"><div><small>Fastest</small><strong>{winningLabel(conditions, "time", true)}</strong></div><div><small>Lowest cost</small><strong>{winningLabel(conditions, "cost", true)}</strong></div><div><small>Most accurate</small><strong>{winningLabel(conditions, "accuracy", false)}</strong></div></div>;
}

function MetricChart({ title, suffix, lower, conditions, field }: { title: string; suffix: string; lower?: boolean; conditions: ChartCondition[]; field: "time" | "cost" | "accuracy" }) {
  const max = Math.max(...conditions.map((item) => item[field]), 0.00001);
  return <article className="chart"><div className="chart-head"><h3>{title}</h3><span>{lower ? "Lower is better" : "Higher is better"}</span></div>{conditions.map((item) => <div className="bar-row" key={item.name}><div><strong>{item.label}</strong><span>{field === "cost" ? "$" : ""}{formatMetric(item[field], field)}{field === "cost" ? "" : suffix}</span></div><div className="track"><span className={item.name} style={{ width: `${Math.max(3, item[field] / max * 100)}%` }} /></div></div>)}</article>;
}

function RunHistoryTable({ runs, selectedRun, setSelectedRun }: { runs: Run[]; selectedRun?: string; setSelectedRun: (id: string) => void }) {
  const ordered = runs.filter((run) => run.status === "completed").sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
  return <section className="history-section"><div className="history-heading"><div><span>Previous completed runs</span><h2>Comparison history</h2></div><small>Most recent first</small></div>{ordered.length ? <div className="table-scroll"><table><thead><tr><th>Date &amp; time</th><th>Speed</th><th>Tokens</th><th>Cost</th><th>Quality</th><th>Overall winner</th></tr></thead><tbody>{ordered.map((run) => {
    const conditions = liveConditions(run);
    const tokens = Object.entries(run.results).filter(([, value]) => value.status === "completed" && value.metrics.input_tokens != null && value.metrics.output_tokens != null).map(([name, value]) => ({ name, label: value.label || labels[name] || name, time: 0, cost: 0, accuracy: Number(value.metrics.input_tokens) + Number(value.metrics.output_tokens) }));
    const overall = overallWinner(conditions);
    return <tr key={run.run_id} className={selectedRun === run.run_id ? "selected" : ""} onClick={() => setSelectedRun(run.run_id)}><td><button className="history-select" onClick={() => setSelectedRun(run.run_id)} aria-label={`Review comparison from ${new Date(run.created_at).toLocaleString()}`}>{new Date(run.created_at).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })}</button><span>{new Date(run.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span></td><td><strong>{winningLabel(conditions, "time", true)}</strong><span>{conditions.length ? `${Math.min(...conditions.map((item) => item.time)).toFixed(2)}s` : "—"}</span></td><td><strong>{tokens.length === 3 ? winningLabel(tokens, "accuracy", true) : "—"}</strong><span>{tokens.length === 3 ? Math.min(...tokens.map((item) => item.accuracy)).toLocaleString() : "Usage unavailable"}</span></td><td><strong>{winningLabel(conditions, "cost", true)}</strong><span>{conditions.length ? `$${Math.min(...conditions.map((item) => item.cost)).toFixed(5)}` : "—"}</span></td><td><strong>{winningLabel(conditions, "accuracy", false)}</strong><span>{conditions.length ? `${Math.max(...conditions.map((item) => item.accuracy)).toFixed(1)}%` : "—"}</span></td><td><span className="winner-pill">{overall}</span></td></tr>;
  })}</tbody></table></div> : <EmptyState text="Completed comparisons will appear here." />}</section>;
}

function winningLabel(items: ChartCondition[], field: "time" | "cost" | "accuracy", lower: boolean) {
  if (!items.length) return "—";
  const best = lower ? Math.min(...items.map((item) => item[field])) : Math.max(...items.map((item) => item[field]));
  const winners = items.filter((item) => Math.abs(item[field] - best) < 0.000001);
  return winners.length === items.length && items.length > 1 ? "Equal" : winners.length > 1 ? `${winners.length}-way tie` : winners[0].label;
}

function overallWinner(items: ChartCondition[]) {
  if (!items.length) return "Incomplete";
  const points = new Map(items.map((item) => [item.label, 0]));
  (["time", "cost", "accuracy"] as const).forEach((field) => {
    const lower = field !== "accuracy";
    const best = lower ? Math.min(...items.map((item) => item[field])) : Math.max(...items.map((item) => item[field]));
    const winners = items.filter((item) => Math.abs(item[field] - best) < 0.000001);
    winners.forEach((winner) => points.set(winner.label, (points.get(winner.label) || 0) + 1 / winners.length));
  });
  const high = Math.max(...points.values());
  const winners = [...points.entries()].filter(([, score]) => Math.abs(score - high) < 0.000001).map(([label]) => label);
  return winners.length > 1 ? "Equal" : winners[0];
}

const factoryStages = [
  { key: "queued", label: "Router", poweredBy: "JEV", detail: "Adapter match and trainability", icon: Router },
  { key: "architecting", label: "Skill architect", poweredBy: "Kimi K3", detail: "Skill spec, schema, evaluation plan", icon: BrainCircuit },
  { key: "integrating", label: "Skill integration", poweredBy: "Devin", detail: "SKILL.md, validator spec, tests, Hugging Face bundle", icon: PackageCheck },
  { key: "building_dataset", label: "Data builder", poweredBy: "Kimi K3", detail: "TRAIN, DEV, frozen TEST, OOD", icon: Database },
  { key: "validating", label: "Dataset validator", poweredBy: "Dapta", detail: "Schema, leakage, duplicates, hashes", icon: ShieldCheck },
  { key: "awaiting_compute", label: "Compute gate", poweredBy: "Dapta", detail: "Budget and GPU readiness", icon: Clock3 },
  { key: "training", label: "QLoRA training", poweredBy: "RunPod A40", detail: "Training, GPU telemetry, cost", icon: Cpu },
  { key: "evaluating", label: "Independent evaluator", poweredBy: "Dapta evals", detail: "Base vs Skill vs LoRA", icon: BarChart3 },
  { key: "publishing", label: "Registry publish", poweredBy: "Hugging Face", detail: "Weights, metrics, manifest", icon: UploadCloud },
] as const;

export function Registry({ adapters, jobs, selectedJob, setSelectedJob, onTest, onRetry }: {
  adapters: Adapter[]; jobs: Job[]; selectedJob?: string;
  setSelectedJob: (id: string) => void; onTest: (adapter?: Adapter) => void;
  onRetry: (jobId: string) => Promise<void>;
}) {
  const job = jobs.find((item) => item.job_id === selectedJob) || jobs[0];
  return (
    <section className="page">
      <div className="page-title"><span><Layers3 size={18} /> Factory</span><h1>Skills & Adapters</h1><p>Watch the factory make each decision, train the specialist, and preserve its evidence.</p></div>
      {jobs.length > 0 && (
        <div className="job-pills">
          {jobs.map((item) => <button key={item.job_id} className={job?.job_id === item.job_id ? "active" : ""} onClick={() => setSelectedJob(item.job_id)}>{item.skill_id || "New specialist"}<small>{stageLabel(item.status)}</small></button>)}
        </div>
      )}
      {job && <FactoryTimeline job={job} onRetry={onRetry} />}
      <div className="registry-heading"><div><h2>Adapter registry</h2><p>Ready specialists restored from Hugging Face.</p></div><button onClick={() => onTest()}><Plus size={16} />New specialist</button></div>
      <div className="registry-grid">
        {adapters.map((adapter) => {
          const needsRepair = adapter.metadata.benchmark_status === "needs_repair";
          return <article className="adapter-card" key={adapter.adapter_id}><div className="adapter-icon"><FileText /></div><span className={needsRepair ? "building-pill" : "ready"}><span />{needsRepair ? "Benchmark repair" : "Ready"}</span><h2>{String(adapter.metadata.name || adapter.skill_id)}</h2><p>{String(adapter.metadata.description || "Reusable LoRA adapter")}</p><dl><div><dt>Base model</dt><dd>{adapter.base_model.split("/").pop()}</dd></div><div><dt>Benchmark</dt><dd>{needsRepair ? "Pending rerun" : `${(Number(adapter.metadata.benchmark_accuracy || 0) * 100).toFixed(1)}% F1`}</dd></div><div><dt>Version</dt><dd>{adapter.version}</dd></div><div><dt>Training cost</dt><dd>${Number(adapter.metadata.training_cost_usd || 0).toFixed(3)}</dd></div></dl><button onClick={() => onTest(adapter)}>Test adapter</button></article>;
        })}
        {!adapters.length && <button className="new-card" onClick={() => onTest()}><Plus /><span>Create a new specialist</span></button>}
      </div>
    </section>
  );
}

function FactoryTimeline({ job, onRetry }: { job: Job; onRetry: (jobId: string) => Promise<void> }) {
  const now = useLiveNow(!["ready", "failed"].includes(job.status));
  const backendOrder = ["queued", "architecting", "integrating", "building_dataset", "validating", "awaiting_compute", "provisioning_gpu", "training", "evaluating", "publishing", "ready"];
  const failedFrom = [...(job.events || [])].reverse().find((event) => event.to_status === "failed")?.from_status;
  const effectiveStatus = job.status === "failed" ? failedFrom || "queued" : job.status;
  const current = Math.max(0, backendOrder.indexOf(effectiveStatus));
  const progress = job.progress;
  const published = Boolean(job.publish_result);
  const overall = job.status === "ready" ? 100 : published ? 96 : Math.max(4, current / (backendOrder.length - 1) * 100);
  const finishedAt = ["ready", "failed"].includes(job.status)
    ? latestJobTimestamp(job) : new Date(now).toISOString();
  const totalSeconds = secondsBetween(job.created_at, finishedAt);
  const totalCost = Number(job.usage?.total_cost_usd || progress?.cost_usd || 0);
  const eta = estimateJobEta(job);
  const repairs = repairSuggestions(job);
  return (
    <article className={`factory-panel ${job.status}`}>
      <div className="factory-head"><div><span className="factory-state">{job.status === "ready" ? <Check size={14} /> : job.status === "failed" ? <X size={14} /> : <Loader2 className="spin" size={14} />}{stageLabel(job.status)}</span><h2>{job.skill_id || "New LoRA specialist"}</h2><p>{job.workflow}</p></div><div className="factory-percent"><strong>{Math.round(overall)}%</strong><span>Round {job.dataset_round || 1}</span></div></div>
      <div className="factory-totals">
        <div><small>End-to-end timer</small><strong>{formatDuration(totalSeconds)}</strong></div>
        <div><small>Universal cost</small><strong>${totalCost.toFixed(4)}</strong></div>
        <div><small>Estimated completion</small><strong>{eta == null ? (published ? "Published" : "Decision needed") : `~${formatDuration(eta)}`}</strong></div>
        <div><small>Devin usage</small><strong>{Number(job.devin_session?.acus_consumed || job.devin_session?.on_demand_usage || 0).toFixed(2)} ACUs</strong></div>
      </div>
      <div className="factory-progress"><span style={{ width: `${overall}%` }} /></div>
      <SentryDecision job={job} />
      <div className="factory-workspace">
        <div className="pipeline">
          {factoryStages.map((stage) => {
          const stageBackendIndex = backendOrder.indexOf(stage.key);
          const completed = job.status === "ready"
            || (stage.key === "queued" && Boolean(job.route))
            || (stage.key === "training" && Boolean(job.training))
            || (stage.key === "evaluating" && Boolean(job.deterministic_evaluation))
            || (stage.key === "publishing" && published)
            || current > stageBackendIndex;
          const active = !completed && (
            stage.key === job.status
            || (stage.key === "training" && job.status === "provisioning_gpu")
          );
          const failed = job.status === "failed" && (
            stage.key === effectiveStatus
            || (stage.key === "training" && effectiveStatus === "provisioning_gpu")
          );
          const Icon = stage.icon;
            const devinLabel = ["waiting_for_user", "inactivity"].includes(job.devin_session?.status_detail || "")
              ? "bundle complete · saving outside GitHub"
              : job.devin_session?.status;
            const timing = stageTiming(job, stage.key, now);
            const stageCost = costForStage(job, stage.key);
            const result = stageResult(job, stage.key);
            return <div className={`pipeline-step ${completed ? "complete" : active ? "active" : failed ? "failed" : "pending"}`} key={stage.key}><div className="pipeline-icon">{completed ? <Check /> : failed ? <X /> : active ? <Loader2 className="spin" /> : <Icon />}</div><div><div className="stage-title"><strong>{stage.label}</strong><span className="model-pill">{stage.poweredBy}</span></div><span>{stage.detail}</span><div className="stage-metrics"><span>{timing == null ? "Not started" : formatDuration(timing)}</span><span>${stageCost.toFixed(4)}</span><span className="sentry-trace"><i />Sentry traced</span>{active && eta != null && <span>ETA ~{formatDuration(eta)}</span>}</div>{result && <small className="stage-result">Result: {result}</small>}{stage.key === "queued" && job.route && <small>{job.route.decision} · {Math.round(Number(job.route.confidence || 0) * 100)}% · {job.route.reason}</small>}{stage.key === "integrating" && job.devin_session && <small>{devinLabel}{job.devin_session.pull_request_url ? " · PR ready" : ""}</small>}{stage.key === "validating" && job.validation && <small>Frozen TEST: {String(job.validation.frozen_test_sha256 || "pending").slice(0, 14)}…</small>}{stage.key === "training" && (active || completed) && progress && <TrainingTelemetry progress={progress} />}{stage.key === "publishing" && job.publish_result && <a href={job.publish_result.artifact_uri.replace("hf://", "https://huggingface.co/")} target="_blank" rel="noreferrer">Open adapter ↗</a>}</div></div>;
          })}
        </div>
        <AgentOutputPanel job={job} />
      </div>
      {job.failure_message && <div className="factory-error repair-card"><Wrench size={18} /><div><strong>{job.failure_code}</strong><span>{job.failure_message}</span>{repairs.length > 0 && <ul>{repairs.map((repair) => <li key={repair}>{repair}</li>)}</ul>}</div><button onClick={() => onRetry(job.job_id)}>{job.failure_code === "EvaluationOutputTruncated" ? "Apply repair & rerun benchmark" : "Retry failed stage"}</button></div>}
      {job.deterministic_evaluation?.conditions && <FactoryResults evaluation={job.deterministic_evaluation} />}
    </article>
  );
}

function SentryDecision({ job }: { job: Job }) {
  const evaluation = job.deterministic_evaluation;
  const suggestions = repairSuggestions(job);
  let decision = "Decision pending";
  let explanation = "Sentry is tracing every factory stage, model call, latency, token, and cost signal. Retraining is decided only after the frozen evaluation finishes.";
  let tone = "pending";
  if (evaluation?.evaluation_verified === false || evaluation?.benchmark_status === "needs_repair") {
    decision = "Do not retrain";
    explanation = "The adapter weights completed successfully. Sentry and the evaluator identify truncated benchmark output, so Dapta reruns evaluation with a larger output budget instead of paying to retrain.";
    tone = "repair";
  } else if (evaluation?.passed) {
    decision = "No retraining needed";
    explanation = "The frozen benchmark passed and Sentry shows the completed inference path. The adapter is ready to publish without another training cycle.";
    tone = "healthy";
  } else if (evaluation && !evaluation.passed) {
    decision = "Apply feedback, then retrain";
    explanation = suggestions[0] || "Quality missed the frozen acceptance threshold. Dapta uses evaluator feedback on DEV data, records the repair in Sentry, and starts a new training round without touching frozen TEST.";
    tone = "retrain";
  } else if (job.status === "failed") {
    decision = "Fix the failed stage first";
    explanation = `${job.failure_code || "Factory failure"} is an operational failure captured by Sentry. Repair or retry that stage before deciding whether the model itself needs retraining.`;
    tone = "retrain";
  }
  return <section className={`sentry-decision ${tone}`}><div className="sentry-heading"><div><ShieldCheck size={17} /><span>Sentry feedback → retraining decision</span></div><strong>{decision}</strong></div><p>{explanation}</p><div className="sentry-signals"><span><i />Errors {job.failure_code ? `· ${job.failure_code}` : "· clear"}</span><span><i />Traces · stage linked</span><span><i />AI telemetry · tokens + cost</span></div></section>;
}

const attemptStage: Record<string, string> = {
  architecting: "skill_architect", integrating: "skill_integration",
  building_dataset: "data_builder", validating: "dataset_validator",
  training: "adapter_training", evaluating: "independent_evaluator",
  publishing: "registry_publish",
};

function useLiveNow(active: boolean) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active]);
  return now;
}

function latestJobTimestamp(job: Job) {
  const candidates = [job.updated_at, ...(job.artifacts || []).map((item) => item.created_at)];
  return candidates.sort((a, b) => Date.parse(b) - Date.parse(a))[0] || job.updated_at;
}

function secondsBetween(start?: string, end?: string) {
  if (!start || !end) return 0;
  return Math.max(0, Math.floor((Date.parse(end) - Date.parse(start)) / 1000));
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds)) return "—";
  const value = Math.max(0, Math.round(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const rest = value % 60;
  return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m ${rest}s` : `${rest}s`;
}

function stageTiming(job: Job, key: string, now: number) {
  if (key === "queued" && job.route) return secondsBetween(job.created_at, job.events?.find((event) => event.to_status === "architecting")?.occurred_at || job.updated_at);
  const attempt = [...(job.stage_attempts || [])].reverse().find((item) => item.stage === attemptStage[key]);
  return attempt ? secondsBetween(attempt.claimed_at, attempt.completed_at || new Date(now).toISOString()) : null;
}

function costForStage(job: Job, key: string) {
  if (key === "training") return Number(job.progress?.cost_usd || job.training?.training_cost_usd || job.usage?.compute_cost_usd || 0);
  const names: Record<string, string[]> = {
    architecting: ["skill_architect"], building_dataset: ["data_builder"],
    evaluating: ["independent_evaluator", "base_no_skill", "base_with_skill", "lora_short_prompt"],
  };
  return (job.usage?.conditions || [])
    .filter((item) => (names[key] || []).includes(item.condition))
    .reduce((sum, item) => sum + Number(item.cost_usd || 0), 0);
}

function stageResult(job: Job, key: string) {
  const output = [...(job.agent_outputs || [])].reverse().find((item) => item.stage === key || item.stage === attemptStage[key]);
  if (output) return output.summary;
  if (key === "training" && job.training) return "LoRA weights trained; four held-out generations completed";
  if (key === "evaluating" && job.deterministic_evaluation) return job.deterministic_evaluation.passed ? "Frozen benchmark passed" : "Benchmark needs repair";
  if (key === "publishing" && job.publish_result) return "Weights and skill bundle saved to Hugging Face";
  return undefined;
}

function repairSuggestions(job: Job) {
  const evaluation = job.deterministic_evaluation;
  if (evaluation?.suggested_repairs?.length) return evaluation.suggested_repairs;
  if (evaluation?.dev_feedback?.instruction) return [evaluation.dev_feedback.instruction];
  return [];
}

function estimateJobEta(job: Job) {
  if (["ready", "failed"].includes(job.status)) return null;
  if (job.progress?.eta_seconds != null) return job.progress.eta_seconds;
  const estimates: Record<string, number> = {
    queued: 1500, architecting: 1440, integrating: 1200, building_dataset: 900,
    validating: 720, awaiting_compute: 660, provisioning_gpu: 600, training: 540,
    evaluating: 240, publishing: 60,
  };
  return estimates[job.status] ?? 600;
}

function AgentOutputPanel({ job }: { job: Job }) {
  const outputs = job.agent_outputs || [];
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    setSelected(Math.max(0, outputs.length - 1));
  }, [job.job_id, outputs.length]);

  // The effect above corrects `selected` only after this render; clamp so a job with fewer outputs
  // (another job selected, or a newer job taking over the list) never reads past the end.
  const shown = Math.min(selected, outputs.length - 1);
  const current = outputs[shown];
  return (
    <aside className="agent-output-panel">
      <div className="agent-output-head">
        <div><span className="live-dot" />Live agent output</div>
        <small>Sanitized · saved to Supabase</small>
      </div>
      {outputs.length ? (
        <>
          <div className="agent-output-tabs">
            {outputs.map((output, index) => (
              <button
                className={index === shown ? "active" : ""}
                key={`${output.stage}-${index}`}
                onClick={() => setSelected(index)}
              >
                <span>{output.agent}</span>
                <small>{output.summary}</small>
              </button>
            ))}
          </div>
          <div className="agent-output-body">
            <div><strong>{current.agent}</strong><span>{current.summary}</span></div>
            <pre>{JSON.stringify(current.output, null, 2)}</pre>
          </div>
        </>
      ) : (
        <div className="agent-output-empty">
          <Loader2 className="spin" />
          <strong>Waiting for the first artifact</strong>
          <span>Agent decisions will appear here as they are saved.</span>
        </div>
      )}
    </aside>
  );
}

function TrainingTelemetry({ progress }: { progress: NonNullable<Job["progress"]> }) {
  const percent = progress.percent || (progress.current && progress.total ? progress.current / progress.total * 100 : 0);
  return <div className="telemetry"><div className="telemetry-bar"><span style={{ width: `${percent}%` }} /></div><div><span>{progress.message || "Training"}</span>{progress.loss != null && <span>loss {progress.loss.toFixed(4)}</span>}{progress.eta_seconds != null && <span>ETA {Math.ceil(progress.eta_seconds / 60)}m</span>}{progress.gpu_utilization_percent != null && <span>GPU {progress.gpu_utilization_percent}%</span>}{progress.cost_usd != null && <span>${Number(progress.cost_usd).toFixed(3)}</span>}</div></div>;
}

function FactoryResults({ evaluation }: { evaluation: NonNullable<Job["deterministic_evaluation"]> }) {
  const labels: Record<string, string> = { base_no_skill: "Qwen Base", base_with_skill: "Qwen + Skill", lora_short_prompt: "Qwen LoRA" };
  if (evaluation.evaluation_verified === false || evaluation.benchmark_status === "needs_repair") {
    return <div className="factory-results"><div className="results-title"><h3>Frozen TEST results</h3><span className="fail">NOT SCORED</span></div><p className="benchmark-notice">All four LoRA generations completed, but the evaluator capped every output before valid JSON finished. These outputs are intentionally not reported as 0% scores. Apply the benchmark repair above to score the saved weights without retraining.</p></div>;
  }
  return <div className="factory-results"><div className="results-title"><h3>Frozen TEST results</h3><span className={evaluation.passed ? "pass" : "fail"}>{evaluation.passed ? "PASS" : "REPAIR"}</span></div><div>{Object.entries(evaluation.conditions || {}).map(([name, result]) => <section key={name}><strong>{labels[name] || name}</strong><b>{((result.field_f1 || 0) * 100).toFixed(1)}% F1</b><small>{((result.schema_pass_rate || 0) * 100).toFixed(1)}% schema · {((result.exact_form_rate || 0) * 100).toFixed(1)}% exact</small></section>)}</div></div>;
}

function TabButton({ active, children, onClick }: { active: boolean; children: React.ReactNode; onClick: () => void }) { return <button className={active ? "active" : ""} onClick={onClick}>{children}</button>; }
function EmptyState({ text }: { text: string }) { return <div className="empty"><Sparkles /><p>{text}</p></div>; }
function liveConditions(run?: Run): ChartCondition[] { return run ? Object.entries(run.results).filter(([, value]) => value.status === "completed").map(([name, value]) => ({ name, label: value.label || labels[name] || name, time: (value.metrics.latency_ms || 0) / 1000, cost: Number(value.metrics.cost_usd || 0), accuracy: value.metrics.accuracy || 0 })) : []; }
function benchmarkConditions(benchmark?: Benchmark): ChartCondition[] { return benchmark ? Object.entries(benchmark.conditions).map(([name, value]) => ({ name, label: String(value.label), time: Number(value.latency_seconds || 0), cost: Number(value.cost_usd || 0), accuracy: Number(value.accuracy || 0) })) : []; }
function formatMetric(value: number, field: string) { return field === "cost" ? value.toFixed(5) : field === "time" ? value.toFixed(2) : value.toFixed(1); }
function stageLabel(stage: string) { return stage.split("_").map((word) => word[0]?.toUpperCase() + word.slice(1)).join(" "); }
