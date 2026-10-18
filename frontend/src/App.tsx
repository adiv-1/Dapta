import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowUp, BarChart3, BrainCircuit, Check, Clock3, Cpu,
  Database, FileText, Layers3, Loader2, PackageCheck, Plus, Router,
  ShieldCheck, Sparkles, Upload, UploadCloud, X,
} from "lucide-react";
import { api } from "./api";
import type { Adapter, Benchmark, Condition, Job, Run } from "./types";

type Tab = "chat" | "performance" | "skills";
type PerformanceView = "live" | "lift";

const labels: Record<string, string> = {
  qwen_lora: "Qwen LoRA",
  frontier_a: "Claude",
  frontier_b: "GPT",
};

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
  const [prompt, setPrompt] = useState("Extract the NDA into a normalized legal intake form.");
  const [documentText, setDocumentText] = useState("");
  const [fileName, setFileName] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    const [nextAdapters, nextBenchmarks, nextRuns, nextJobs] = await Promise.all([
      api.adapters(), api.benchmarks(), api.runs(), api.jobs(),
    ]);
    setAdapters(nextAdapters);
    setBenchmarks(nextBenchmarks);
    setRuns(nextRuns);
    setJobs(nextJobs);
    setSelectedRun((current) => current || nextRuns[0]?.run_id);
    setSelectedBenchmark((current) => current || nextBenchmarks[0]?.benchmark_id);
    setSelectedJob((current) => current || nextJobs[0]?.job_id);
  };

  useEffect(() => { refresh().catch((reason) => setError(reason.message)); }, []);

  useEffect(() => {
    const active = runs.some((run) => !["completed", "failed"].includes(run.status))
      || jobs.some((job) => !["ready", "failed"].includes(job.status));
    if (!active) return;
    const timer = window.setInterval(() => refresh().catch(() => undefined), 1500);
    return () => window.clearInterval(timer);
  }, [runs, jobs]);

  const currentRun = runs.find((run) => run.run_id === selectedRun);
  const currentBenchmark = benchmarks.find((item) => item.benchmark_id === selectedBenchmark);

  async function onFile(file?: File) {
    if (!file) return;
    setBusy(true);
    setError(undefined);
    try {
      const extracted = await api.extract(file);
      setDocumentText(extracted.text);
      setFileName(extracted.filename);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not read the file");
    } finally {
      setBusy(false);
    }
  }

  async function submit() {
    if (!prompt.trim()) return;
    setBusy(true);
    setError(undefined);
    try {
      const route = await api.route(prompt);
      if (route.decision === "unsupported") {
        throw new Error(route.reason || "This workflow is not supported.");
      }
      if (route.decision === "existing_adapter" && route.adapter_id) {
        if (!documentText.trim()) {
          throw new Error("Paste or upload a document before running this adapter.");
        }
        if (route.requires_confirmation && !window.confirm(
          `Use the matched adapter ${route.adapter_id}?\n\n${route.reason}`,
        )) return;
        const run = await api.compare(prompt, documentText, route.adapter_id, fileName);
        setRuns((current) => [run, ...current]);
        setSelectedRun(run.run_id);
      } else {
        const job = await api.createJob(prompt, route);
        setJobs((current) => [job, ...current]);
        setSelectedJob(job.job_id);
        setTab("skills");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#"><span className="brand-mark">D</span>Dapta</a>
        <nav>
          <TabButton active={tab === "chat"} onClick={() => setTab("chat")}>Chat</TabButton>
          <TabButton active={tab === "performance"} onClick={() => setTab("performance")}>Performance</TabButton>
          <TabButton active={tab === "skills"} onClick={() => setTab("skills")}>Skills & Adapters</TabButton>
        </nav>
        <span className="secure-pill"><span />Invite-only demo</span>
      </header>

      <main>
        {error && <div className="error-banner">{error}</div>}
        {tab === "chat" && (
          <Chat
            prompt={prompt} setPrompt={setPrompt} documentText={documentText}
            setDocumentText={setDocumentText} fileName={fileName} busy={busy}
            fileInput={fileInput} onFile={onFile} submit={submit} run={currentRun}
          />
        )}
        {tab === "performance" && (
          <Performance
            view={view} setView={setView} runs={runs} selectedRun={selectedRun}
            setSelectedRun={setSelectedRun} run={currentRun} benchmarks={benchmarks}
            selectedBenchmark={selectedBenchmark} setSelectedBenchmark={setSelectedBenchmark}
            benchmark={currentBenchmark}
          />
        )}
        {tab === "skills" && (
          <Registry
            adapters={adapters}
            jobs={jobs}
            selectedJob={selectedJob}
            setSelectedJob={setSelectedJob}
            onTest={() => setTab("chat")}
            onRetry={async (jobId) => {
              const retried = await api.retryJob(jobId);
              setJobs((current) => current.map((item) => item.job_id === jobId ? retried : item));
            }}
          />
        )}
      </main>
    </div>
  );
}

function Chat(props: {
  prompt: string; setPrompt: (value: string) => void; documentText: string;
  setDocumentText: (value: string) => void; fileName?: string; busy: boolean;
  fileInput: React.RefObject<HTMLInputElement>; onFile: (file?: File) => void;
  submit: () => void; run?: Run;
}) {
  const compact = Boolean(props.run);
  return (
    <section className={compact ? "chat-page compact" : "chat-page"}>
      {!compact && <div className="hero"><span className="eyebrow"><Sparkles size={14} /> Skills become models</span><h1>Run the work.<br /><em>Measure the difference.</em></h1><p>Give the same document to a specialized open model and two frontier models. Dapta records every token, second, dollar, and result.</p></div>}
      <div className="composer">
        <textarea value={props.prompt} onChange={(event) => props.setPrompt(event.target.value)} aria-label="Workflow" />
        <textarea className="document-input" value={props.documentText} onChange={(event) => props.setDocumentText(event.target.value)} placeholder="Paste document text or upload a PDF…" aria-label="Document text" />
        <div className="composer-row">
          <input ref={props.fileInput} type="file" accept=".pdf,.txt,.md,.json,.csv" hidden onChange={(event) => props.onFile(event.target.files?.[0])} />
          <button className="attach" onClick={() => props.fileInput.current?.click()}><Upload size={16} />{props.fileName || "Upload document"}</button>
          <button className="run-button" disabled={props.busy || !props.prompt.trim()} onClick={props.submit}>{props.busy ? <Loader2 className="spin" size={18} /> : <ArrowUp size={18} />}Run comparison</button>
        </div>
      </div>
      {props.run && <ModelGrid run={props.run} />}
    </section>
  );
}

function ModelGrid({ run }: { run: Run }) {
  const conditions = ["qwen_lora", "frontier_a", "frontier_b"];
  return <div className="model-grid">{conditions.map((name) => <ModelCard key={name} name={name} condition={run.results[name]} running={run.status === "queued" || run.status === "running"} />)}</div>;
}

function ModelCard({ name, condition, running }: { name: string; condition?: Condition; running: boolean }) {
  const metrics = condition?.metrics;
  return (
    <article className="model-card">
      <div className="model-head"><span className={`model-dot ${name}`} /><div><strong>{condition?.label || labels[name]}</strong><small>{condition?.model || "Waiting for model"}</small></div><span className="status">{condition?.status === "completed" ? <Check size={14} /> : <Loader2 className={running ? "spin" : ""} size={14} />}{condition?.status || (running ? "running" : "queued")}</span></div>
      <pre>{condition ? JSON.stringify(condition.output, null, 2) : "Waiting for output…"}</pre>
      <div className="model-stats"><span><Clock3 size={14} />{metrics?.latency_ms ? `${(metrics.latency_ms / 1000).toFixed(2)}s` : "—"}</span><span>{metrics?.input_tokens ?? 0} in</span><span>{metrics?.output_tokens ?? 0} out</span><span>${Number(metrics?.cost_usd || 0).toFixed(5)}</span></div>
    </article>
  );
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
      <div className="page-title"><span><Activity size={18} /> Evidence</span><h1>Performance over time</h1><p>Saved results only. Switching tests never calls another model.</p></div>
      <div className="segmented"><button className={props.view === "live" ? "active" : ""} onClick={() => props.setView("live")}>Live comparison</button><button className={props.view === "lift" ? "active" : ""} onClick={() => props.setView("lift")}>Adapter lift</button></div>
      <div className="run-pills">
        {props.view === "live" ? props.runs.map((run) => <button className={props.selectedRun === run.run_id ? "active" : ""} onClick={() => props.setSelectedRun(run.run_id)} key={run.run_id}>{new Date(run.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}<small>{run.attachment_name || run.status}</small></button>) : props.benchmarks.map((item) => <button className={props.selectedBenchmark === item.benchmark_id ? "active" : ""} onClick={() => props.setSelectedBenchmark(item.benchmark_id)} key={item.benchmark_id}>{item.name}<small>{item.test_documents} frozen cases</small></button>)}
      </div>
      {conditions.length ? <><Summary conditions={conditions} /><div className="chart-grid"><MetricChart title="Time" suffix="s" lower conditions={conditions} field="time" /><MetricChart title="Cost" suffix="$" lower conditions={conditions} field="cost" /><MetricChart title={props.view === "live" ? "Evaluator score" : "Accuracy"} suffix="%" conditions={conditions} field="accuracy" /></div></> : <EmptyState text="Run a comparison to populate performance." />}
      {props.view === "lift" && props.benchmark && <p className="fine-print">Frozen test SHA-256: <code>{props.benchmark.test_sha256}</code>. Accuracy is labeled field F1; schema validity and exact-form rate remain available in the saved benchmark.</p>}
    </section>
  );
}

type ChartCondition = { name: string; label: string; time: number; cost: number; accuracy: number };

function Summary({ conditions }: { conditions: ChartCondition[] }) {
  const fastest = minBy(conditions, "time"); const cheapest = minBy(conditions, "cost"); const accurate = maxBy(conditions, "accuracy");
  return <div className="summary-strip"><div><small>Fastest</small><strong>{fastest.label}</strong></div><div><small>Lowest cost</small><strong>{cheapest.label}</strong></div><div><small>Most accurate</small><strong>{accurate.label}</strong></div></div>;
}

function MetricChart({ title, suffix, lower, conditions, field }: { title: string; suffix: string; lower?: boolean; conditions: ChartCondition[]; field: "time" | "cost" | "accuracy" }) {
  const max = Math.max(...conditions.map((item) => item[field]), 0.00001);
  return <article className="chart"><div className="chart-head"><h3>{title}</h3><span>{lower ? "Lower is better" : "Higher is better"}</span></div>{conditions.map((item) => <div className="bar-row" key={item.name}><div><strong>{item.label}</strong><span>{field === "cost" ? "$" : ""}{formatMetric(item[field], field)}{field === "cost" ? "" : suffix}</span></div><div className="track"><span className={item.name} style={{ width: `${Math.max(3, item[field] / max * 100)}%` }} /></div></div>)}</article>;
}

const factoryStages = [
  { key: "queued", label: "JEV Router", detail: "Adapter match and trainability", icon: Router },
  { key: "architecting", label: "Kimi K3 — Skill Architect", detail: "Skill spec, schema, evaluation plan", icon: BrainCircuit },
  { key: "integrating", label: "Devin — Skill Integration", detail: "SKILL.md, validator spec, tests, Hugging Face bundle", icon: PackageCheck },
  { key: "building_dataset", label: "Kimi K3 — Data Builder", detail: "TRAIN, DEV, frozen TEST, OOD", icon: Database },
  { key: "validating", label: "Dataset Validator", detail: "Schema, leakage, duplicates, hashes", icon: ShieldCheck },
  { key: "awaiting_compute", label: "Compute Gate", detail: "Budget and GPU readiness", icon: Clock3 },
  { key: "training", label: "RunPod A40 — QLoRA", detail: "Training, GPU telemetry, cost", icon: Cpu },
  { key: "evaluating", label: "Independent Evaluator", detail: "Base vs Skill vs LoRA", icon: BarChart3 },
  { key: "publishing", label: "Hugging Face Registry", detail: "Weights, metrics, manifest", icon: UploadCloud },
] as const;

function Registry({ adapters, jobs, selectedJob, setSelectedJob, onTest, onRetry }: {
  adapters: Adapter[]; jobs: Job[]; selectedJob?: string;
  setSelectedJob: (id: string) => void; onTest: () => void;
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
      <div className="registry-heading"><div><h2>Adapter registry</h2><p>Ready specialists restored from Hugging Face.</p></div><button onClick={onTest}><Plus size={16} />New specialist</button></div>
      <div className="registry-grid">
        {adapters.map((adapter) => <article className="adapter-card" key={adapter.adapter_id}><div className="adapter-icon"><FileText /></div><span className="ready"><span />Ready</span><h2>{String(adapter.metadata.name || adapter.skill_id)}</h2><p>{String(adapter.metadata.description || "Reusable LoRA adapter")}</p><dl><div><dt>Base model</dt><dd>{adapter.base_model.split("/").pop()}</dd></div><div><dt>Field F1</dt><dd>{(Number(adapter.metadata.benchmark_accuracy || 0) * 100).toFixed(1)}%</dd></div><div><dt>Version</dt><dd>{adapter.version}</dd></div></dl><button onClick={onTest}>Test adapter</button></article>)}
        {!adapters.length && <button className="new-card" onClick={onTest}><Plus /><span>Create a new specialist</span></button>}
      </div>
    </section>
  );
}

function FactoryTimeline({ job, onRetry }: { job: Job; onRetry: (jobId: string) => Promise<void> }) {
  const backendOrder = ["queued", "architecting", "integrating", "building_dataset", "validating", "awaiting_compute", "provisioning_gpu", "training", "evaluating", "publishing", "ready"];
  const failedFrom = [...(job.events || [])].reverse().find((event) => event.to_status === "failed")?.from_status;
  const effectiveStatus = job.status === "failed" ? failedFrom || "queued" : job.status;
  const current = Math.max(0, backendOrder.indexOf(effectiveStatus));
  const progress = job.progress;
  const overall = job.status === "ready" ? 100 : Math.max(4, current / (backendOrder.length - 1) * 100);
  return (
    <article className={`factory-panel ${job.status}`}>
      <div className="factory-head"><div><span className="factory-state">{job.status === "ready" ? <Check size={14} /> : job.status === "failed" ? <X size={14} /> : <Loader2 className="spin" size={14} />}{stageLabel(job.status)}</span><h2>{job.skill_id || "New LoRA specialist"}</h2><p>{job.workflow}</p></div><div className="factory-percent"><strong>{Math.round(overall)}%</strong><span>Round {job.dataset_round || 1}</span></div></div>
      <div className="factory-progress"><span style={{ width: `${overall}%` }} /></div>
      <div className="factory-workspace">
        <div className="pipeline">
          {factoryStages.map((stage) => {
          const stageBackendIndex = backendOrder.indexOf(stage.key);
          const completed = job.status === "ready"
            || (stage.key === "queued" && Boolean(job.route))
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
            return <div className={`pipeline-step ${completed ? "complete" : active ? "active" : failed ? "failed" : "pending"}`} key={stage.key}><div className="pipeline-icon">{completed ? <Check /> : failed ? <X /> : active ? <Loader2 className="spin" /> : <Icon />}</div><div><strong>{stage.label}</strong><span>{stage.detail}</span>{stage.key === "queued" && job.route && <small>{job.route.decision} · {Math.round(job.route.confidence * 100)}% · {job.route.reason}</small>}{stage.key === "integrating" && job.devin_session && <small>{devinLabel}{job.devin_session.pull_request_url ? " · PR ready" : ""}</small>}{stage.key === "validating" && job.validation && <small>Frozen TEST: {String(job.validation.frozen_test_sha256 || "pending").slice(0, 14)}…</small>}{stage.key === "training" && active && progress && <TrainingTelemetry progress={progress} />}{stage.key === "publishing" && job.publish_result && <a href={job.publish_result.artifact_uri.replace("hf://", "https://huggingface.co/")} target="_blank" rel="noreferrer">Open adapter ↗</a>}</div></div>;
          })}
        </div>
        <AgentOutputPanel job={job} />
      </div>
      {job.failure_message && <div className="factory-error"><div><strong>{job.failure_code}</strong><span>{job.failure_message}</span></div><button onClick={() => onRetry(job.job_id)}>Retry stage</button></div>}
      {job.deterministic_evaluation?.conditions && <FactoryResults evaluation={job.deterministic_evaluation} />}
    </article>
  );
}

function AgentOutputPanel({ job }: { job: Job }) {
  const outputs = job.agent_outputs || [];
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    setSelected(Math.max(0, outputs.length - 1));
  }, [job.job_id, outputs.length]);

  const current = outputs[selected];
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
                className={index === selected ? "active" : ""}
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
  return <div className="factory-results"><div className="results-title"><h3>Frozen TEST results</h3><span className={evaluation.passed ? "pass" : "fail"}>{evaluation.passed ? "PASS" : "REPAIR"}</span></div><div>{Object.entries(evaluation.conditions || {}).map(([name, result]) => <section key={name}><strong>{labels[name] || name}</strong><b>{((result.field_f1 || 0) * 100).toFixed(1)}% F1</b><small>{((result.schema_pass_rate || 0) * 100).toFixed(1)}% schema · {((result.exact_form_rate || 0) * 100).toFixed(1)}% exact</small></section>)}</div></div>;
}

function TabButton({ active, children, onClick }: { active: boolean; children: React.ReactNode; onClick: () => void }) { return <button className={active ? "active" : ""} onClick={onClick}>{children}</button>; }
function EmptyState({ text }: { text: string }) { return <div className="empty"><Sparkles /><p>{text}</p></div>; }
function liveConditions(run?: Run): ChartCondition[] { return run ? Object.entries(run.results).filter(([, value]) => value.status === "completed").map(([name, value]) => ({ name, label: value.label || labels[name] || name, time: (value.metrics.latency_ms || 0) / 1000, cost: Number(value.metrics.cost_usd || 0), accuracy: value.metrics.accuracy || 0 })) : []; }
function benchmarkConditions(benchmark?: Benchmark): ChartCondition[] { return benchmark ? Object.entries(benchmark.conditions).map(([name, value]) => ({ name, label: String(value.label), time: Number(value.latency_seconds || 0), cost: Number(value.cost_usd || 0), accuracy: Number(value.accuracy || 0) })) : []; }
function minBy(items: ChartCondition[], field: "time" | "cost") { return items.reduce((best, item) => item[field] < best[field] ? item : best); }
function maxBy(items: ChartCondition[], field: "accuracy") { return items.reduce((best, item) => item[field] > best[field] ? item : best); }
function formatMetric(value: number, field: string) { return field === "cost" ? value.toFixed(5) : field === "time" ? value.toFixed(2) : value.toFixed(1); }
function stageLabel(stage: string) { return stage.split("_").map((word) => word[0]?.toUpperCase() + word.slice(1)).join(" "); }
