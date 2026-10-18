import { useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowUp, ArrowUpRight, Check, ChevronDown, FileText, Layers3, Loader2, Paperclip, Plus, Router, X } from "lucide-react";
import * as Sentry from "@sentry/react";
import { api } from "./api";
import type { Adapter, Job, RouteDecision, Run } from "./types";

type Turn = { id: string; text: string; file?: string; route?: RouteDecision; reply?: string; runId?: string; jobId?: string; failed?: boolean };
type Pending = { workflow: string; route: RouteDecision; step: "input" | "contract" | "examples" | "review" | "match"; input?: string; filename?: string };

export function Conversation({ adapters, jobs, runs, onRun, onJob, renderRun, renderJob, openLibrary, selectedAdapter, clearAdapter, openedJob, clearJob }: {
  adapters: Adapter[]; jobs: Job[]; runs: Run[];
  onRun: (run: Run) => void; onJob: (job: Job) => void;
  renderRun: (run?: Run) => ReactNode; renderJob: (job: Job) => ReactNode;
  openLibrary: () => void; selectedAdapter?: Adapter; clearAdapter: () => void;
  openedJob?: Job; clearJob: () => void;
}) {
  const [draft, setDraft] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [pending, setPending] = useState<Pending>();
  const [attachment, setAttachment] = useState<{ name: string; text: string }>();
  const [busy, setBusy] = useState<string>();
  const [error, setError] = useState<string>();
  const fileInput = useRef<HTMLInputElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const latest = useRef<HTMLDivElement>(null);
  const lock = useRef(false);

  useEffect(() => {
    if (selectedAdapter) {
      setPending(undefined);
      setDraft(`Use ${String(selectedAdapter.metadata.name || selectedAdapter.skill_id)}. ${String(selectedAdapter.metadata.description || "")}`);
      input.current?.focus();
    }
  }, [selectedAdapter]);
  useEffect(() => { latest.current?.scrollIntoView?.({ behavior: "smooth", block: "nearest" }); }, [turns.length, pending?.step, openedJob?.job_id]);

  function update(id: string, patch: Partial<Turn>) { setTurns((current) => current.map((turn) => turn.id === id ? { ...turn, ...patch } : turn)); }

  async function compare(id: string, context: Pending, material: string) {
    if (!context.route.adapter_id) throw new Error("JEV did not return a valid adapter. Send the request again.");
    setBusy("Running all three models");
    update(id, { reply: "The specialist, Claude, and GPT are working on the same input." });
    const run = await api.compare(context.workflow, material, context.route.adapter_id, context.filename);
    onRun(run);
    update(id, { runId: run.run_id });
    setPending(undefined);
  }

  async function submit() {
    if (lock.current || (!draft.trim() && !attachment)) return;
    lock.current = true;
    const text = draft.trim() || "Process the attached document.";
    const file = attachment;
    const id = crypto.randomUUID();
    const workflow = pending ? `${pending.workflow}\n\nUser clarification:\n${text}${pending.step !== "input" && file ? `\nExample input:\n${file.text}` : ""}` : text;
    setTurns((current) => [...current, { id, text, file: file?.name }]);
    setDraft(""); setAttachment(undefined); setError(undefined); setBusy("JEV is finding the right path");
    clearJob(); clearAdapter();
    try {
      // Every submitted message, including clarification, goes through the same router.
      const route = await api.route(workflow);
      update(id, { route });
      if (route.decision === "unsupported") {
        update(id, { reply: route.reason || "This request needs a different workflow. Describe a stable, repeatable task to get started." });
        setPending(undefined);
      } else if (route.decision === "existing_adapter" && route.adapter_id) {
        const material = file?.text || (pending?.step === "input" ? text : pending?.input);
        const context: Pending = { workflow, route, step: "input", input: material, filename: file?.name || pending?.filename };
        if (route.requires_confirmation) {
          setPending({ ...context, step: "match" });
          update(id, { reply: "I found a possible specialist. Review the routing decision below, then confirm to compare it with Claude and GPT." });
        } else if (material) {
          await compare(id, context, material);
        } else {
          setPending(context);
          update(id, { reply: "There’s a specialist for this. Send the text you want it to process, or attach a document. If your message already contains the input, you can use it directly below." });
        }
      } else {
        const step = !pending || pending.step === "input" || pending.step === "match" ? "contract" : pending.step === "contract" ? "examples" : "review";
        setPending({ workflow, route, step });
        const reply = step === "contract"
          ? "A new specialist looks like the right path. Before I build it, what should it return? Describe the fields or format, the rules it must follow, and how you’ll know an answer is correct. You can say “use the requirements above” if you’ve already included them."
          : step === "examples"
            ? "What will a typical input look like? Share one short example and any exceptions the specialist should handle. You can also ask me to generate synthetic examples from your rules."
            : "The brief is ready. I’ll create the skill, build and validate separate datasets, train the adapter, evaluate it, and publish the result. The configured factory budget and time limits apply. Review the brief below, then start the build when you’re ready.";
        update(id, { reply });
      }
      Sentry.addBreadcrumb({ category: "dapta.conversation", message: "routed", data: { decision: route.decision, source: route.source, clarification: Boolean(pending) } });
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "The request could not finish.";
      update(id, { reply: message, failed: true });
      setError("Your message is saved above. You can resend it or adjust the request.");
    } finally { setBusy(undefined); lock.current = false; }
  }

  async function confirm(mode: "build" | "compare") {
    if (!pending || lock.current) return;
    lock.current = true; setBusy(mode === "build" ? "Starting your specialist" : "Running all three models"); setError(undefined);
    const id = crypto.randomUUID();
    setTurns((current) => [...current, { id, text: mode === "build" ? "Build this specialist." : "Use this specialist and compare." }]);
    try {
      if (mode === "compare") {
        if (pending.step === "match" && !pending.input) {
          setPending({ ...pending, step: "input", route: { ...pending.route, requires_confirmation: false } });
          update(id, { reply: "Specialist confirmed. Send the text or attach the document you want to process." });
        } else {
          await compare(id, pending, pending.input || [...turns].reverse().find((turn) => turn.route)?.text || pending.workflow);
        }
      } else {
        const job = await api.createJob(pending.workflow, pending.route);
        onJob(job); update(id, { jobId: job.job_id, reply: "Your specialist is in the factory. You can follow the build right here." });
        setPending(undefined);
      }
    } catch (reason) { update(id, { reply: reason instanceof Error ? reason.message : "Could not start the work.", failed: true }); }
    finally { setBusy(undefined); lock.current = false; }
  }

  async function upload(file?: File) {
    if (!file || lock.current) return;
    lock.current = true; setBusy("Reading attachment"); setError(undefined);
    try { const result = await api.extract(file); setAttachment({ name: result.filename, text: result.text }); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not read the document."); }
    finally { lock.current = false; setBusy(undefined); if (fileInput.current) fileInput.current.value = ""; }
  }

  function startFresh() { if (busy) return; setTurns([]); setPending(undefined); setDraft(""); setAttachment(undefined); setError(undefined); clearAdapter(); clearJob(); input.current?.focus(); }
  const hasWork = turns.length > 0 || Boolean(openedJob);
  const last = turns[turns.length - 1];
  return <section className={`conversation ${hasWork ? "has-work" : ""}`}>
    {!hasWork ? <div className="conversation-welcome"><div className="welcome-kicker"><span /> Your work, adapted.</div><h1>What do you do<br /><span>over and over?</span></h1><p>Tell Dapta once. JEV finds a specialist or helps you create one.<br className="desktop-break" /> See what changes in seconds, tokens, and dollars.</p><div className="starter-prompts">{["Extract fields from an NDA", "Route customer support tickets", "Create a new specialist"].map((text) => <button key={text} onClick={() => { setDraft(text); input.current?.focus(); }}>{text}<ArrowUpRight size={14} /></button>)}</div></div> : <div className="conversation-thread">
      <div className="thread-topline"><span>Conversation</span><button onClick={startFresh} disabled={Boolean(busy)}><Plus size={14} />New conversation</button></div>
      {turns.map((turn) => <div className="conversation-turn" key={turn.id}>
        <div className="user-message"><span>You</span><p>{turn.text}</p>{turn.file && <small><FileText size={13} />{turn.file}</small>}</div>
        <div className="assistant-message"><div className="assistant-identity"><span className="jev-avatar"><Router size={17} /></span><strong>JEV</strong><small>Dapta’s router</small></div>
          {turn.route && <RouteReceipt route={turn.route} adapters={adapters} />}
          {turn.reply ? <p className={turn.failed ? "reply-error" : ""}>{turn.reply}</p> : busy && turn.id === last?.id ? <div className="routing-live"><Loader2 size={15} className="spin" />{busy}</div> : null}
        </div>
        {turn.runId && renderRun(runs.find((run) => run.run_id === turn.runId))}
        {turn.jobId && jobs.find((job) => job.job_id === turn.jobId) && <div className="inline-build">{renderJob(jobs.find((job) => job.job_id === turn.jobId)!)}</div>}
      </div>)}
      {openedJob && <div className="inline-build"><div className="assistant-identity"><span className="jev-avatar"><Layers3 size={17} /></span><strong>Specialist build</strong></div>{renderJob(openedJob)}</div>}
      {pending?.step === "review" && <div className="build-review"><div><span className="welcome-kicker">Ready for the factory</span><h2>Your specialist brief</h2></div><details><summary>Review requirements<ChevronDown size={15} /></summary><p>{pending.workflow}</p></details><div className="build-review-footer"><span>Training starts after you confirm.</span><button className="primary-action" onClick={() => confirm("build")} disabled={Boolean(busy)}><Layers3 size={16} />Build specialist<ArrowUpRight size={15} /></button></div></div>}
      {(pending?.step === "input" || pending?.step === "match") && <button className="inline-action" onClick={() => confirm("compare")} disabled={Boolean(busy)}>{pending.step === "match" ? "Confirm specialist & compare" : "Use my message as input"}<ArrowUpRight size={15} /></button>}
      <div ref={latest} />
    </div>}
    <div className="chat-dock"><div className="chat-compose">{selectedAdapter && <div className="attachment-chip"><Layers3 size={13} />{String(selectedAdapter.metadata.name || selectedAdapter.skill_id)}<button aria-label="Remove specialist suggestion" onClick={clearAdapter}><X size={12} /></button></div>}{attachment && <div className="attachment-chip"><FileText size={13} />{attachment.name}<button aria-label="Remove attachment" onClick={() => setAttachment(undefined)}><X size={12} /></button></div>}
      <textarea ref={input} aria-label="Message Dapta" placeholder={pending ? "Reply to JEV…" : "Message Dapta…"} value={draft} onChange={(event) => { setDraft(event.target.value); event.target.style.height = "auto"; event.target.style.height = `${Math.min(event.target.scrollHeight, 200)}px`; }} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(); } }} rows={2} />
      <div className="compose-toolbar"><input ref={fileInput} type="file" accept=".pdf,.txt,.md,.json,.csv" hidden onChange={(event) => upload(event.target.files?.[0])} /><button className="attach-button" aria-label="Attach a document" disabled={Boolean(busy)} onClick={() => fileInput.current?.click()}><Paperclip size={18} /></button><span>{busy ? <><Loader2 size={12} className="spin" />{busy}</> : <><span className="compose-live-dot" />Routed by JEV</>}</span><button className="send-button" aria-label="Send message" disabled={Boolean(busy) || (!draft.trim() && !attachment)} onClick={() => void submit()}>{busy ? <Loader2 className="spin" size={18} /> : <ArrowUp size={19} />}</button></div>
    </div>{error && <p role="alert" className="conversation-error">{error}</p>}<div className="dock-bottom"><button className="library-toggle" onClick={openLibrary}><Layers3 size={14} />Prior adapters<span>{adapters.length}</span><ChevronDown size={14} /></button><span>Enter to send · Shift + Enter for a new line</span></div></div>
  </section>;
}

function RouteReceipt({ route, adapters }: { route: RouteDecision; adapters: Adapter[] }) {
  const matched = adapters.find((adapter) => adapter.adapter_id === route.adapter_id);
  const label = route.decision === "existing_adapter" ? String(matched?.metadata.name || matched?.skill_id || "Specialist found") : route.decision === "create_adapter" ? "New specialist recommended" : "Needs a different approach";
  return <div className="route-receipt"><div><Check size={13} /><strong>{label}</strong><span>{route.source === "registry_fallback" ? "Registry fallback" : "JEV decision"}</span></div><p>{route.reason}</p></div>;
}

export function AdapterDrawer({ adapters, jobs, onClose, onSelect, onOpenJob }: { adapters: Adapter[]; jobs: Job[]; onClose: () => void; onSelect: (adapter: Adapter) => void; onOpenJob: (job: Job) => void }) {
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement;
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden"; close.current?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab") {
        const elements = Array.from(document.querySelectorAll<HTMLElement>(".adapter-drawer button, .adapter-drawer a"));
        const first = elements[0]; const last = elements[elements.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener("keydown", keydown);
    return () => { document.body.style.overflow = oldOverflow; document.removeEventListener("keydown", keydown); previous?.focus(); };
  }, [onClose]);
  return <div className="drawer-overlay" onClick={onClose}><section className="adapter-drawer" role="dialog" aria-modal="true" aria-labelledby="library-title" onClick={(event) => event.stopPropagation()}><div className="drawer-handle" /><header><div><span className="welcome-kicker">Your growing collection</span><h2 id="library-title">Prior adapters</h2><p>Choose a specialist. JEV will verify the fit in your conversation.</p></div><button ref={close} className="close-drawer" onClick={onClose} aria-label="Close prior adapters"><X size={19} /></button></header><div className="drawer-content"><div className="library-grid">{adapters.map((adapter) => <button className="library-card" key={adapter.adapter_id} onClick={() => onSelect(adapter)}><span className={`library-status ${adapter.metadata.benchmark_status === "needs_repair" ? "needs-repair" : ""}`}><span />{adapter.metadata.benchmark_status === "needs_repair" ? "Evaluation needs repair" : "Ready to use"}</span><h3>{String(adapter.metadata.name || adapter.skill_id)}</h3><p>{String(adapter.metadata.description || "A reusable specialist for your workflow.")}</p><footer><span>{adapter.version}</span><span>Use in chat<ArrowUpRight size={14} /></span></footer></button>)}</div>{!adapters.length && <p className="library-empty">Your first specialist will appear here after its build is published.</p>}{jobs.length > 0 && <section className="library-builds"><h3>Build activity</h3>{jobs.map((job) => <button key={job.job_id} onClick={() => onOpenJob(job)}><span><strong>{job.skill_id || "New specialist"}</strong><small>{new Date(job.created_at).toLocaleDateString()}</small></span><span className={`build-status ${job.status}`}>{job.status.replaceAll("_", " ")}<ArrowUpRight size={14} /></span></button>)}</section>}</div></section></div>;
}
