import { createClientFromRequest } from "npm:@base44/sdk";
import { secrets } from "base44:runtime";

type RequestPayload = {
  path: string;
  method?: "GET" | "POST";
  body?: unknown;
  idempotencyKey?: string | null;
  file?: { name: string; type?: string; base64: string };
};

const ndaAdapter = {
  adapter_id: "nda-intake/qwen3-vl-8b-v1",
  skill_id: "nda-intake",
  version: "v1",
  base_model: "Qwen/Qwen3-VL-8B-Instruct",
  artifact_uri: "hf://adityave/dapta-lora-adapters/nda-intake/qwen3-vl-8b-v1",
  created_at: "2026-10-17T00:00:00Z",
  metadata: {
    name: "NDA Intake",
    description: "Long NDA to normalized legal intake JSON",
    status: "ready",
    benchmark_accuracy: 0.7074626865671643,
    training_minutes: 13.7073,
    training_cost_usd: "0",
  },
};

const ndaBenchmark = {
  benchmark_id: "nda-kleister-v1",
  name: "NDA Intake — Kleister frozen test",
  created_at: "2026-10-17T00:00:00Z",
  view: "adapter_lift",
  test_documents: 83,
  test_sha256: "b5a42b7d8c4d9c634bf17eff469b18f4207750c4a028b150b9517564d6ebba65",
  conditions: {
    base_no_skill: {
      label: "Qwen Base", accuracy: 57.57997218358831,
      schema_validity: 98.79518072289156, exact_form_rate: 6.024096385542169,
      latency_seconds: 1.82796287110845, input_tokens: 291462, output_tokens: 4947,
    },
    base_with_skill: {
      label: "Qwen + Skill", accuracy: 61.28133704735376,
      schema_validity: 98.79518072289156, exact_form_rate: 8.433734939759036,
      latency_seconds: 1.7702271022048388, input_tokens: 303331, output_tokens: 4738,
    },
    lora_short_prompt: {
      label: "Qwen LoRA", accuracy: 70.74626865671643,
      schema_validity: 85.54216867469879, exact_form_rate: 33.734939759036145,
      latency_seconds: 2.1004295580602177, input_tokens: 291379, output_tokens: 3282,
    },
  },
};

export default async function (request: Request): Promise<Response> {
  const base44 = createClientFromRequest(request);
  const user = await base44.auth.me();
  if (!user) return Response.json({ detail: "Authentication required" }, { status: 401 });

  const payload = (await request.json()) as RequestPayload;
  const method = payload.method || "GET";
  if (!payload.path?.startsWith("/v1/") || !["GET", "POST"].includes(method)) {
    return Response.json({ detail: "Invalid controller request" }, { status: 400 });
  }

  const controllerUrl = secrets.get("DAPTA_CONTROLLER_URL");
  const controllerToken = secrets.get("DAPTA_CONTROL_TOKEN");
  if (!controllerUrl || !controllerToken) {
    const fallback = registryFallback(payload.path, method);
    if (fallback !== undefined) return Response.json(fallback);
    return Response.json({ detail: "Dapta controller is not configured" }, { status: 503 });
  }

  const headers = new Headers({ "X-Dapta-Token": controllerToken });
  if (payload.idempotencyKey) headers.set("Idempotency-Key", payload.idempotencyKey);
  let body: BodyInit | undefined;

  if (payload.file) {
    const bytes = Uint8Array.from(atob(payload.file.base64), (character) => character.charCodeAt(0));
    const form = new FormData();
    form.append("file", new Blob([bytes], { type: payload.file.type || "application/octet-stream" }), payload.file.name);
    body = form;
  } else if (method === "POST") {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(payload.body || {});
  }

  const upstream = await fetch(`${controllerUrl.replace(/\/$/, "")}${payload.path}`, { method, headers, body });
  const text = await upstream.text();
  let result: unknown;
  try { result = JSON.parse(text); } catch { result = { detail: text }; }

  if (upstream.ok) {
    try {
      await mirrorSnapshots(base44, payload.path, result);
    } catch (error) {
      console.warn("Snapshot mirror skipped", error instanceof Error ? error.message : error);
    }
  }
  return Response.json(result, { status: upstream.status });
}

function registryFallback(path: string, method: string): unknown | undefined {
  if (method !== "GET") return undefined;
  if (path === "/v1/adapters") return [ndaAdapter];
  if (path === "/v1/benchmarks") return [ndaBenchmark];
  if (path === "/v1/runs" || path === "/v1/jobs") return [];
  return undefined;
}

async function mirrorSnapshots(base44: any, path: string, result: any): Promise<void> {
  const kind = path.includes("/runs") ? "run"
    : path.includes("/jobs") ? "job"
      : path.includes("/adapters") ? "adapter"
        : path.includes("/benchmarks") ? "benchmark" : null;
  if (!kind) return;
  const records = Array.isArray(result) ? result : [result?.job || result];
  for (const record of records) {
    if (!record || typeof record !== "object") continue;
    const externalId = String(record.run_id || record.job_id || record.adapter_id || record.benchmark_id || "");
    if (!externalId) continue;
    const snapshotRecord = kind === "job" ? compactJob(record) : record;
    const snapshot = {
      external_id: externalId,
      kind,
      status: String(record.status || "stored"),
      payload_json: JSON.stringify(snapshotRecord),
      source_updated_at: String(record.updated_at || record.created_at || new Date().toISOString()),
    };
    const existing = await base44.asServiceRole.entities.DaptaSnapshot.filter(
      { "data.external_id": externalId, "data.kind": kind }, undefined, 1,
    );
    if (existing.length) await base44.asServiceRole.entities.DaptaSnapshot.update(existing[0].id, snapshot);
    else await base44.asServiceRole.entities.DaptaSnapshot.create(snapshot);
  }
}

function compactJob(record: any): any {
  return {
    job_id: record.job_id,
    skill_id: record.skill_id,
    status: record.status,
    dataset_round: record.dataset_round,
    progress: record.progress,
    validation: record.validation,
    deterministic_evaluation: record.deterministic_evaluation,
    publish_result: record.publish_result,
    agent_outputs: Array.isArray(record.agent_outputs)
      ? record.agent_outputs.map((output: any) => ({
        stage: output.stage,
        agent: output.agent,
        summary: output.summary,
        source: output.source,
      }))
      : [],
    created_at: record.created_at,
    updated_at: record.updated_at,
  };
}
