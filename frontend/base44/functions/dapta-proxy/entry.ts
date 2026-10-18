import { createClientFromRequest } from "npm:@base44/sdk";
import { secrets } from "base44:runtime";

type RequestPayload = {
  path: string;
  method?: "GET" | "POST";
  body?: unknown;
  idempotencyKey?: string | null;
  file?: { name: string; type?: string; base64: string };
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

  if (upstream.ok) await mirrorSnapshots(base44, payload.path, result);
  return Response.json(result, { status: upstream.status });
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
    const snapshot = {
      external_id: externalId,
      kind,
      status: String(record.status || "stored"),
      payload_json: JSON.stringify(record),
      source_updated_at: String(record.updated_at || record.created_at || new Date().toISOString()),
    };
    const existing = await base44.asServiceRole.entities.DaptaSnapshot.filter(
      { "data.external_id": externalId, "data.kind": kind }, undefined, 1,
    );
    if (existing.length) await base44.asServiceRole.entities.DaptaSnapshot.update(existing[0].id, snapshot);
    else await base44.asServiceRole.entities.DaptaSnapshot.create(snapshot);
  }
}
