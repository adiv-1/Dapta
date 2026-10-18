// Trace context carried inside the dapta-proxy payload. The Base44 function call is
// cross-origin, so Sentry cannot attach sentry-trace/baggage headers to it; the proxy
// turns these fields back into headers for the controller.
export type ProxyTrace = { sentry_trace: string; baggage?: string };

const SENTRY_TRACE = /^[0-9a-f]{32}-[0-9a-f]{16}(-[01])?$/;
const BAGGAGE_MEMBER = /^sentry-[A-Za-z0-9_.-]+=[^,;\s]*$/;
const MAX_BAGGAGE = 8192;

export function sentryBaggage(baggage: string | undefined): string | undefined {
  if (!baggage || baggage.length > MAX_BAGGAGE) return undefined;
  const members = baggage.split(",").map((member) => member.trim()).filter((member) => BAGGAGE_MEMBER.test(member));
  return members.length ? members.join(",") : undefined;
}

export function toProxyTrace(traceData: { "sentry-trace"?: string; baggage?: string }): ProxyTrace | undefined {
  const sentryTrace = traceData["sentry-trace"];
  if (!sentryTrace || !SENTRY_TRACE.test(sentryTrace)) return undefined;
  const baggage = sentryBaggage(traceData.baggage);
  return baggage ? { sentry_trace: sentryTrace, baggage } : { sentry_trace: sentryTrace };
}

// Automatic trace headers only for a directly called Dapta API (local dev: VITE_DAPTA_API_URL=/api).
// In production the API is reached through the Base44 function, so nothing is propagated
// automatically: a pattern like /^\/api\// would also match Base44's own same-origin /api calls.
export function propagationTargets(apiUrl: string | undefined): RegExp[] {
  if (!apiUrl) return [];
  const escaped = apiUrl.replace(/\/+$/, "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return [new RegExp(`^${escaped}/`)];
}
