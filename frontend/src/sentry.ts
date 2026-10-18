import * as Sentry from "@sentry/react";
import { propagationTargets } from "./tracing";

export function initializeSentry() {
  const dsn = import.meta.env.VITE_SENTRY_DSN;
  if (!dsn) return;
  Sentry.init({
    dsn,
    environment: import.meta.env.MODE,
    release: import.meta.env.VITE_SENTRY_RELEASE,
    integrations: [
      Sentry.browserTracingIntegration(),
      Sentry.replayIntegration({ maskAllText: true, blockAllMedia: true }),
    ],
    tracesSampleRate: sampleRate(import.meta.env.VITE_SENTRY_TRACES_SAMPLE_RATE, 0.05),
    // Only a directly called Dapta API gets trace headers automatically (local dev); production calls
    // go through the Base44 function and carry the trace in the request payload instead.
    tracePropagationTargets: propagationTargets(import.meta.env.VITE_DAPTA_API_URL),
    replaysSessionSampleRate: sampleRate(import.meta.env.VITE_SENTRY_REPLAYS_SESSION_SAMPLE_RATE, 0.05),
    replaysOnErrorSampleRate: sampleRate(import.meta.env.VITE_SENTRY_REPLAYS_ON_ERROR_SAMPLE_RATE, 1),
    sendDefaultPii: false,
    beforeSend(event) {
      if (event.request) event.request.data = undefined;
      return event;
    },
  });
}

function sampleRate(value: string | undefined, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && value !== "" ? Math.min(1, Math.max(0, parsed)) : fallback;
}
