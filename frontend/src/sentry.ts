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
    tracesSampleRate: 1,
    // Only a directly called Dapta API gets trace headers automatically (local dev); production calls
    // go through the Base44 function and carry the trace in the request payload instead.
    tracePropagationTargets: propagationTargets(import.meta.env.VITE_DAPTA_API_URL),
    replaysSessionSampleRate: 1,
    replaysOnErrorSampleRate: 1,
    sendDefaultPii: false,
    beforeSend(event) {
      if (event.request) event.request.data = undefined;
      return event;
    },
  });
}
