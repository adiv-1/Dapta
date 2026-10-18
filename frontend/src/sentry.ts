import * as Sentry from "@sentry/react";

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
    replaysSessionSampleRate: 1,
    replaysOnErrorSampleRate: 1,
    sendDefaultPii: false,
    beforeSend(event) {
      if (event.request) event.request.data = undefined;
      return event;
    },
  });
}
