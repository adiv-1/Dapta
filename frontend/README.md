# Frontend

The production UI implements the approved three-tab demo: live comparison,
performance evidence, and the adapter factory registry.

```bash
npm install
npm run dev
```

Local Vite proxies `/api` to `http://localhost:8000`. The Base44 production
build automatically invokes the deployed `dapta-proxy` function. Set only
`DAPTA_CONTROLLER_URL` and `DAPTA_CONTROL_TOKEN` as Base44 secrets; do not set
`VITE_DAPTA_API_URL` in production. Keep `BASE_44_API_KEY` out of all `VITE_*`
variables.

Invite-only access is enforced by Base44 before the proxy forwards a request.
Sentry Replay masks all text and blocks document media by default. Browser tracing and routine
Replay sessions default to 5% sampling, while error Replays remain at 100%; all three rates are
configurable with the `VITE_SENTRY_*_SAMPLE_RATE` variables shown in `.env.example`.

The Skills & Adapters timeline is backed by persisted job events, not simulated
progress. It shows a universal timer/cost until the terminal job state, plus
per-stage elapsed time, cost, result, GPU telemetry, and ETA. The content area
scrolls independently and keeps the previous render visible during refreshes to
avoid a blank-screen state transition.

`EvaluationOutputTruncated` is special: **Apply repair & rerun benchmark** calls
`POST /v1/jobs/{id}/repair-evaluation`. It regenerates frozen benchmark outputs
from saved weights and never rebuilds the dataset or retrains. Model-quality
failures instead display suggested DEV-only repairs and require a user choice.
