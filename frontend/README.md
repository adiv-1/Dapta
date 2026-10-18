# Frontend

The production UI uses a JEV-routed conversation, a Performance view, and a
Prior adapters drawer. Factory progress and model comparisons appear inline.

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

## Model output previews

All three comparison columns use `ModelOutput`, while time/tokens/cost stay in
the single comparison matrix above them. Preview supports:

- Inline Vega-Lite charts (`dashboard_spec`, `vega_lite_spec`, `chart_spec`, or
  a direct specification), including multi-chart compositions. If a chart has
  no data anywhere, returned `cleaned_rows` supplies its data without changing
  the saved answer. Explicit chart data is never overwritten.
- Paginated cleaned-data/object-array tables, CSV text, extracted field lists,
  support routing badges, and receipt/expense reports with line-item tables.
- Expanded accessible views, original JSON copy/download, spreadsheet-safe CSV
  exports, and SVG chart downloads. Currencies and missing values are preserved;
  totals are model-reported, not a fabricated cross-currency sum.

Vega compilation/rendering happens in an on-demand worker, not on the UI thread.
The renderer uses the expression interpreter, disables all resource loading,
rejects URLs/hyperlinks/image marks/browser controls, limits charts to 5,000 rows
and 1 MB, and terminates after eight seconds. SVGs display as isolated images;
Python, HTML and other generated code is shown only as text. Invalid specs leave
the other output fields and original JSON available. Static previews do not
execute dashboard apps or add receipt-image training/upload capabilities.

Run `npm test` and `npm run build`. For visual checks without paid model calls,
start Vite and open `/tests/fixtures/output-preview.html`. This local-only entry
has explicitly labelled synthetic chart/receipt/routing fixtures and is not
included in the production build.
