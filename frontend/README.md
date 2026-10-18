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
Sentry Replay masks all text and blocks document media by default.
