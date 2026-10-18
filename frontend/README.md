# Frontend

`index.html` is a local bench for the LoRA agent. The production three-tab demo
is still the wireframe in `docs/wireframes/dapta-demo-wireframe.html`.

Start the bench from the repository root:

```bash
PYTHONPATH=backend python -m dapta.demo
```

Open http://127.0.0.1:8765. The page runs the real orchestrator on CPU: task
understanding, a Kimi K3 skill document, Kimi K3 data, a CPU LoRA loop, and a
Devin review. Devin, training, and the adapter store are local stand-ins.
