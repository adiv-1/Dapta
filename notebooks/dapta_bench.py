"""Four-way serving benchmark through vLLM: base vs full skill (uncached, cached) vs LoRA.

The notebook kernel runs this client; vLLM runs as a separate server process from its own
virtual environment (it pins a different torch than the Unsloth training stack). Quality,
cost, speed and scale come from the same streamed requests. Nothing here imports torch.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import csv
import io
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import dapta_vlm_engine as engine

CONDITIONS = ("base", "skill_uncached", "skill_cached", "lora")
# Same weights and settings everywhere; only prefix caching and the adapter differ between servers.
SERVERS = {
    "plain": {"conditions": ("base", "skill_uncached"), "prefix_cache": False, "lora": False},
    "cached": {"conditions": ("skill_cached",), "prefix_cache": True, "lora": False},
    "lora": {"conditions": ("lora",), "prefix_cache": False, "lora": True},
}
SKILL_CONDITIONS = ("skill_uncached", "skill_cached")


@dataclass(frozen=True)
class BenchConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    dtype: str = "bfloat16"
    adapter_dir: str = "/content/dapta_lora/full/adapter"
    lora_mode: str = "adapter"  # "adapter": vLLM --enable-lora; "merged": adapter merged into BF16 weights
    merged_dir: str = "/content/dapta_lora/full/merged_bf16"
    vllm_python: str = "/content/vllm-env/bin/python"
    host: str = "127.0.0.1"
    port: int = 8000
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85
    max_new_tokens: int = 1536
    concurrencies: tuple[int, ...] = (1, 8, 32)
    repeats: int = 3
    warmup_requests: int = 3
    quick_docs: int = 5
    bootstrap_samples: int = 2000
    seed: int = 42
    gpu_hour_usd: float = 1.50
    gpu_hour_note: str = "Assumed on-demand price of one A100 40GB GPU-hour; edit to your actual rate."
    results_dir: str = "/content/dapta_lora/results"
    server_start_timeout_s: int = 1800
    identical_output_sample: int = 10
    server_command_override: tuple[str, ...] | None = None  # tests only: replaces the vLLM command

    def validate(self) -> "BenchConfig":
        _require(self.lora_mode in ("adapter", "merged"), "lora_mode must be 'adapter' or 'merged'")
        _require(self.repeats >= 3, "timing needs at least 3 repeats")
        _require(all(c >= 1 for c in self.concurrencies), "concurrency must be >= 1")
        _require(1 in self.concurrencies, "concurrency 1 is required for latency and quality")
        _require(self.gpu_hour_usd > 0, "gpu_hour_usd must be positive")
        _require(0.3 <= self.gpu_memory_utilization <= 0.95, "gpu_memory_utilization must be 0.3..0.95")
        return self


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# --------------------------------------------------------------------------- documents and prompts


def encode_docs(split: Any, indices: Sequence[int], cfg: engine.EngineConfig) -> list[dict[str, Any]]:
    """Each receipt resized exactly as in training and evaluation, sent as a lossless PNG."""
    docs = []
    for index in indices:
        row = split[int(index)]
        image = engine.prepare_image(row["image"], cfg.image_long_side)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        docs.append({"doc_id": int(index), "gold": engine.gold_parse(row),
                     "data_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()})
    return docs


def messages_for(condition: str, doc: Mapping[str, Any], full_skill: str) -> list[dict[str, Any]]:
    image = {"type": "image_url", "image_url": {"url": doc["data_url"]}}
    if condition in ("base", "lora"):
        return [{"role": "user", "content": [image, {"type": "text", "text": engine.PROMPT}]}]
    # The skill is the system message, so it is an identical prefix before every image.
    return [{"role": "system", "content": full_skill},
            {"role": "user", "content": [image, {"type": "text", "text": engine.SKILL_USER_PROMPT}]}]


def served_model(condition: str) -> str:
    return "lora" if condition == "lora" else "base"


# --------------------------------------------------------------------------- server lifecycle


def adapter_rank(adapter_dir: str) -> int:
    config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    return int(config["r"])


def server_command(cfg: BenchConfig, server: str) -> list[str]:
    spec = SERVERS[server]
    if cfg.server_command_override:
        return [*cfg.server_command_override, "--port", str(cfg.port),
                "--prefix-cache", "on" if spec["prefix_cache"] else "off"]
    merged = spec["lora"] and cfg.lora_mode == "merged"
    command = [
        cfg.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.merged_dir if merged else cfg.model_id,
        "--served-model-name", "lora" if merged else "base",
        "--dtype", cfg.dtype,
        "--max-model-len", str(cfg.max_model_len),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--seed", str(cfg.seed),
        "--generation-config", "vllm",  # ignore the model's sampling defaults; requests set greedy decoding
        "--limit-mm-per-prompt", json.dumps({"image": 1, "video": 0}),
        "--mm-processor-cache-gb", "0",  # no cross-request image cache: repeats must not get cheaper
        "--host", cfg.host, "--port", str(cfg.port),
        "--enable-prefix-caching" if spec["prefix_cache"] else "--no-enable-prefix-caching",
    ]
    if spec["lora"] and not merged:
        command += ["--enable-lora", "--lora-modules", f"lora={cfg.adapter_dir}",
                    "--max-lora-rank", str(adapter_rank(cfg.adapter_dir)), "--max-loras", "1"]
    return command


class Server:
    """A vLLM server process with its log on disk; always stopped, even when a run fails."""

    def __init__(self, cfg: BenchConfig, server: str, log_dir: Path):
        self.cfg, self.server = cfg, server
        self.log_path = log_dir / f"server_{server}.log"
        self.base_url = f"http://{cfg.host}:{cfg.port}"
        self.process: subprocess.Popen | None = None
        self.log = None

    def __enter__(self) -> "Server":
        import httpx

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = server_command(self.cfg, self.server)
        env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}  # enables /reset_prefix_cache
        self.log = self.log_path.open("w", encoding="utf-8")
        self.log.write("$ " + " ".join(command) + "\n")
        self.log.flush()
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + self.cfg.server_start_timeout_s
        started = time.monotonic()
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.server} server exited ({self.process.returncode}):\n{self.log_tail()}")
            try:
                if httpx.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    self.start_seconds = round(time.monotonic() - started, 1)
                    print(f"[server {self.server}] ready in {self.start_seconds}s", flush=True)
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(2)
        self.__exit__(None, None, None)
        raise TimeoutError(f"{self.server} server not ready after {self.cfg.server_start_timeout_s}s:\n{self.log_tail()}")

    def __exit__(self, *exc: Any) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        if self.log:
            self.log.close()
        # The next server needs the memory back; vLLM workers can release it a little after exit.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            used = gpu_memory_used_mib()
            if used is None or used < 2500:
                return
            time.sleep(2)
        raise RuntimeError(f"GPU memory still in use after stopping the {self.server} server: {gpu_memory_used_mib()} MiB")

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return "(no log)"


def gpu_memory_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None  # no nvidia-smi (tests): reported as unavailable


# --------------------------------------------------------------------------- client


async def stream_request(client: Any, base_url: str, condition: str, messages: list[dict[str, Any]],
                         cfg: BenchConfig) -> dict[str, Any]:
    """One streamed chat completion: TTFT from the first content token, usage from the final chunk."""
    payload = {
        "model": served_model(condition), "messages": messages, "stream": True,
        "stream_options": {"include_usage": True}, "temperature": 0.0, "max_tokens": cfg.max_new_tokens,
        "seed": cfg.seed, "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    first = None
    parts: list[str] = []
    usage = None
    async with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload, timeout=None) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")[:500]
            raise RuntimeError(f"{condition}: HTTP {response.status_code}: {body}")
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    parts.append(piece)
    ended = time.perf_counter()
    _require(usage is not None, f"{condition}: server returned no usage; token counts unavailable")
    completion = int(usage["completion_tokens"])
    ttft = (first or ended) - started
    latency = ended - started
    return {
        "text": "".join(parts), "ttft_s": ttft, "latency_s": latency,
        "prompt_tokens": int(usage["prompt_tokens"]), "completion_tokens": completion,
        "decode_tokens_per_s": (completion - 1) / (latency - ttft) if completion > 1 and latency > ttft else None,
    }


async def _metrics(client: Any, base_url: str) -> dict[str, float]:
    text = (await client.get(f"{base_url}/metrics", timeout=10)).text
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"^(vllm:[a-z_]+)(?:\{[^}]*\})?\s+([-0-9.eE+]+)$", line.strip())
        if match:
            values[match.group(1)] = values.get(match.group(1), 0.0) + float(match.group(2))
    return values


def _kv_usage(values: Mapping[str, float]) -> float | None:
    for key in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
        if key in values:
            return values[key]
    return None


def _prefix_counters(values: Mapping[str, float]) -> tuple[float, float]:
    queries = next((v for k, v in values.items() if k.startswith("vllm:prefix_cache_queries")), 0.0)
    hits = next((v for k, v in values.items() if k.startswith("vllm:prefix_cache_hits")), 0.0)
    return queries, hits


async def timed_run(cfg: BenchConfig, base_url: str, condition: str, docs: Sequence[Mapping[str, Any]],
                    full_skill: str, concurrency: int) -> dict[str, Any]:
    """All docs at a fixed concurrency, with KV-cache usage sampled while requests are in flight."""
    import httpx

    limits = httpx.Limits(max_connections=max(64, concurrency * 2), max_keepalive_connections=64)
    async with httpx.AsyncClient(limits=limits) as client:
        before = await _metrics(client, base_url)
        gate = asyncio.Semaphore(concurrency)
        kv_peak: list[float] = []
        done = asyncio.Event()

        async def sample() -> None:
            while not done.is_set():
                usage = _kv_usage(await _metrics(client, base_url))
                if usage is not None:
                    kv_peak.append(usage)
                await asyncio.sleep(0.5)

        async def one(doc: Mapping[str, Any]) -> dict[str, Any]:
            async with gate:
                result = await stream_request(client, base_url, condition, messages_for(condition, doc, full_skill), cfg)
                return {"doc_id": doc["doc_id"], **result}

        sampler = asyncio.create_task(sample())
        started = time.perf_counter()
        try:
            results = await asyncio.gather(*(one(doc) for doc in docs))
        finally:
            done.set()
            await sampler
        wall = time.perf_counter() - started
        after = await _metrics(client, base_url)
    queries = _prefix_counters(after)[0] - _prefix_counters(before)[0]
    hits = _prefix_counters(after)[1] - _prefix_counters(before)[1]
    return {"results": results, "wall_s": wall, "kv_cache_usage_peak": max(kv_peak) if kv_peak else None,
            "prefix_cache_hit_rate": hits / queries if queries else None,
            "gpu_memory_used_mib": gpu_memory_used_mib()}


async def warm(cfg: BenchConfig, base_url: str, condition: str, warm_docs: Sequence[Mapping[str, Any]],
               full_skill: str, reset_prefix_cache: bool) -> None:
    """Warmup on validation receipts. The cached server is reset first, so only the skill prefix is warm."""
    import httpx

    async with httpx.AsyncClient() as client:
        if reset_prefix_cache:
            response = await client.post(f"{base_url}/reset_prefix_cache", timeout=60)
            _require(response.status_code == 200, f"prefix cache reset failed: HTTP {response.status_code}")
        for doc in list(warm_docs)[: cfg.warmup_requests]:
            await stream_request(client, base_url, condition, messages_for(condition, doc, full_skill), cfg)


# --------------------------------------------------------------------------- scoring and statistics


def score_output(condition: str, text: str, gold: Mapping[str, Any], validator: Any) -> dict[str, Any]:
    parsed = engine.parse_json_object(text)
    gold_receipt = engine.cord_to_expense(gold)
    if condition in ("base", "lora"):
        cord = engine.score_example(parsed, gold)
        expense = engine.score_expense({"receipts": [engine.cord_to_expense(parsed)]} if parsed is not None else None,
                                       gold_receipt, validator)
        schema_valid = parsed is not None  # CORD has no formal schema: a JSON object is schema-valid
    else:
        cord = None
        expense = engine.score_expense(parsed, gold_receipt, validator)
        schema_valid = expense["schema_valid"]
    return {"valid_json": parsed is not None, "schema_valid": schema_valid, "cord": cord, "expense": expense,
            "exact_match": expense_exact(expense)}


def expense_exact(score: Mapping[str, Any]) -> bool:
    """Whole receipt right: every gold item matched, nothing extra, every scalar right or rightly absent."""
    scalars_ok = all(v["correct"] if v["gold"] else not v["pred"] for v in score["scalars"].values())
    return bool(score["schema_valid"] and score["items_tp"] == score["items_gold"] == score["items_pred"] and scalars_ok)


def _prf(tp: float, pred: float, gold: float) -> tuple[float, float, float]:
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def quality(scored: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    expense = [s["expense"] for s in scored]
    tp = sum(e["items_tp"] for e in expense) + sum(v["correct"] for e in expense for v in e["scalars"].values())
    pred = sum(e["items_pred"] for e in expense) + sum(v["pred"] for e in expense for v in e["scalars"].values())
    gold = sum(e["items_gold"] for e in expense) + sum(v["gold"] for e in expense for v in e["scalars"].values())
    p, r, f1 = _prf(tp, pred, gold)
    totals = [e["scalars"]["total"] for e in expense if e["scalars"]["total"]["gold"]]
    metrics: dict[str, float | None] = {
        "main_f1": f1, "main_precision": p, "main_recall": r,
        "total_accuracy": sum(t["correct"] for t in totals) / len(totals) if totals else None,
        "exact_match": sum(s["exact_match"] for s in scored) / len(scored),
        "schema_valid": sum(s["schema_valid"] for s in scored) / len(scored),
        "valid_json": sum(s["valid_json"] for s in scored) / len(scored),
    }
    cord = [s["cord"] for s in scored if s["cord"] is not None]
    if len(cord) == len(scored):
        cp, cr, cf = _prf(sum(c["tp"] for c in cord), sum(c["pred_fields"] for c in cord), sum(c["gold_fields"] for c in cord))
        present = [c for c in cord if c["total_present"]]
        metrics.update({"cord_f1": cf, "cord_precision": cp, "cord_recall": cr,
                        "cord_total_exact": sum(c["total_exact"] for c in present) / len(present) if present else None,
                        "cord_document_exact": sum(c["document_exact"] for c in cord) / len(cord)})
    return metrics


def bootstrap(scored: Sequence[Mapping[str, Any]], samples: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(samples):
        resampled = [scored[rng.randrange(len(scored))] for _ in scored]
        for key, value in quality(resampled).items():
            if value is not None:
                draws.setdefault(key, []).append(value)
    return {key: _interval(values) for key, values in draws.items()}


def paired_bootstrap(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], samples: int,
                     seed: int) -> dict[str, dict[str, float]]:
    _require(len(a) == len(b), "paired bootstrap needs the same documents in the same order")
    rng = random.Random(seed)
    point_a, point_b = quality(a), quality(b)
    draws: dict[str, list[float]] = {}
    for _ in range(samples):
        picks = [rng.randrange(len(a)) for _ in a]
        qa, qb = quality([a[i] for i in picks]), quality([b[i] for i in picks])
        for key in qa:
            if qa[key] is not None and qb.get(key) is not None:
                draws.setdefault(key, []).append(qa[key] - qb[key])
    return {key: {"delta": point_a[key] - point_b[key], "ci95": _interval(values)}
            for key, values in draws.items() if point_a.get(key) is not None and point_b.get(key) is not None}


def _interval(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return ordered[int(0.025 * (len(ordered) - 1))], ordered[int(math.ceil(0.975 * (len(ordered) - 1)))]


def _pct(values: Sequence[float], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    position = (len(clean) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return clean[low] + (clean[high] - clean[low]) * (position - low)


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    clean = [v for v in values if v is not None]
    return {"mean": statistics.mean(clean) if clean else None, "p50": _pct(clean, 0.5), "p95": _pct(clean, 0.95)}


# --------------------------------------------------------------------------- orchestration


def run(cfg: BenchConfig, docs: Sequence[Mapping[str, Any]], warm_docs: Sequence[Mapping[str, Any]],
        full_skill: str, output_schema: Mapping[str, Any], *, quick: bool, provenance: Mapping[str, Any],
        training: Mapping[str, Any]) -> dict[str, Any]:
    """Synchronous entry point; safe inside a notebook whose event loop is already running."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run(cfg, docs, warm_docs, full_skill, output_schema, quick=quick,
                                             provenance=provenance, training=training)).result()


async def _run(cfg: BenchConfig, docs: Sequence[Mapping[str, Any]], warm_docs: Sequence[Mapping[str, Any]],
               full_skill: str, output_schema: Mapping[str, Any], *, quick: bool, provenance: Mapping[str, Any],
               training: Mapping[str, Any]) -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    cfg.validate()
    docs = list(docs)[: cfg.quick_docs] if quick else list(docs)
    repeats = 1 if quick else cfg.repeats
    out = Path(cfg.results_dir) / ("quick" if quick else "full")
    out.mkdir(parents=True, exist_ok=True)
    validator = Draft202012Validator(output_schema)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    servers_meta: dict[str, Any] = {}
    clock = time.perf_counter()

    for server, spec in SERVERS.items():
        with Server(cfg, server, out) as live:
            servers_meta[server] = {"start_seconds": live.start_seconds, "command": " ".join(server_command(cfg, server))}
            for condition in spec["conditions"]:
                for repeat in range(1, repeats + 1):
                    for concurrency in cfg.concurrencies:
                        await warm(cfg, live.base_url, condition, warm_docs, full_skill,
                                   reset_prefix_cache=spec["prefix_cache"])
                        result = await timed_run(cfg, live.base_url, condition, docs, full_skill, concurrency)
                        results = result["results"]
                        runs.append({"condition": condition, "server": server, "repeat": repeat,
                                     "concurrency": concurrency, "documents": len(results), "wall_s": result["wall_s"],
                                     "docs_per_s": len(results) / result["wall_s"],
                                     "output_tokens_per_s": sum(r["completion_tokens"] for r in results) / result["wall_s"],
                                     "kv_cache_usage_peak": result["kv_cache_usage_peak"],
                                     "prefix_cache_hit_rate": result["prefix_cache_hit_rate"],
                                     "gpu_memory_used_mib": result["gpu_memory_used_mib"]})
                        gold = {d["doc_id"]: d["gold"] for d in docs}
                        for r in results:
                            scored = score_output(condition, r["text"], gold[r["doc_id"]], validator)
                            rows.append({"condition": condition, "server": server, "repeat": repeat,
                                         "concurrency": concurrency, **r, **scored})
                        last = runs[-1]
                        print(f"[{condition}] repeat {repeat}/{repeats} | concurrency {concurrency} | "
                              f"{len(results)} docs in {result['wall_s']:.1f}s | {last['docs_per_s']:.2f} docs/s | "
                              f"TTFT p50 {_pct([r['ttft_s'] for r in results], 0.5):.3f}s | "
                              f"latency p50 {_pct([r['latency_s'] for r in results], 0.5):.2f}s", flush=True)
    elapsed = time.perf_counter() - clock
    report = summarize(cfg, rows, runs, docs, quick=quick, provenance=provenance, training=training,
                       servers=servers_meta, elapsed_s=elapsed)
    write_outputs(cfg, out, rows, runs, report)
    return report


def quality_rows(rows: Sequence[Mapping[str, Any]], condition: str) -> list[Mapping[str, Any]]:
    """Quality comes from the first repeat at concurrency 1, ordered by document."""
    selected = [r for r in rows if r["condition"] == condition and r["repeat"] == 1 and r["concurrency"] == 1]
    return sorted(selected, key=lambda r: r["doc_id"])


def summarize(cfg: BenchConfig, rows: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]],
              docs: Sequence[Mapping[str, Any]], *, quick: bool, provenance: Mapping[str, Any],
              training: Mapping[str, Any], servers: Mapping[str, Any], elapsed_s: float) -> dict[str, Any]:
    # skill_cached and skill_uncached must generate the same text; quality is then scored on both but they agree.
    cached, uncached = quality_rows(rows, "skill_cached"), quality_rows(rows, "skill_uncached")
    identical = [a["text"] == b["text"] for a, b in zip(cached, uncached)]
    sample = identical[: cfg.identical_output_sample]
    _require(all(sample), f"skill_cached and skill_uncached outputs differ on the first {len(sample)} documents: "
                          f"{[c['doc_id'] for c, same in zip(cached, sample) if not same]}")

    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        scored = quality_rows(rows, condition)
        speed = [r for r in rows if r["condition"] == condition and r["concurrency"] == 1]
        per_repeat = {}
        for repeat in sorted({r["repeat"] for r in speed}):
            subset = [r for r in speed if r["repeat"] == repeat]
            per_repeat[repeat] = {"ttft_p50": _pct([r["ttft_s"] for r in subset], 0.5),
                                  "latency_p50": _pct([r["latency_s"] for r in subset], 0.5)}
        throughput = {}
        for concurrency in cfg.concurrencies:
            matching = [r for r in runs if r["condition"] == condition and r["concurrency"] == concurrency]
            throughput[concurrency] = {
                "docs_per_s": statistics.mean(r["docs_per_s"] for r in matching),
                "docs_per_s_range": (min(r["docs_per_s"] for r in matching), max(r["docs_per_s"] for r in matching)),
                "output_tokens_per_s": statistics.mean(r["output_tokens_per_s"] for r in matching),
                "kv_cache_usage_peak": max((r["kv_cache_usage_peak"] for r in matching if r["kv_cache_usage_peak"] is not None), default=None),
                "gpu_memory_used_mib": max((r["gpu_memory_used_mib"] for r in matching if r["gpu_memory_used_mib"] is not None), default=None),
                "prefix_cache_hit_rate": statistics.mean([r["prefix_cache_hit_rate"] for r in matching if r["prefix_cache_hit_rate"] is not None]) if any(r["prefix_cache_hit_rate"] is not None for r in matching) else None,
            }
        top = max(cfg.concurrencies)
        cost = {f"usd_per_1000_docs_at_{c}": 1000 / throughput[c]["docs_per_s"] / 3600 * cfg.gpu_hour_usd
                for c in cfg.concurrencies}
        conditions[condition] = {
            "quality": quality(scored),
            "quality_ci95": bootstrap(scored, cfg.bootstrap_samples, cfg.seed),
            "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in scored),
            "output_tokens_mean": statistics.mean(r["completion_tokens"] for r in scored),
            "ttft_s": _stats([r["ttft_s"] for r in speed]),
            "latency_s": _stats([r["latency_s"] for r in speed]),
            "decode_tokens_per_s": _stats([r["decode_tokens_per_s"] for r in speed]),
            "speed_by_repeat": per_repeat,
            "throughput": throughput,
            "cost": cost,
            "top_concurrency": top,
        }

    lora, cached_q = quality_rows(rows, "lora"), quality_rows(rows, "skill_cached")
    top = max(cfg.concurrencies)
    saving = conditions["skill_cached"]["cost"][f"usd_per_1000_docs_at_{top}"] / 1000 - \
        conditions["lora"]["cost"][f"usd_per_1000_docs_at_{top}"] / 1000
    training_cost = training.get("train_seconds", 0) / 3600 * cfg.gpu_hour_usd
    tuning_seconds = training.get("train_seconds_including_tuning")
    with_tuning = tuning_seconds / 3600 * cfg.gpu_hour_usd if tuning_seconds is not None else None
    return {
        "mode": "quick" if quick else "full", "documents": len(docs), "repeats": max(r["repeat"] for r in rows),
        "provenance": dict(provenance), "config": {k: v for k, v in asdict(cfg).items() if k != "server_command_override"},
        "servers": dict(servers), "elapsed_s": elapsed_s,
        "skill_outputs_identical": {"sample_checked": len(sample), "identical_in_sample": sum(sample),
                                    "identical_all": sum(identical), "documents": len(identical)},
        "conditions": conditions,
        "lora_vs_skill_cached": paired_bootstrap(lora, cached_q, cfg.bootstrap_samples, cfg.seed),
        "one_time_lora_cost": {
            "train_seconds": training.get("train_seconds"),
            "train_seconds_including_tuning": training.get("train_seconds_including_tuning"),
            "usd": training_cost, "usd_including_tuning": with_tuning,
            "saving_per_doc_vs_skill_cached_usd": saving,
            "break_even_docs": math.ceil(training_cost / saving) if saving > 0 else None,
            "break_even_docs_including_tuning": math.ceil(with_tuning / saving) if saving > 0 and with_tuning is not None else None,
        },
    }


def estimate_full_minutes(report: Mapping[str, Any], cfg: BenchConfig, documents: int) -> dict[str, float]:
    """Full-run GPU time extrapolated from a quick run's measured latency and throughput."""
    total = 0.0
    for name, c in report["conditions"].items():
        c1 = c["latency_s"]["mean"] or 0.0
        per_repeat = cfg.warmup_requests * c1 * len(cfg.concurrencies)
        for concurrency in cfg.concurrencies:
            per_repeat += documents / c["throughput"][concurrency]["docs_per_s"]
        total += per_repeat * cfg.repeats
    total += sum(s["start_seconds"] for s in report["servers"].values())
    return {"minutes": total / 60, "usd": total / 3600 * cfg.gpu_hour_usd}


# --------------------------------------------------------------------------- outputs


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(point: float | None, interval: tuple[float, float] | None) -> str:
    if point is None:
        return "n/a"
    return f"{point:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else f"{point:.3f}"


def headline(report: Mapping[str, Any]) -> list[str]:
    top = max(report["config"]["concurrencies"])
    lines = [f"| Condition | Main accuracy (field F1, 95% CI) | OOD accuracy | Prompt tokens | TTFT p50 (s) | Latency p50 (s) | Throughput @{top} (docs/s) |",
             "|---|---|---|---|---|---|---|"]
    for name, c in report["conditions"].items():
        lines.append(f"| {name} | {_ci(c['quality']['main_f1'], c['quality_ci95'].get('main_f1'))} | n/a (no OOD set) | "
                     f"{c['prompt_tokens_mean']:.0f} | {_fmt(c['ttft_s']['p50'])} | {_fmt(c['latency_s']['p50'], 2)} | "
                     f"{c['throughput'][top]['docs_per_s']:.2f} |")
    return lines


def plain_read(report: Mapping[str, Any]) -> list[str]:
    """Generated from the numbers; no judgement is written that the data does not support."""
    c = report["conditions"]
    top = max(report["config"]["concurrencies"])
    diff = report["lora_vs_skill_cached"].get("main_f1")
    lines = []
    top_f1 = max(v["quality"]["main_f1"] for v in c.values())
    leaders = [k for k, v in c.items() if abs(v["quality"]["main_f1"] - top_f1) < 1e-9]
    lines.append(f"- Highest main accuracy: **{' / '.join(leaders)}** ({top_f1:.3f})"
                 + (" (tied)." if len(leaders) > 1 else "."))
    if diff:
        lo, hi = diff["ci95"]
        verdict = ("LoRA is more accurate" if lo > 0 else "the skill is more accurate" if hi < 0
                   else "the difference is not statistically clear")
        lines.append(f"- LoRA vs cached skill, field F1: {diff['delta']:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}); {verdict}.")
    lp, sp = c["lora"]["prompt_tokens_mean"], c["skill_cached"]["prompt_tokens_mean"]
    lines.append(f"- Prompt tokens: LoRA {lp:.0f} vs skill {sp:.0f} ({(lp - sp) / sp:+.0%}).")
    for key, label in (("ttft_s", "TTFT p50"), ("latency_s", "Latency p50")):
        lines.append(f"- {label}: " + ", ".join(f"{k} {_fmt(v[key]['p50'])}s" for k, v in c.items()) + ".")
    lines.append(f"- Throughput at concurrency {top}: " + ", ".join(f"{k} {v['throughput'][top]['docs_per_s']:.2f} docs/s" for k, v in c.items()) + ".")
    lines.append(f"- Cost per 1,000 documents at concurrency {top}: " + ", ".join(
        f"{k} ${v['cost'][f'usd_per_1000_docs_at_{top}']:.3f}" for k, v in c.items()) + ".")
    be = report["one_time_lora_cost"]
    if be["break_even_docs"] is None:
        lines.append("- Break-even: none. LoRA is not cheaper per document than the cached skill at this concurrency.")
    else:
        tuning = (f"{be['break_even_docs_including_tuning']:,} including tuning (${be['usd_including_tuning']:.2f})"
                  if be["break_even_docs_including_tuning"] is not None else "tuning logs unavailable, so no figure including tuning")
        lines.append(f"- Break-even vs cached skill: {be['break_even_docs']:,} documents for the final training run "
                     f"(${be['usd']:.2f}); {tuning}.")
    return lines


def write_outputs(cfg: BenchConfig, out: Path, rows: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]],
                  report: Mapping[str, Any]) -> None:
    fields = ["condition", "server", "repeat", "concurrency", "doc_id", "prompt_tokens", "completion_tokens", "ttft_s",
              "latency_s", "decode_tokens_per_s", "valid_json", "schema_valid", "exact_match",
              "fields_correct", "fields_predicted", "fields_gold"]
    with (out / "benchmark_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            e = r["expense"]
            writer.writerow({**{k: r.get(k) for k in fields[:12]}, "exact_match": r["exact_match"],
                             "fields_correct": e["items_tp"] + sum(v["correct"] for v in e["scalars"].values()),
                             "fields_predicted": e["items_pred"] + sum(v["pred"] for v in e["scalars"].values()),
                             "fields_gold": e["items_gold"] + sum(v["gold"] for v in e["scalars"].values())})
    summary_fields = ["condition"]
    summary_rows = []
    for name, c in report["conditions"].items():
        row = {"condition": name, "prompt_tokens_mean": c["prompt_tokens_mean"], "output_tokens_mean": c["output_tokens_mean"]}
        for key, value in c["quality"].items():
            row[key] = value
            interval = c["quality_ci95"].get(key)
            row[f"{key}_ci95_low"], row[f"{key}_ci95_high"] = interval if interval else (None, None)
        for metric in ("ttft_s", "latency_s", "decode_tokens_per_s"):
            for stat, value in c[metric].items():
                row[f"{metric}_{stat}"] = value
        for concurrency, t in c["throughput"].items():
            row[f"docs_per_s_c{concurrency}"] = t["docs_per_s"]
            row[f"docs_per_s_c{concurrency}_min"], row[f"docs_per_s_c{concurrency}_max"] = t["docs_per_s_range"]
            row[f"output_tokens_per_s_c{concurrency}"] = t["output_tokens_per_s"]
            row[f"kv_cache_usage_peak_c{concurrency}"] = t["kv_cache_usage_peak"]
            row[f"gpu_memory_used_mib_c{concurrency}"] = t["gpu_memory_used_mib"]
        row.update(c["cost"])
        summary_rows.append(row)
        summary_fields += [k for k in row if k not in summary_fields]
    with (out / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    (out / "benchmark_runs_per_run.json").write_text(json.dumps(list(runs), indent=2))
    (out / "benchmark_report.json").write_text(json.dumps(report, indent=2, default=str))
    charts(report, out)
    (out / "DECISIONS.md").write_text(decisions(report), encoding="utf-8")
    (out / "benchmark_report.md").write_text(markdown(report), encoding="utf-8")


def charts(report: Mapping[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["conditions"])
    c = report["conditions"]
    figure, axis = plt.subplots(figsize=(7, 4))
    points = [c[n]["quality"]["main_f1"] for n in names]
    intervals = [c[n]["quality_ci95"].get("main_f1", (p, p)) for n, p in zip(names, points)]
    axis.bar(names, points, yerr=[[p - i[0] for p, i in zip(points, intervals)], [i[1] - p for p, i in zip(points, intervals)]], capsize=6)
    axis.set_ylabel("Field F1 (95% CI)"); axis.set_ylim(0, 1); axis.set_title("Accuracy by condition")
    figure.tight_layout(); figure.savefig(out / "accuracy_by_condition.png", dpi=150); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(names, [c[n]["prompt_tokens_mean"] for n in names])
    axis.set_ylabel("Mean prompt tokens"); axis.set_title("Prompt tokens by condition")
    figure.tight_layout(); figure.savefig(out / "prompt_tokens_by_condition.png", dpi=150); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, label in ((axes[0], "ttft_s", "TTFT (s)"), (axes[1], "latency_s", "Latency (s)")):
        p50 = [c[n][metric]["p50"] or 0 for n in names]
        p95 = [c[n][metric]["p95"] or 0 for n in names]
        positions = range(len(names))
        axis.bar([x - 0.2 for x in positions], p50, width=0.4, label="p50")
        axis.bar([x + 0.2 for x in positions], p95, width=0.4, label="p95")
        axis.set_xticks(list(positions)); axis.set_xticklabels(names); axis.set_ylabel(label); axis.legend()
    axes[0].set_title("Time to first token (concurrency 1)"); axes[1].set_title("End-to-end latency (concurrency 1)")
    figure.tight_layout(); figure.savefig(out / "ttft_and_latency.png", dpi=150); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for n in names:
        levels = sorted(c[n]["throughput"], key=int)
        axis.plot([int(x) for x in levels], [c[n]["throughput"][x]["docs_per_s"] for x in levels], marker="o", label=n)
    axis.set_xlabel("Concurrency"); axis.set_ylabel("Documents per second"); axis.set_xscale("log", base=2)
    axis.set_title("Throughput vs concurrency"); axis.legend()
    figure.tight_layout(); figure.savefig(out / "throughput_vs_concurrency.png", dpi=150); plt.close(figure)


def markdown(report: Mapping[str, Any]) -> str:
    c = report["conditions"]
    lines = [f"# Four-way benchmark: receipts (CORD) · {report['mode']} run", "", *headline(report), "",
             "## What won, and where", *plain_read(report), "", "## Quality detail (95% bootstrap CIs)", "",
             "| Condition | Field P | Field R | Total accuracy | Exact match | Schema valid | CORD F1 |", "|---|---|---|---|---|---|---|"]
    for name, x in c.items():
        q, ci = x["quality"], x["quality_ci95"]
        lines.append(f"| {name} | {_ci(q['main_precision'], ci.get('main_precision'))} | {_ci(q['main_recall'], ci.get('main_recall'))} | "
                     f"{_ci(q['total_accuracy'], ci.get('total_accuracy'))} | {_ci(q['exact_match'], ci.get('exact_match'))} | "
                     f"{_ci(q['schema_valid'], ci.get('schema_valid'))} | {_ci(q.get('cord_f1'), ci.get('cord_f1'))} |")
    lines += ["", "## Speed at concurrency 1 (all repeats)", "", "| Condition | TTFT mean / p50 / p95 | Latency mean / p50 / p95 | Decode tok/s p50 | p50 latency across repeats |", "|---|---|---|---|---|"]
    for name, x in c.items():
        spread = [v["latency_p50"] for v in x["speed_by_repeat"].values()]
        lines.append(f"| {name} | {_fmt(x['ttft_s']['mean'])} / {_fmt(x['ttft_s']['p50'])} / {_fmt(x['ttft_s']['p95'])} | "
                     f"{_fmt(x['latency_s']['mean'], 2)} / {_fmt(x['latency_s']['p50'], 2)} / {_fmt(x['latency_s']['p95'], 2)} | "
                     f"{_fmt(x['decode_tokens_per_s']['p50'], 1)} | {min(spread):.2f}–{max(spread):.2f} s |")
    levels = report["config"]["concurrencies"]
    lines += ["", "## Scale", "", "| Condition | " + " | ".join(f"docs/s @{l} (min–max)" for l in levels) + " | Peak KV cache use | GPU memory (MiB) |",
              "|---|" + "---|" * (len(levels) + 2)]
    for name, x in c.items():
        t = x["throughput"]
        kv = max((t[l]["kv_cache_usage_peak"] for l in levels if t[l]["kv_cache_usage_peak"] is not None), default=None)
        mem = max((t[l]["gpu_memory_used_mib"] for l in levels if t[l]["gpu_memory_used_mib"] is not None), default=None)
        lines.append(f"| {name} | " + " | ".join(f"{t[l]['docs_per_s']:.2f} ({t[l]['docs_per_s_range'][0]:.2f}–{t[l]['docs_per_s_range'][1]:.2f})" for l in levels)
                     + f" | {_fmt(kv)} | {_fmt(mem)} |")
    be = report["one_time_lora_cost"]
    same = report["skill_outputs_identical"]
    lines += ["", "## Cost and break-even", "",
              f"- GPU price: ${report['config']['gpu_hour_usd']}/GPU-hour ({report['config']['gpu_hour_note']})",
              f"- Final LoRA training: {_fmt(be['train_seconds'], 0)} s (${be['usd']:.2f}); including tuning: "
              + (f"{_fmt(be['train_seconds_including_tuning'], 0)} s (${be['usd_including_tuning']:.2f})."
                 if be["usd_including_tuning"] is not None else "not available (tuning logs were lost with an earlier runtime)."),
              f"- Saving per 1,000 documents vs cached skill: ${1000 * be['saving_per_doc_vs_skill_cached_usd']:.4f}.",
              "", "## Caveats", "",
              f"- skill_cached and skill_uncached produced identical text on {same['identical_in_sample']}/{same['sample_checked']} sampled "
              f"documents ({same['identical_all']}/{same['documents']} overall).",
              "- No OOD set exists for the CORD receipt skill; the OOD column is n/a.",
              "- All four conditions run BF16 through vLLM; the adapter was trained on the 4-bit (NF4) base.",
              f"- LoRA served as: {report['config']['lora_mode']}.",
              "- See DECISIONS.md for every choice made where the spec was ambiguous.",
              "", "## Provenance", "", "```json", json.dumps(report["provenance"], indent=2), "```",
              "", "Charts: accuracy_by_condition.png, prompt_tokens_by_condition.png, ttft_and_latency.png, throughput_vs_concurrency.png"]
    return "\n".join(lines) + "\n"


def decisions(report: Mapping[str, Any]) -> str:
    cfg = report["config"]
    return "\n".join([
        "# DECISIONS",
        "",
        "Choices made where the benchmark spec was ambiguous, with the simplest option taken.",
        "",
        f"1. **Precision.** All four conditions run `{cfg['model_id']}` in {cfg['dtype']} through vLLM. The adapter was trained "
        "with QLoRA on the 4-bit NF4 base; serving it on BF16 weights is standard QLoRA deployment but not bit-identical to training.",
        f"2. **LoRA serving.** `{cfg['lora_mode']}`: 'adapter' uses vLLM `--enable-lora`; 'merged' serves the adapter merged into BF16 weights.",
        "3. **Main accuracy.** The skill conditions use the repo's receipt-to-expense skill unchanged, which answers in an expense-report "
        "format. The only metric defined for all four conditions is the skill-format field F1 (line items + subtotal, tax, service "
        "charge, total, per the skill's evaluation_plan.json); CORD-format answers are converted with the same code as the gold labels. "
        "The baseline CORD scorer is also reported for base and lora.",
        "4. **Schema validity.** Skill outputs: validates against the skill's output_schema.json. CORD outputs have no formal schema, "
        "so schema-valid means the output parses as a JSON object.",
        "5. **Exact match.** Every gold line item matched with nothing extra, and every scalar correct or correctly absent.",
        "6. **Test set.** The pinned CORD test parquet; its sha256 is asserted before running and recorded under provenance. "
        "No OOD set exists for this skill (SROIE is excluded by project decision).",
        "7. **Thinking mode.** Qwen3-VL-8B-Instruct has no thinking mode; `enable_thinking=False` is sent anyway.",
        "8. **Warmup and caching fairness.** Warmup uses validation receipts, never test receipts. Before every timed run the cached "
        "server's prefix cache is reset and re-warmed, so only the shared skill prefix is cached. vLLM's multimodal processor cache is "
        "disabled on every server so repeated documents do not get cheaper.",
        "9. **Quality source.** Quality is scored on the first repeat at concurrency 1. skill_cached and skill_uncached are asserted "
        f"identical on the first {report['skill_outputs_identical']['sample_checked']} documents.",
        f"10. **Cost.** ${cfg['gpu_hour_usd']}/GPU-hour ({cfg['gpu_hour_note']}). Cost per document = GPU time per document at the "
        "stated concurrency (1 / throughput). Training cost uses the logged wall-clock seconds at the same rate.",
        "11. **Image preprocessing.** Receipts are resized to the same 1280px long side used for training and evaluation and sent as PNG; "
        "vLLM's Qwen3-VL processor applies its defaults after that.",
        f"12. **Serving stack.** vLLM {report['provenance'].get('vllm', 'n/a')} (torch {report['provenance'].get('vllm_torch', 'n/a')}) in its own "
        "virtual environment, separate from the Unsloth training stack; one server configuration per row of the SERVERS table, "
        "identical except for prefix caching and the adapter.",
        "",
    ])
