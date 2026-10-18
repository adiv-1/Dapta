"""Stand-in for the vLLM OpenAI server, for testing dapta_bench without a GPU.

Implements /health, streamed /v1/chat/completions with usage, /metrics and /reset_prefix_cache.
Prefill time grows with prompt length and shrinks when the skill prefix is cached.
"""

import argparse
import asyncio
import json

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--prefix-cache", choices=("on", "off"), required=True)
args = parser.parse_args()

state = {"in_flight": 0, "queries": 0, "hits": 0, "cached_prefixes": set()}
CORD = json.dumps({"menu": [{"nm": "Kopi Susu", "cnt": "1", "price": "18,000"}], "total": {"total_price": "18,000"}})
EXPENSE = json.dumps({"receipts": [{"merchant": None, "address": None, "transaction_date": None, "currency": None,
                                    "subtotal": None, "tax": None, "service_charge": None, "total": 18000,
                                    "line_items": [{"description": "Kopi Susu", "quantity": 1, "amount": 18000}]}]})


async def health(request):
    return PlainTextResponse("ok")


async def metrics(request):
    return PlainTextResponse(
        "# HELP vllm:kv_cache_usage_perc fake\n"
        f'vllm:kv_cache_usage_perc{{engine="0"}} {min(1.0, state["in_flight"] / 40):.4f}\n'
        f'vllm:prefix_cache_queries_total{{engine="0"}} {state["queries"]}\n'
        f'vllm:prefix_cache_hits_total{{engine="0"}} {state["hits"]}\n')


async def reset(request):
    state["cached_prefixes"].clear()
    return PlainTextResponse("ok")


async def chat(request):
    body = await request.json()
    messages = body["messages"]
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    prompt_tokens = len(json.dumps(messages)) // 400 + (len(system) // 4 if system else 0) + 60
    answer = EXPENSE if system else CORD
    cached = 0
    if args.prefix_cache == "on" and system:
        state["queries"] += len(system) // 4
        if system in state["cached_prefixes"]:
            cached = len(system) // 4
            state["hits"] += cached
        state["cached_prefixes"].add(system)

    async def stream():
        state["in_flight"] += 1
        try:
            await asyncio.sleep(0.00005 * (prompt_tokens - cached))  # prefill
            pieces = [answer[i:i + 12] for i in range(0, len(answer), 12)]
            for piece in pieces:
                chunk = {"choices": [{"index": 0, "delta": {"content": piece}}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.002)
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(pieces), "total_tokens": prompt_tokens + len(pieces)}
            yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            state["in_flight"] -= 1

    if body.get("model") not in ("base", "lora"):
        return JSONResponse({"error": "unknown model"}, status_code=404)
    return StreamingResponse(stream(), media_type="text/event-stream")


app = Starlette(routes=[Route("/health", health), Route("/metrics", metrics),
                        Route("/reset_prefix_cache", reset, methods=["POST"]),
                        Route("/v1/chat/completions", chat, methods=["POST"])])
uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
