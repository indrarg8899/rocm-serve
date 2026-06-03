"""
rocm-serve: OpenAI-compatible LLM inference server for AMD ROCm/MI300X.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import ServerConfig
from engine import InferenceEngine
from scheduler import ContinuousBatchingScheduler

logger = logging.getLogger("rocm-serve")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: load model on startup, cleanup on shutdown."""
    config = ServerConfig.from_args()
    app.state.config = config
    app.state.engine = InferenceEngine(config)
    app.state.engine.initialize()
    app.state.scheduler = ContinuousBatchingScheduler(config, app.state.engine)
    await app.state.scheduler.start()
    logger.info("rocm-serve ready on port %d", config.port)
    yield
    await app.state.scheduler.stop()
    app.state.engine.shutdown()
    logger.info("rocm-serve shutdown complete")


app = FastAPI(
    title="rocm-serve",
    version="0.1.0",
    description="High-throughput LLM inference server for AMD ROCm/MI300X",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# OpenAI-Compatible Endpoints
# ──────────────────────────────────────────────


@app.get("/v1/models")
async def list_models():
    """List available models."""
    config = app.state.config
    return {
        "object": "list",
        "data": [
            {
                "id": config.model_name,
                "object": "model",
                "owned_by": "rocm-serve",
                "permission": [],
            }
        ],
    }


@app.post("/v1/completions")
async def completions(request: dict):
    """OpenAI-compatible completions endpoint."""
    engine = app.state.engine
    scheduler = app.state.scheduler

    prompt = request.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0]  # Simplified: single prompt handling

    max_tokens = request.get("max_tokens", 256)
    temperature = request.get("temperature", 1.0)
    top_p = request.get("top_p", 1.0)
    stream = request.get("stream", False)
    stop = request.get("stop")
    if isinstance(stop, str):
        stop = [stop]

    request_id = f"cmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    params = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }

    if stream:
        return StreamingResponse(
            _stream_completion(request_id, config.model_name, created, params),
            media_type="text/event-stream",
        )

    result = await scheduler.submit_and_wait(params)

    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": app.state.config.model_name,
        "choices": [
            {
                "text": result["text"],
                "index": 0,
                "logprobs": None,
                "finish_reason": result.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """OpenAI-compatible chat completions endpoint."""
    engine = app.state.engine
    scheduler = app.state.scheduler

    messages = request.get("messages", [])
    max_tokens = request.get("max_tokens", 256)
    temperature = request.get("temperature", 1.0)
    top_p = request.get("top_p", 1.0)
    stream = request.get("stream", False)
    stop = request.get("stop")
    if isinstance(stop, str):
        stop = [stop]

    # Apply chat template
    prompt = engine.apply_chat_template(messages)

    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    params = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }

    if stream:
        return StreamingResponse(
            _stream_chat_completion(request_id, config.model_name, created, params),
            media_type="text/event-stream",
        )

    result = await scheduler.submit_and_wait(params)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": app.state.config.model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"],
                },
                "finish_reason": result.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    engine = app.state.engine
    scheduler = app.state.scheduler
    return {
        "status": "healthy",
        "engine_ready": engine.is_ready(),
        "queue_size": scheduler.queue_size,
        "active_requests": scheduler.active_count,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    engine = app.state.engine
    scheduler = app.state.scheduler
    return {
        "prompt_tokens_total": scheduler.metrics.prompt_tokens,
        "generated_tokens_total": scheduler.metrics.generated_tokens,
        "requests_total": scheduler.metrics.requests_total,
        "requests_running": scheduler.active_count,
        "requests_waiting": scheduler.queue_size,
        "gpu_memory_used_bytes": engine.gpu_memory_used,
        "gpu_memory_total_bytes": engine.gpu_memory_total,
    }


# ──────────────────────────────────────────────
# Streaming Helpers
# ──────────────────────────────────────────────


async def _stream_completion(request_id, model, created, params):
    """Yield SSE chunks for completions streaming."""
    scheduler = app.state.scheduler
    async for chunk in scheduler.stream(params):
        data = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "text": chunk["text"],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": chunk.get("finish_reason"),
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_chat_completion(request_id, model, created, params):
    """Yield SSE chunks for chat completions streaming."""
    scheduler = app.state.scheduler
    async for chunk in scheduler.stream(params):
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk["text"]},
                    "finish_reason": chunk.get("finish_reason"),
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"


import json  # noqa: E402


def main():
    """CLI entry point."""
    config = ServerConfig.from_args()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.port,
        log_level=config.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
