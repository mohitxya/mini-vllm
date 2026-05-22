from __future__ import annotations

import json
import threading
import time
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mini_vllm.backends.hf_backend import HFBackend
from mini_vllm.runtime.request import GenerationRequest
from mini_vllm.runtime.sampling import SamplingConfig, SamplingStrategy
from mini_vllm.runtime.step_runtime import StepRuntime


# ---------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------


class GenerateRequestBody(BaseModel):
    """
    Request body for POST /generate.

    Example:

        {
            "prompt": "Explain KV cache simply",
            "max_new_tokens": 40,
            "strategy": "top_k",
            "temperature": 0.8,
            "top_k": 50
        }
    """

    prompt: str = Field(
        ...,
        min_length=1,
        description="Input prompt for generation.",
    )

    max_new_tokens: int = Field(
        default=50,
        ge=1,
        le=256,
        description="Maximum number of new tokens to generate.",
    )

    strategy: Literal["greedy", "temperature", "top_k"] = Field(
        default="greedy",
        description="Token sampling strategy.",
    )

    temperature: float = Field(
        default=0.8,
        gt=0.0,
        le=5.0,
        description="Sampling temperature. Used for temperature and top_k.",
    )

    top_k: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Top-k value. Used only when strategy is top_k.",
    )

    seed: int | None = Field(
        default=None,
        description="Optional random seed for reproducible sampling.",
    )


class GenerateResponseBody(BaseModel):
    """
    Response body for POST /generate.
    """

    request_id: str
    prompt: str
    generated_text: str
    max_new_tokens: int
    num_generated_tokens: int
    strategy: str
    total_time_seconds: float | None
    runtime_seconds: float | None
    device: str
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    cuda_available: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    device: str
    cuda_available: bool


# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------


app = FastAPI(
    title="mini-vLLM API",
    description="Educational LLM inference runtime API.",
    version="0.1.0",
)


# Global backend/runtime.
#
# Important:
# We load the model once when the server starts.
# We do NOT reload the model for every request.
MODEL_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

backend: HFBackend | None = None
runtime: StepRuntime | None = None

# Simple lock to avoid concurrent access issues in this educational version.
#
# Later, when we build a proper async scheduler, requests should enter a queue.
# For now, this ensures one generation runs at a time.
generation_lock = threading.Lock()


@app.on_event("startup")
def startup_event() -> None:
    """
    Load model once at server startup.

    This is important because model loading is expensive.
    """

    global backend, runtime

    print("\n[startup] Loading mini-vLLM backend...")
    print(f"[startup] Model:  {MODEL_NAME}")
    print(f"[startup] Device: {DEVICE}")

    backend = HFBackend(
        model_name=MODEL_NAME,
        device=DEVICE,
    )

    runtime = StepRuntime(backend=backend)

    print("[startup] Backend ready.\n")


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health check endpoint.

    Useful for confirming the server is running.
    """

    return HealthResponse(
        status="ok",
        model_loaded=backend is not None,
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
    )


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """
    Return model/device information.
    """

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
    )


@app.post("/generate", response_model=GenerateResponseBody)
def generate(body: GenerateRequestBody) -> GenerateResponseBody:
    """
    Generate text from a prompt.

    This endpoint uses StepRuntime:

        request
        → prefill
        → decode_one repeatedly
        → final generated text

    In this milestone, generation is blocking:
        client waits until the full response is finished.

    Streaming will come in Milestone 11.
    """

    if runtime is None or backend is None:
        raise HTTPException(
            status_code=503,
            detail="Model backend is not loaded yet.",
        )

    try:
        sampling_config = SamplingConfig(
            strategy=SamplingStrategy(body.strategy),
            temperature=body.temperature,
            top_k=body.top_k,
            seed=body.seed,
        )

        request = GenerationRequest(
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            sampling_config=sampling_config,
        )

        start = time.perf_counter()

        # For now, serialize generation.
        # Without this, multiple HTTP requests could call model forward at the same time.
        with generation_lock:
            completed_request = runtime.run_request(
                request=request,
                debug=False,
            )

        total_api_time = time.perf_counter() - start

        if completed_request.status.value == "FAILED":
            raise HTTPException(
                status_code=500,
                detail=completed_request.error or "Generation failed.",
            )

        return GenerateResponseBody(
            request_id=completed_request.request_id,
            prompt=completed_request.prompt,
            generated_text=completed_request.generated_text or "",
            max_new_tokens=completed_request.max_new_tokens,
            num_generated_tokens=completed_request.num_generated_tokens,
            strategy=body.strategy,
            total_time_seconds=completed_request.total_time_seconds or total_api_time,
            runtime_seconds=completed_request.runtime_seconds,
            device=DEVICE,
            model_name=MODEL_NAME,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
@app.post("/generate_stream")
def generate_stream(body: GenerateRequestBody):
    """
    Stream generated tokens as newline-delimited JSON.

    Response format:
        Each line is a JSON object.

    Example events:

        {"type": "start", "request_id": "..."}
        {"type": "token", "text": " The"}
        {"type": "token", "text": " GPU"}
        {"type": "done", "request_id": "...", "generated_text": "..."}
        {"type": "error", "message": "..."}

    This is not Server-Sent Events yet.
    It is NDJSON-style streaming:
        one JSON object per line.

    Use curl with -N:
        curl -N -X POST ...
    """

    if runtime is None or backend is None:
        raise HTTPException(
            status_code=503,
            detail="Model backend is not loaded yet.",
        )

    def stream_generator():
        try:
            sampling_config = SamplingConfig(
                strategy=SamplingStrategy(body.strategy),
                temperature=body.temperature,
                top_k=body.top_k,
                seed=body.seed,
            )

            request = GenerationRequest(
                prompt=body.prompt,
                max_new_tokens=body.max_new_tokens,
                sampling_config=sampling_config,
            )

            start_payload = {
                "type": "start",
                "request_id": request.request_id,
                "model_name": MODEL_NAME,
                "device": DEVICE,
            }

            yield json.dumps(start_payload) + "\n"

            start_time = time.perf_counter()

            with generation_lock:
                # -------------------------------
                # Prefill phase
                # -------------------------------
                backend.prefill(request)

                if request.status.value == "FAILED":
                    error_payload = {
                        "type": "error",
                        "request_id": request.request_id,
                        "message": request.error or "Prefill failed.",
                    }
                    yield json.dumps(error_payload) + "\n"
                    return

                # Prefill may generate the first token.
                if request.last_token_text:
                    token_payload = {
                        "type": "token",
                        "request_id": request.request_id,
                        "text": request.last_token_text,
                        "num_generated_tokens": request.num_generated_tokens,
                    }
                    yield json.dumps(token_payload) + "\n"

                # -------------------------------
                # Decode phase
                # -------------------------------
                while not request.is_finished():
                    backend.decode_one(request)

                    if request.status.value == "FAILED":
                        error_payload = {
                            "type": "error",
                            "request_id": request.request_id,
                            "message": request.error or "Decode failed.",
                        }
                        yield json.dumps(error_payload) + "\n"
                        return

                    if request.last_token_text:
                        token_payload = {
                            "type": "token",
                            "request_id": request.request_id,
                            "text": request.last_token_text,
                            "num_generated_tokens": request.num_generated_tokens,
                        }
                        yield json.dumps(token_payload) + "\n"

            total_time = time.perf_counter() - start_time

            done_payload = {
                "type": "done",
                "request_id": request.request_id,
                "generated_text": request.generated_text or "",
                "generated_continuation": request.generated_continuation,
                "num_generated_tokens": request.num_generated_tokens,
                "total_time_seconds": request.total_time_seconds or total_time,
                "runtime_seconds": request.runtime_seconds,
            }

            yield json.dumps(done_payload) + "\n"

        except Exception as exc:
            error_payload = {
                "type": "error",
                "message": str(exc),
            }
            yield json.dumps(error_payload) + "\n"

    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson",
    )