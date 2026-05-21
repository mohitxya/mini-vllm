from __future__ import annotations
from mini_vllm.runtime.sampling import SamplingConfig
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestStatus(str, Enum):
    """
    Lifecycle states for a generation request.

    WAITING:
        Request exists but generation has not started.

    RUNNING:
        Request is currently being processed.

    FINISHED:
        Request completed successfully.

    FAILED:
        Request failed due to an exception.
    """

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


@dataclass
class GenerationRequest:
    """
    Represents one user generation request.

    Milestone 4:
        This stored prompt, status, output, and timing metadata.

    Milestone 5:
        This now also stores generation state:
            - input token IDs
            - generated token IDs
            - latest token ID
            - KV cache
            - number of generated tokens

    Why?

    Because from now onward, the backend will not generate the whole response
    in one call. Instead, the runtime/scheduler will repeatedly call:

        backend.prefill(request)
        backend.decode_one(request)

    That means the request object must remember its state between steps.
    """

    prompt: str
    max_new_tokens: int = 30

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: RequestStatus = RequestStatus.WAITING

    generated_text: str | None = None
    error: str | None = None

    created_at: float = field(default_factory=time.perf_counter)
    started_at: float | None = None
    finished_at: float | None = None
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    # -------------------------------
    # Milestone 5 generation state
    # -------------------------------

    # Token IDs of original prompt.
    input_ids: Any | None = None

    # Token IDs of prompt + generated continuation.
    generated_ids: Any | None = None

    # The most recently generated token.
    # During decode, this is the only token sent to the model.
    last_token_id: Any | None = None

    # Transformer KV cache.
    # Hugging Face may store this as DynamicCache or legacy tuple format.
    past_key_values: Any | None = None

    # Number of new tokens generated after the prompt.
    num_generated_tokens: int = 0

    # Whether prefill has been run.
    has_prefilled: bool = False

    def mark_running(self) -> None:
        self.status = RequestStatus.RUNNING

        if self.started_at is None:
            self.started_at = time.perf_counter()

    def mark_finished(self, generated_text: str) -> None:
        self.status = RequestStatus.FINISHED
        self.generated_text = generated_text
        self.finished_at = time.perf_counter()

    def mark_failed(self, error: Exception | str) -> None:
        self.status = RequestStatus.FAILED
        self.error = str(error)
        self.finished_at = time.perf_counter()

    def is_finished(self) -> bool:
        return self.status in {RequestStatus.FINISHED, RequestStatus.FAILED}

    @property
    def wait_time_seconds(self) -> float | None:
        if self.started_at is None:
            return None

        return self.started_at - self.created_at

    @property
    def runtime_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None

        return self.finished_at - self.started_at

    @property
    def total_time_seconds(self) -> float | None:
        if self.finished_at is None:
            return None

        return self.finished_at - self.created_at

    def short_id(self) -> str:
        return self.request_id[:8]