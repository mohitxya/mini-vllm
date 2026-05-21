from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mini_vllm.runtime.request import GenerationRequest


class Backend(ABC):
    """
    Abstract base class for inference backends.

    The backend executes the model.

    The runtime/scheduler decides:
        - which requests run
        - in what order
        - whether they run one-by-one or as a batch
    """

    @abstractmethod
    def generate_one(self, prompt: str, max_new_tokens: int = 30) -> str:
        """
        Generate text for one prompt from start to finish.
        """
        pass

    @abstractmethod
    def prefill(self, request: "GenerationRequest") -> None:
        """
        Process the full prompt once and create initial KV cache.

        Used by step-wise KV-cache schedulers.
        """
        pass

    @abstractmethod
    def decode_one(self, request: "GenerationRequest") -> None:
        """
        Decode exactly one token for one request using KV cache.

        Used by round-robin scheduler.
        """
        pass

    @abstractmethod
    def prepare_for_batch_recompute(self, request: "GenerationRequest") -> None:
        """
        Prepare a request for simplified batched decoding.

        This path does not use KV cache.
        It stores tokenized prompt IDs in the request.
        """
        pass

    @abstractmethod
    def decode_batch_recompute(self, requests: list["GenerationRequest"]) -> None:
        """
        Decode one token for a batch of requests.

        Simplified batching version:
            - pads variable-length sequences
            - runs one model forward
            - extracts next token for each request
            - appends each token independently

        This recomputes full context every step.
        """
        pass