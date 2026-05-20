from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mini_vllm.runtime.request import GenerationRequest


class Backend(ABC):
    """
    Abstract base class for inference backends.

    The backend is responsible for model execution.

    The runtime/scheduler is responsible for deciding when requests run.

    Milestone 5 adds step-level generation methods:

        prefill(request)
        decode_one(request)

    This allows the scheduler to control generation one token at a time.
    """

    @abstractmethod
    def generate_one(self, prompt: str, max_new_tokens: int = 30) -> str:
        """
        Generate text for one prompt from start to finish.

        This is still useful for simple scripts and baseline testing.
        """
        pass

    @abstractmethod
    def prefill(self, request: "GenerationRequest") -> None:
        """
        Process the full prompt once.

        This should:
            - tokenize the prompt
            - run the model on the full prompt
            - create initial KV cache
            - generate the first token
            - update request state
        """
        pass

    @abstractmethod
    def decode_one(self, request: "GenerationRequest") -> None:
        """
        Generate exactly one more token for an already-prefilled request.

        This should:
            - pass only request.last_token_id to the model
            - reuse request.past_key_values
            - update KV cache
            - append newly generated token
            - finish request if EOS or max_new_tokens reached
        """
        pass