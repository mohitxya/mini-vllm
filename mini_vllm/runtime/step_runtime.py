from __future__ import annotations

from mini_vllm.backends.base import Backend
from mini_vllm.runtime.request import GenerationRequest


class StepRuntime:
    """
    Runtime that uses the prefill/decode_one API.

    This is still single-request-at-a-time.

    But unlike NaiveRuntime, it does not ask the backend to generate the
    whole response in one call.

    Instead it explicitly does:

        prefill(request)
        decode_one(request)
        decode_one(request)
        decode_one(request)
        ...

    This prepares us for schedulers in the next milestone.
    """

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def run_request(self, request: GenerationRequest, debug: bool = False) -> GenerationRequest:
        """
        Run one request using step-wise generation.
        """

        print(f"\n[STEP RUNTIME] Starting request={request.short_id()}")

        self.backend.prefill(request)

        if debug and hasattr(self.backend, "print_request_state"):
            self.backend.print_request_state(request)

        while not request.is_finished():
            self.backend.decode_one(request)

            if debug and hasattr(self.backend, "print_request_state"):
                self.backend.print_request_state(request)

        print(
            f"[STEP RUNTIME] Finished request={request.short_id()} "
            f"status={request.status}"
        )

        return request