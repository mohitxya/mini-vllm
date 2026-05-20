from __future__ import annotations

from typing import List

from mini_vllm.backends.base import Backend
from mini_vllm.runtime.request import GenerationRequest, RequestStatus
from mini_vllm.runtime.request_queue import RequestQueue


class NaiveRuntime:
    """
    Naive multi-request runtime.

    This is intentionally simple.

    It processes requests one at a time:

        request A fully
        request B fully
        request C fully

    This is NOT efficient.

    But it gives us:
        - request lifecycle tracking
        - queue management
        - baseline behavior
        - benchmark baseline for future schedulers

    Later milestones will replace this with:
        - round-robin scheduler
        - continuous batching scheduler
        - async serving runtime
    """

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.queue = RequestQueue()
        self.completed_requests: List[GenerationRequest] = []
        self.failed_requests: List[GenerationRequest] = []

    def submit(self, request: GenerationRequest) -> str:
        """
        Submit a request to the runtime.

        Returns:
            request_id
        """

        self.queue.add(request)
        return request.request_id

    def submit_many(self, requests: list[GenerationRequest]) -> list[str]:
        """
        Submit multiple requests at once.
        """

        request_ids = []

        for request in requests:
            request_id = self.submit(request)
            request_ids.append(request_id)

        return request_ids

    def run_until_empty(self) -> list[GenerationRequest]:
        """
        Process all queued requests sequentially.

        This method blocks until the queue is empty.

        Returns:
            List of successfully completed requests.
        """

        print(f"\nNaiveRuntime starting.")
        print(f"Initial queue size: {self.queue.size()}")

        while not self.queue.is_empty():
            request = self.queue.pop_next()

            if request is None:
                break

            self._process_one_request(request)

        print("\nNaiveRuntime finished.")
        print(f"Completed: {len(self.completed_requests)}")
        print(f"Failed:    {len(self.failed_requests)}")

        return self.completed_requests

    def _process_one_request(self, request: GenerationRequest) -> None:
        """
        Process exactly one request from start to finish.
        """

        print(
            f"\n[RUNNING] request={request.short_id()} "
            f"max_new_tokens={request.max_new_tokens}"
        )

        request.mark_running()

        try:
            generated_text = self.backend.generate_one(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
            )

            request.mark_finished(generated_text)
            self.completed_requests.append(request)

            print(
                f"[FINISHED] request={request.short_id()} "
                f"runtime={request.runtime_seconds:.4f}s"
            )

        except Exception as exc:
            request.mark_failed(exc)
            self.failed_requests.append(request)

            print(
                f"[FAILED] request={request.short_id()} "
                f"error={request.error}"
            )

    def all_requests(self) -> list[GenerationRequest]:
        """
        Return all requests the runtime knows about after execution.
        """

        return self.completed_requests + self.failed_requests

    def print_summary(self) -> None:
        """
        Print a simple runtime summary.
        """

        all_done = self.all_requests()

        if not all_done:
            print("No processed requests.")
            return

        print("\n================ Runtime Summary ================")

        for request in all_done:
            print(f"\nRequest: {request.short_id()}")
            print(f"Status:  {request.status}")

            if request.wait_time_seconds is not None:
                print(f"Wait time:    {request.wait_time_seconds:.4f}s")

            if request.runtime_seconds is not None:
                print(f"Runtime:      {request.runtime_seconds:.4f}s")

            if request.total_time_seconds is not None:
                print(f"Total time:   {request.total_time_seconds:.4f}s")

            if request.status == RequestStatus.FAILED:
                print(f"Error: {request.error}")

        print("\n=================================================")