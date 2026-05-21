from __future__ import annotations

from collections import deque
from typing import Deque, List

from mini_vllm.backends.base import Backend
from mini_vllm.runtime.request import GenerationRequest, RequestStatus


class ContinuousBatchScheduler:
    """
    Simplified continuous batching scheduler.

    This scheduler keeps an active set of requests.

    At each iteration:
        1. Admit waiting requests into the active batch.
        2. Run one batched decode step for all active requests.
        3. Remove finished/failed requests.
        4. Admit more waiting requests if space opens.
        5. Repeat.

    This is "continuous" because the active batch can change over time:

        iteration 1: [A, B]
        iteration 2: [A, B]
        iteration 3: [B, C]   # A finished, C entered
        iteration 4: [C, D]   # B finished, D entered

    This implementation uses simplified recompute batching:
        - true batched forward passes
        - no KV cache
        - full sequence recomputation each step
    """

    def __init__(
        self,
        backend: Backend,
        max_batch_size: int = 4,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        self.backend = backend
        self.max_batch_size = max_batch_size

        self.waiting: Deque[GenerationRequest] = deque()
        self.active: List[GenerationRequest] = []

        self.completed: List[GenerationRequest] = []
        self.failed: List[GenerationRequest] = []

        self.batch_steps: int = 0
        self.total_requests_admitted: int = 0
        self.batch_sizes_seen: list[int] = []

    def submit(self, request: GenerationRequest) -> str:
        self.waiting.append(request)
        return request.request_id

    def submit_many(self, requests: list[GenerationRequest]) -> list[str]:
        request_ids = []

        for request in requests:
            request_ids.append(self.submit(request))

        return request_ids

    def run_until_done(self, debug: bool = False) -> list[GenerationRequest]:
        """
        Run until both waiting and active queues are empty.
        """

        print("\nContinuousBatchScheduler starting.")
        print(f"Waiting requests: {len(self.waiting)}")
        print(f"Max batch size:   {self.max_batch_size}")

        while self.waiting or self.active:
            self._admit_waiting_requests()

            if not self.active:
                continue

            self.batch_steps += 1
            current_batch_size = len(self.active)
            self.batch_sizes_seen.append(current_batch_size)

            active_ids = ", ".join(
                request.short_id()
                for request in self.active
            )

            print(
                f"\n[BATCH STEP {self.batch_steps}] "
                f"batch_size={current_batch_size} "
                f"requests=[{active_ids}]"
            )

            self.backend.decode_batch_recompute(self.active)

            if debug:
                self._print_active_state()

            self._remove_finished_requests()

        print("\nContinuousBatchScheduler finished.")
        print(f"Completed: {len(self.completed)}")
        print(f"Failed:    {len(self.failed)}")
        print(f"Batch steps: {self.batch_steps}")

        if self.batch_sizes_seen:
            avg_batch_size = sum(self.batch_sizes_seen) / len(self.batch_sizes_seen)
            print(f"Average batch size: {avg_batch_size:.2f}")

        return self.completed

    def _admit_waiting_requests(self) -> None:
        """
        Move requests from waiting queue into active batch until capacity.
        """

        while self.waiting and len(self.active) < self.max_batch_size:
            request = self.waiting.popleft()

            print(f"[ADMIT] request={request.short_id()}")

            self.backend.prepare_for_batch_recompute(request)

            self.total_requests_admitted += 1

            if request.status == RequestStatus.FINISHED:
                self.completed.append(request)
            elif request.status == RequestStatus.FAILED:
                self.failed.append(request)
            else:
                self.active.append(request)

    def _remove_finished_requests(self) -> None:
        """
        Remove finished/failed requests from active batch.
        """

        still_active: list[GenerationRequest] = []

        for request in self.active:
            if request.status == RequestStatus.FINISHED:
                print(f"[FINISHED] request={request.short_id()}")
                self.completed.append(request)

            elif request.status == RequestStatus.FAILED:
                print(f"[FAILED] request={request.short_id()} error={request.error}")
                self.failed.append(request)

            else:
                still_active.append(request)

        self.active = still_active

    def _print_active_state(self) -> None:
        """
        Debug helper.
        """

        print("Active request state:")

        for request in self.active:
            print(
                f"  request={request.short_id()} "
                f"generated={request.num_generated_tokens}/{request.max_new_tokens} "
                f"status={request.status}"
            )

    def all_requests(self) -> list[GenerationRequest]:
        return self.completed + self.failed

    def print_summary(self) -> None:
        all_done = self.all_requests()

        if not all_done:
            print("No processed requests.")
            return

        print("\n================ Continuous Batch Summary ================")

        print(f"Batch steps: {self.batch_steps}")
        print(f"Total requests admitted: {self.total_requests_admitted}")

        if self.batch_sizes_seen:
            avg_batch_size = sum(self.batch_sizes_seen) / len(self.batch_sizes_seen)
            max_batch_size = max(self.batch_sizes_seen)
            min_batch_size = min(self.batch_sizes_seen)

            print(f"Average batch size: {avg_batch_size:.2f}")
            print(f"Min batch size:     {min_batch_size}")
            print(f"Max batch size:     {max_batch_size}")

        for request in all_done:
            print(f"\nRequest: {request.short_id()}")
            print(f"Status:  {request.status}")
            print(f"Generated tokens: {request.num_generated_tokens}")

            if request.wait_time_seconds is not None:
                print(f"Wait time:  {request.wait_time_seconds:.4f}s")

            if request.runtime_seconds is not None:
                print(f"Runtime:    {request.runtime_seconds:.4f}s")

            if request.total_time_seconds is not None:
                print(f"Total time: {request.total_time_seconds:.4f}s")

            if request.status == RequestStatus.FAILED:
                print(f"Error: {request.error}")

        print("\n==========================================================")