from __future__ import annotations

from collections import deque
from typing import Deque, List

from mini_vllm.backends.base import Backend
from mini_vllm.runtime.request import GenerationRequest, RequestStatus
from mini_vllm.cache.paged_allocator import PagedKVCacheAllocator

class RoundRobinScheduler:
    """
    Round-robin scheduler for generation requests.

    This scheduler demonstrates the key runtime idea:

        The backend does not generate full responses by itself.
        The scheduler controls which request gets the next decode step.

    Execution pattern:

        1. Submit requests.
        2. Prefill each request once.
        3. Put unfinished requests into an active queue.
        4. Pop one request.
        5. Decode one token.
        6. If unfinished, push it back.
        7. Repeat until active queue is empty.

    This is not batched yet.

    It is still:

        one request -> one model forward

    But requests are now interleaved fairly.
    """

    def __init__(self, backend: Backend, cache_allocator: PagedKVCacheAllocator | None = None,) -> None:
        self.backend = backend

        # Requests waiting to be prefilled.
        self.waiting: Deque[GenerationRequest] = deque()

        # Requests that have been prefilled and are actively decoding.
        self.active: Deque[GenerationRequest] = deque()

        # Final successful requests.
        self.completed: List[GenerationRequest] = []

        # Failed requests.
        self.failed: List[GenerationRequest] = []

        # Scheduler-level counters.
        self.total_decode_steps: int = 0
        self.total_prefill_steps: int = 0

        self.cache_allocator = cache_allocator
    def submit(self, request: GenerationRequest) -> str:
        """
        Add one request to the waiting queue.

        Returns:
            request_id
        """

        self.waiting.append(request)
        return request.request_id

    def submit_many(self, requests: list[GenerationRequest]) -> list[str]:
        """
        Submit many requests.
        """

        request_ids = []

        for request in requests:
            request_ids.append(self.submit(request))

        return request_ids

    def run_until_done(self, debug: bool = False) -> list[GenerationRequest]:
        """
        Run scheduler until both queues are empty.

        Main phases:
            1. Prefill waiting requests.
            2. Decode active requests in round-robin order.
        """

        print("\nRoundRobinScheduler starting.")
        print(f"Waiting requests: {len(self.waiting)}")

        self._prefill_all_waiting(debug=debug)
        self._decode_round_robin(debug=debug)

        print("\nRoundRobinScheduler finished.")
        print(f"Completed: {len(self.completed)}")
        print(f"Failed:    {len(self.failed)}")
        print(f"Prefill steps: {self.total_prefill_steps}")
        print(f"Decode steps:  {self.total_decode_steps}")

        return self.completed
    def _track_cache_token(self, request: GenerationRequest) -> None:
        """
        Simulate storing one generated token in paged KV cache.

        This does not affect actual Hugging Face KV cache.
        It only tracks metadata for learning/observability.
        """

        if self.cache_allocator is None:
            return

        before_pages = self.cache_allocator.get_request_page_ids(
            request.request_id
        )

        self.cache_allocator.append_token(request.request_id)

        after_pages = self.cache_allocator.get_request_page_ids(
            request.request_id
        )

        request.cache_page_ids = after_pages

        if len(after_pages) > len(before_pages):
            new_page = after_pages[-1]
            print(
                f"[CACHE ALLOC] request={request.short_id()} "
                f"allocated_page={new_page}"
            )
    def _prefill_all_waiting(self, debug: bool = False) -> None:
        """
        Prefill all waiting requests once.

        Prefill processes the full prompt and generates the first token.

        If request is not finished after prefill, it enters active queue.
        """

        print("\n=== Prefill phase ===")

        while self.waiting:
            request = self.waiting.popleft()

            print(f"[PREFILL] request={request.short_id()}")

            tokens_before = request.num_generated_tokens

            self.backend.prefill(request)
            self.total_prefill_steps += 1

            tokens_after = request.num_generated_tokens

            if tokens_after > tokens_before:
                self._track_cache_token(request)
            if request.num_generated_tokens > 0:
                self._track_cache_token(request)
            if debug and hasattr(self.backend, "print_request_state"):
                self.backend.print_request_state(request)

            self._route_after_step(request)

        print(f"Active requests after prefill: {len(self.active)}")

    def _decode_round_robin(self, debug: bool = False) -> None:
        """
        Decode active requests one token at a time.

        Pattern:

            pop left
            decode one token
            if unfinished: append right
            if finished: store completed/failed
        """

        print("\n=== Round-robin decode phase ===")

        round_number = 0

        while self.active:
            round_number += 1

            request = self.active.popleft()

            print(
                f"[DECODE step={round_number}] "
                f"request={request.short_id()} "
                f"generated={request.num_generated_tokens}/{request.max_new_tokens}"
            )

            tokens_before = request.num_generated_tokens

            self.backend.decode_one(request)
            self.total_decode_steps += 1

            tokens_after = request.num_generated_tokens

            if tokens_after > tokens_before:
                self._track_cache_token(request)
            if (
                not request.is_finished()
                or request.status == RequestStatus.FINISHED
            ):
                # If decode generated a token, num_generated_tokens increased.
                # For this simple simulator, track one slot per decode call
                # only if the request has at least one generated token.
                self._track_cache_token(request)
            if debug and hasattr(self.backend, "print_request_state"):
                self.backend.print_request_state(request)

            self._route_after_step(request)
            if self.cache_allocator is not None and self.total_decode_steps % 5 == 0:
                self.cache_allocator.print_stats()
                self.cache_allocator.print_request_table()

    def _route_after_step(self, request: GenerationRequest) -> None:
        """
        After prefill or decode, route request to the correct place.

        Cases:
            - FINISHED -> completed list
            - FAILED -> failed list
            - still RUNNING -> active queue
        """

        if request.status == RequestStatus.FINISHED:
            self.completed.append(request)
            print(f"[FINISHED] request={request.short_id()}")

            if self.cache_allocator is not None:
                self.cache_allocator.free_request(request.request_id)
                request.cache_page_ids = []
                print(f"[CACHE FREE] request={request.short_id()}")

            return

        if request.status == RequestStatus.FAILED:
            self.failed.append(request)
            print(f"[FAILED] request={request.short_id()} error={request.error}")

            if self.cache_allocator is not None:
                self.cache_allocator.free_request(request.request_id)
                request.cache_page_ids = []
                print(f"[CACHE FREE] request={request.short_id()}")

            return

        # If not finished or failed, it should continue decoding later.
        self.active.append(request)

    def all_requests(self) -> list[GenerationRequest]:
        """
        Return all requests known after execution.
        """

        return self.completed + self.failed

    def print_summary(self) -> None:
        """
        Print timing summary for all requests.
        """

        all_done = self.all_requests()

        if not all_done:
            print("No processed requests.")
            return

        print("\n================ Scheduler Summary ================")

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

        print("\n===================================================")