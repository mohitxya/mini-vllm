from __future__ import annotations

from collections import deque
from typing import Deque

from mini_vllm.runtime.request import GenerationRequest


class RequestQueue:
    """
    FIFO queue for generation requests.

    FIFO = First In, First Out.

    In this naive milestone, requests are processed in arrival order.

    Later, this can evolve into:
        - priority queue
        - fair scheduler queue
        - batching queue
        - queue with admission control
    """

    def __init__(self) -> None:
        self._queue: Deque[GenerationRequest] = deque()

    def add(self, request: GenerationRequest) -> None:
        """
        Add a request to the back of the queue.
        """

        self._queue.append(request)

    def pop_next(self) -> GenerationRequest | None:
        """
        Pop the next request from the front of the queue.

        Returns None if the queue is empty.
        """

        if self.is_empty():
            return None

        return self._queue.popleft()

    def is_empty(self) -> bool:
        """
        Return True if there are no waiting requests.
        """

        return len(self._queue) == 0

    def size(self) -> int:
        """
        Number of requests currently waiting.
        """

        return len(self._queue)

    def __len__(self) -> int:
        """
        Allows usage:

            len(queue)
        """

        return self.size()