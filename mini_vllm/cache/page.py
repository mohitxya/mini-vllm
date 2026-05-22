from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVPage:
    """
    Metadata for one simulated KV-cache page.

    In a real LLM serving engine, a KV page would store actual K/V tensors.

    In this educational simulator, we only track:
        - page_id
        - owner request
        - capacity in token slots
        - how many slots are used

    Example:
        page_size = 16
        used_slots = 10

    Meaning:
        this page can hold 16 token positions,
        and currently 10 are occupied.
    """

    page_id: int
    capacity: int
    owner_request_id: str | None = None
    used_slots: int = 0

    def is_free(self) -> bool:
        """
        A page is free if it has no owner.
        """

        return self.owner_request_id is None

    def has_space(self) -> bool:
        """
        Return True if page can store at least one more token slot.
        """

        return self.used_slots < self.capacity

    def remaining_slots(self) -> int:
        """
        Number of unused token slots in this page.
        """

        return self.capacity - self.used_slots

    def allocate_to(self, request_id: str) -> None:
        """
        Assign this page to a request.

        A free page starts with zero used slots.
        """

        if not self.is_free():
            raise RuntimeError(
                f"Page {self.page_id} is already owned by {self.owner_request_id}"
            )

        self.owner_request_id = request_id
        self.used_slots = 0

    def append_token(self) -> None:
        """
        Store one token slot in this page.
        """

        if self.is_free():
            raise RuntimeError(
                f"Cannot append token to free page {self.page_id}"
            )

        if not self.has_space():
            raise RuntimeError(
                f"Page {self.page_id} is full"
            )

        self.used_slots += 1

    def free(self) -> None:
        """
        Release page back to allocator.
        """

        self.owner_request_id = None
        self.used_slots = 0