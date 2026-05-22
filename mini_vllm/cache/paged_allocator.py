from __future__ import annotations

from dataclasses import dataclass

from mini_vllm.cache.page import KVPage


@dataclass
class CacheStats:
    """
    Snapshot of allocator statistics.
    """

    total_pages: int
    free_pages: int
    used_pages: int
    page_size: int
    total_token_capacity: int
    used_token_slots: int
    wasted_token_slots: int
    utilization: float


class PagedKVCacheAllocator:
    """
    Simulated paged KV-cache allocator.

    This allocator models the page-management idea behind vLLM-style memory
    management, but it does not store actual K/V tensors.

    It tracks:

        request_id -> list[page_id]

    and pages contain token-slot usage metadata.

    Example with page_size=4:

        Request A generates 10 tokens.

        A needs:
            page 0: 4 slots used
            page 1: 4 slots used
            page 2: 2 slots used

        Internal waste:
            page 2 has 2 unused slots

    This teaches:
        - allocation
        - freeing
        - internal fragmentation
        - cache pressure
        - page utilization
    """

    def __init__(
        self,
        num_pages: int = 64,
        page_size: int = 16,
    ) -> None:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")

        if page_size <= 0:
            raise ValueError("page_size must be positive")

        self.num_pages = num_pages
        self.page_size = page_size

        self.pages: list[KVPage] = [
            KVPage(
                page_id=i,
                capacity=page_size,
            )
            for i in range(num_pages)
        ]

        # request_id -> page ids owned by that request
        self.request_pages: dict[str, list[int]] = {}

    def allocate_page(self, request_id: str) -> KVPage:
        """
        Allocate one free page to a request.

        Raises:
            MemoryError if no free pages exist.
        """

        for page in self.pages:
            if page.is_free():
                page.allocate_to(request_id)

                if request_id not in self.request_pages:
                    self.request_pages[request_id] = []

                self.request_pages[request_id].append(page.page_id)
                return page

        raise MemoryError("Paged KV cache is out of free pages")

    def append_token(self, request_id: str) -> None:
        """
        Simulate storing one token's KV cache for a request.

        If the request has no page, allocate one.

        If its last page is full, allocate a new page.

        Then append one token slot.
        """

        if request_id not in self.request_pages:
            page = self.allocate_page(request_id)
            page.append_token()
            return

        page_ids = self.request_pages[request_id]

        if not page_ids:
            page = self.allocate_page(request_id)
            page.append_token()
            return

        last_page_id = page_ids[-1]
        last_page = self.pages[last_page_id]

        if not last_page.has_space():
            last_page = self.allocate_page(request_id)

        last_page.append_token()

    def append_tokens(self, request_id: str, num_tokens: int) -> None:
        """
        Simulate storing multiple token positions.
        """

        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")

        for _ in range(num_tokens):
            self.append_token(request_id)

    def free_request(self, request_id: str) -> None:
        """
        Free all pages owned by a request.
        """

        page_ids = self.request_pages.pop(request_id, [])

        for page_id in page_ids:
            self.pages[page_id].free()

    def get_request_page_ids(self, request_id: str) -> list[int]:
        """
        Return page IDs owned by a request.
        """

        return list(self.request_pages.get(request_id, []))

    def get_request_token_slots(self, request_id: str) -> int:
        """
        Return how many token slots are used by a request.
        """

        page_ids = self.request_pages.get(request_id, [])

        return sum(
            self.pages[page_id].used_slots
            for page_id in page_ids
        )

    def get_request_wasted_slots(self, request_id: str) -> int:
        """
        Return unused slots inside pages owned by one request.

        This is internal fragmentation.
        """

        page_ids = self.request_pages.get(request_id, [])

        return sum(
            self.pages[page_id].remaining_slots()
            for page_id in page_ids
        )

    def stats(self) -> CacheStats:
        """
        Return allocator-level statistics.
        """

        free_pages = sum(1 for page in self.pages if page.is_free())
        used_pages = self.num_pages - free_pages

        used_token_slots = sum(
            page.used_slots
            for page in self.pages
            if not page.is_free()
        )

        total_token_capacity = self.num_pages * self.page_size

        # Wasted token slots are unused slots inside allocated pages.
        wasted_token_slots = sum(
            page.remaining_slots()
            for page in self.pages
            if not page.is_free()
        )

        if total_token_capacity > 0:
            utilization = used_token_slots / total_token_capacity
        else:
            utilization = 0.0

        return CacheStats(
            total_pages=self.num_pages,
            free_pages=free_pages,
            used_pages=used_pages,
            page_size=self.page_size,
            total_token_capacity=total_token_capacity,
            used_token_slots=used_token_slots,
            wasted_token_slots=wasted_token_slots,
            utilization=utilization,
        )

    def print_stats(self) -> None:
        """
        Print allocator summary.
        """

        s = self.stats()

        print("\n================ Paged KV Cache Stats ================")
        print(f"Total pages:          {s.total_pages}")
        print(f"Used pages:           {s.used_pages}")
        print(f"Free pages:           {s.free_pages}")
        print(f"Page size:            {s.page_size} token slots")
        print(f"Total token capacity: {s.total_token_capacity}")
        print(f"Used token slots:     {s.used_token_slots}")
        print(f"Wasted token slots:   {s.wasted_token_slots}")
        print(f"Utilization:          {s.utilization * 100:.2f}%")
        print("======================================================")

    def print_request_table(self) -> None:
        """
        Print per-request page ownership.
        """

        print("\n================ Request → Pages ================")

        if not self.request_pages:
            print("No active requests.")
            print("=================================================")
            return

        for request_id, page_ids in self.request_pages.items():
            used = self.get_request_token_slots(request_id)
            wasted = self.get_request_wasted_slots(request_id)

            print(
                f"request={request_id[:8]} "
                f"pages={page_ids} "
                f"used_slots={used} "
                f"wasted_slots={wasted}"
            )

        print("=================================================")