from mini_vllm.cache.paged_allocator import PagedKVCacheAllocator


def main():
    allocator = PagedKVCacheAllocator(
        num_pages=8,
        page_size=4,
    )

    print("\nInitial state")
    allocator.print_stats()
    allocator.print_request_table()

    print("\nRequest A appends 10 tokens")
    allocator.append_tokens("request_A", 10)
    allocator.print_stats()
    allocator.print_request_table()

    print("\nRequest B appends 3 tokens")
    allocator.append_tokens("request_B", 3)
    allocator.print_stats()
    allocator.print_request_table()

    print("\nRequest C appends 7 tokens")
    allocator.append_tokens("request_C", 7)
    allocator.print_stats()
    allocator.print_request_table()

    print("\nFree Request A")
    allocator.free_request("request_A")
    allocator.print_stats()
    allocator.print_request_table()

    print("\nRequest D appends 5 tokens")
    allocator.append_tokens("request_D", 5)
    allocator.print_stats()
    allocator.print_request_table()


if __name__ == "__main__":
    main()