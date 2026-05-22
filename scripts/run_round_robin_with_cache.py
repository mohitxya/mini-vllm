from mini_vllm.backends.hf_backend import HFBackend
from mini_vllm.cache.paged_allocator import PagedKVCacheAllocator
from mini_vllm.runtime.request import GenerationRequest
from mini_vllm.runtime.round_robin_scheduler import RoundRobinScheduler
from mini_vllm.runtime.sampling import SamplingConfig, SamplingStrategy


def main():
    backend = HFBackend(model_name="distilgpt2")

    allocator = PagedKVCacheAllocator(
        num_pages=16,
        page_size=4,
    )

    scheduler = RoundRobinScheduler(
        backend=backend,
        cache_allocator=allocator,
    )

    requests = [
        GenerationRequest(
            prompt="A GPU is",
            max_new_tokens=10,
            sampling_config=SamplingConfig(
                strategy=SamplingStrategy.GREEDY,
            ),
        ),
        GenerationRequest(
            prompt="KV cache helps because",
            max_new_tokens=12,
            sampling_config=SamplingConfig(
                strategy=SamplingStrategy.GREEDY,
            ),
        ),
        GenerationRequest(
            prompt="A serving runtime",
            max_new_tokens=8,
            sampling_config=SamplingConfig(
                strategy=SamplingStrategy.GREEDY,
            ),
        ),
    ]

    scheduler.submit_many(requests)

    print("\nInitial cache state:")
    allocator.print_stats()
    allocator.print_request_table()

    scheduler.run_until_done(debug=False)

    print("\nFinal cache state:")
    allocator.print_stats()
    allocator.print_request_table()


if __name__ == "__main__":
    main()