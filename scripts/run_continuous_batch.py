from mini_vllm.backends.hf_backend import HFBackend
from mini_vllm.runtime.continuous_batch_scheduler import ContinuousBatchScheduler
from mini_vllm.runtime.request import GenerationRequest
from mini_vllm.runtime.sampling import SamplingConfig, SamplingStrategy


def clean_for_display(text: str) -> str:
    lines = text.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]
    return "\n".join(non_empty_lines)


def main():
    backend = HFBackend(model_name="distilgpt2")

    scheduler = ContinuousBatchScheduler(
        backend=backend,
        max_batch_size=2,
    )

    requests = [
        GenerationRequest(
            prompt="In simple terms, a GPU is",
            max_new_tokens=15,
            sampling_config=SamplingConfig(
                strategy=SamplingStrategy.TOP_K,
                temperature=0.8,
                top_k=40,
                seed=10,
            ),
        ),
        GenerationRequest(
            prompt="The reason KV cache speeds up inference is",
            max_new_tokens=15,
            sampling_config=SamplingConfig(
                strategy=SamplingStrategy.TOP_K,
                temperature=0.9,
                top_k=50,
                seed=20,
            ),
        ),
        GenerationRequest(
            prompt="A model serving runtime decides",
            max_new_tokens=15,
            sampling_config=SamplingConfig(
                strategy=SamplingStrategy.TEMPERATURE,
                temperature=0.7,
                seed=30,
            ),
        ),
        GenerationRequest(
            prompt="Continuous batching improves throughput by",
            max_new_tokens=15,
            sampling_config=SamplingConfig(
                strategy=SamplingStrategy.TOP_K,
                temperature=0.85,
                top_k=50,
                seed=40,
            ),
        ),
    ]

    request_ids = scheduler.submit_many(requests)

    print("\nSubmitted requests:")
    for request_id in request_ids:
        print(f"- {request_id}")

    completed = scheduler.run_until_done(debug=True)

    scheduler.print_summary()

    print("\n================ Generated Outputs ================")

    for request in completed:
        print(f"\n--- Request {request.short_id()} ---")
        print("Prompt:")
        print(request.prompt)

        print("\nSampling config:")
        print(request.sampling_config)

        print("\nGenerated text:")
        print(clean_for_display(request.generated_text or ""))

    print("\n===================================================")


if __name__ == "__main__":
    main()