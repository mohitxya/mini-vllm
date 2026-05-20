from mini_vllm.backends.hf_backend import HFBackend
from mini_vllm.runtime.naive_runtime import NaiveRuntime
from mini_vllm.runtime.request import GenerationRequest
def clean_for_display(text: str) -> str:
    lines = text.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]
    return "\n".join(non_empty_lines)

def main():
    backend = HFBackend(model_name="distilgpt2")

    runtime = NaiveRuntime(backend=backend)

    requests = [
        GenerationRequest(
            prompt="In simple terms, a GPU is",
            max_new_tokens=25,
        ),
        GenerationRequest(
            prompt="The main reason KV cache makes LLM inference faster is",
            max_new_tokens=25,
        ),
        GenerationRequest(
            prompt="A model serving runtime is responsible for",
            max_new_tokens=25,
        ),
    ]

    request_ids = runtime.submit_many(requests)

    print("\nSubmitted requests:")
    for request_id in request_ids:
        print(f"- {request_id}")

    completed = runtime.run_until_empty()

    runtime.print_summary()

    print("\n================ Generated Outputs ================")

    for request in completed:
        print(f"\n--- Request {request.short_id()} ---")
        print("Prompt:")
        print(request.prompt)
        print("\nGenerated text:")
        print(clean_for_display(request.generated_text or ""))

    print("\n===================================================")


if __name__ == "__main__":
    main()