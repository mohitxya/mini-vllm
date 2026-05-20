from mini_vllm.backends.hf_backend import HFBackend
from mini_vllm.runtime.request import GenerationRequest
from mini_vllm.runtime.step_runtime import StepRuntime


def main():
    backend = HFBackend(model_name="distilgpt2")

    runtime = StepRuntime(backend=backend)

    request = GenerationRequest(
        prompt="A model serving runtime is responsible for",
        max_new_tokens=20,
    )

    completed_request = runtime.run_request(
        request=request,
        debug=True,
    )

    print("\n================ Final Result ================")
    print("Request ID:", completed_request.request_id)
    print("Status:", completed_request.status)
    print("Generated tokens:", completed_request.num_generated_tokens)
    print("Runtime:", completed_request.runtime_seconds)

    print("\nGenerated text:")
    print(completed_request.generated_text)


if __name__ == "__main__":
    main()