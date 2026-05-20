from mini_vllm.backends.hf_backend import HFBackend


def main():
    backend = HFBackend(model_name="sshleifer/tiny-gpt2")

    prompt = "In the future, AI inference systems will become"
    output = backend.generate_one_with_kv_cache_debug(prompt=prompt, max_new_tokens=5)

    print("\n=== FINAL OUTPUT ===")
    print(output)

    backend.compare_kv_cache_speed(
        prompt=prompt, 
        max_new_tokens=30,
    )

if __name__ == "__main__":
    main()
