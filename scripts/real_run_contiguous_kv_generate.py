import torch

from mini_vllm_real.model.config import TinyGPTConfig
from mini_vllm_real.model.generation_kv import generate_contiguous_kv
from mini_vllm_real.model.tiny_gpt_kv import TinyGPTKV


def main():
    torch.manual_seed(42)

    config = TinyGPTConfig(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
        dropout=0.0,
    )

    model = TinyGPTKV(config)
    model.eval()

    input_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    generated_ids = generate_contiguous_kv(
        model=model,
        input_ids=input_ids,
        max_new_tokens=12,
    )

    print("Contiguous KV generation test")
    print("\nInput IDs:")
    print(input_ids)
    print("Input shape:", tuple(input_ids.shape))

    print("\nGenerated IDs:")
    print(generated_ids)
    print("Generated shape:", tuple(generated_ids.shape))

    print("\nSuccess: contiguous KV generation runs.")


if __name__ == "__main__":
    main()