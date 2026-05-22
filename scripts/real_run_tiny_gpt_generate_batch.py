import torch

from mini_vllm_real.model.config import TinyGPTConfig
from mini_vllm_real.model.generation import generate_full_recompute
from mini_vllm_real.model.tiny_gpt import TinyGPT


def main():
    torch.manual_seed(123)

    config = TinyGPTConfig(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
        dropout=0.0,
    )

    model = TinyGPT(config)
    model.eval()

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.long,
    )

    generated_ids = generate_full_recompute(
        model=model,
        input_ids=input_ids,
        max_new_tokens=8,
    )

    print("Batch generation test")
    print("\nInput:")
    print(input_ids)
    print("Input shape:", tuple(input_ids.shape))

    print("\nGenerated:")
    print(generated_ids)
    print("Generated shape:", tuple(generated_ids.shape))

    assert generated_ids.shape == (3, 12)

    print("\nSuccess: batch generation works.")


if __name__ == "__main__":
    main()