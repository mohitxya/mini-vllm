import torch

from mini_vllm_real.model.config import TinyGPTConfig
from mini_vllm_real.model.generation import generate_full_recompute
from mini_vllm_real.model.tiny_gpt import TinyGPT


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

    model = TinyGPT(config)
    model.eval()

    input_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    max_new_tokens = 12

    generated_ids = generate_full_recompute(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=None,
    )

    print("TinyGPT generation test")
    print("\nConfig:")
    print(config)

    print("\nInitial input_ids:")
    print(input_ids)
    print("Initial shape:", tuple(input_ids.shape))

    print("\nGenerated IDs:")
    print(generated_ids)
    print("Generated shape:", tuple(generated_ids.shape))

    expected_len = input_ids.shape[1] + max_new_tokens
    actual_len = generated_ids.shape[1]

    assert actual_len == expected_len, (
        f"Expected length {expected_len}, got {actual_len}"
    )

    assert generated_ids.shape[0] == input_ids.shape[0], (
        "Batch size changed unexpectedly."
    )

    assert generated_ids.max().item() < config.vocab_size, (
        "Generated token ID outside vocabulary."
    )

    assert generated_ids.min().item() >= 0, (
        "Generated negative token ID."
    )

    print("\nSuccess: TinyGPT autoregressive generation works.")


if __name__ == "__main__":
    main()