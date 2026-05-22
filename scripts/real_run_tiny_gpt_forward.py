import torch

from mini_vllm_real.model.config import TinyGPTConfig
from mini_vllm_real.model.tiny_gpt import TinyGPT


def main():
    config = TinyGPTConfig(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=128,
        dropout=0.0,
    )

    model = TinyGPT(config)
    model.eval()

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(input_ids)

    print("TinyGPT config:")
    print(config)

    print("\ninput_ids shape:", input_ids.shape)
    print("logits shape:   ", logits.shape)

    expected_shape = (
        batch_size,
        seq_len,
        config.vocab_size,
    )

    assert logits.shape == expected_shape, (
        f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}"
    )

    print("\nSuccess: TinyGPT forward pass works.")


if __name__ == "__main__":
    main()