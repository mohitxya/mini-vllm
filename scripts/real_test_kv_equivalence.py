import torch

from mini_vllm_real.model.config import TinyGPTConfig
from mini_vllm_real.model.generation import generate_full_recompute
from mini_vllm_real.model.generation_kv import generate_contiguous_kv
from mini_vllm_real.model.tiny_gpt_kv import TinyGPTKV


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

    model = TinyGPTKV(config)
    model.eval()

    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5]],
        dtype=torch.long,
    )

    max_new_tokens = 16

    with torch.no_grad():
        full_generated = generate_full_recompute(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

        kv_generated = generate_contiguous_kv(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

    print("Full recompute generated:")
    print(full_generated)

    print("\nContiguous KV generated:")
    print(kv_generated)

    same = torch.equal(full_generated, kv_generated)

    print("\nAre generated token IDs exactly equal?", same)

    if not same:
        diff_positions = torch.nonzero(full_generated != kv_generated)
        print("\nDifferent positions:")
        print(diff_positions)

        raise AssertionError(
            "Full recompute and contiguous KV generation produced different tokens."
        )

    print("\nSuccess: full recompute == contiguous KV generation.")


if __name__ == "__main__":
    main()