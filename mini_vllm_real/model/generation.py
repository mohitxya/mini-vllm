from __future__ import annotations

import torch

from mini_vllm_real.model.tiny_gpt import TinyGPT


@torch.no_grad()
def generate_full_recompute(
    model: TinyGPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Generate tokens autoregressively using full-sequence recomputation.

    This is the simplest possible generation loop.

    At every step:
        1. Run the model on the entire current sequence.
        2. Take logits from the final token position.
        3. Choose the next token greedily.
        4. Append that token.
        5. Repeat.

    This is intentionally inefficient.

    Why?
        Because it is the correctness baseline.

    Later we will implement:
        - contiguous KV-cache generation
        - paged KV-cache generation

    Those optimized versions should produce the same tokens as this function
    when using greedy decoding and the same model weights.

    Args:
        model:
            TinyGPT model.

        input_ids:
            Tensor of shape [batch, seq_len].

        max_new_tokens:
            Number of new tokens to generate.

        eos_token_id:
            Optional stop token. If every batch item generates EOS, stop early.

    Returns:
        generated_ids:
            Tensor of shape [batch, seq_len + generated_tokens].
    """

    if input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}"
        )

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    model.eval()

    generated_ids = input_ids

    for _ in range(max_new_tokens):
        seq_len = generated_ids.shape[1]

        if seq_len > model.config.max_seq_len:
            raise ValueError(
                f"Generated sequence length {seq_len} exceeded model max_seq_len "
                f"{model.config.max_seq_len}"
            )

        logits = model(generated_ids)

        # logits shape:
        # [batch, seq_len, vocab_size]
        #
        # We need the final position because it predicts the next token.
        next_token_logits = logits[:, -1, :]

        # Greedy decoding.
        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        # next_token_id shape:
        # [batch, 1]

        generated_ids = torch.cat(
            [generated_ids, next_token_id],
            dim=1,
        )

        if eos_token_id is not None:
            # Stop only if all batch items produced EOS.
            if torch.all(next_token_id.squeeze(-1) == eos_token_id):
                break

    return generated_ids