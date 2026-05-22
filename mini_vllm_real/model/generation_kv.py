from __future__ import annotations

import torch

from mini_vllm_real.cache.contiguous_kv_cache import ContiguousKVCache
from mini_vllm_real.model.tiny_gpt_kv import TinyGPTKV


@torch.no_grad()
def generate_contiguous_kv(
    model: TinyGPTKV,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Generate using real contiguous KV cache.

    This function processes the prompt token by token into the cache, then
    continues decoding one token at a time.

    It is intentionally simple and supports batch size 1 for now.

    Args:
        model:
            TinyGPTKV model.

        input_ids:
            Shape [1, seq_len]

        max_new_tokens:
            Number of tokens to generate.

    Returns:
        generated_ids:
            Shape [1, seq_len + generated_tokens]
    """

    if input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must have shape [1, seq_len], got {tuple(input_ids.shape)}"
        )

    if input_ids.shape[0] != 1:
        raise ValueError(
            "generate_contiguous_kv currently supports batch size 1 only"
        )

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    model.eval()

    device = input_ids.device

    cache = model.create_cache(
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    generated_ids = input_ids.clone()

    # ------------------------------------------------------------
    # Prefill prompt token by token.
    # ------------------------------------------------------------
    #
    # For correctness, we feed every prompt token through decode_one.
    # That fills the cache with K/V for prompt tokens.
    #
    # We do not sample during prompt prefill.
    # We only use the logits from the final prompt token to choose the first
    # generated token.
    last_logits = None

    prompt_len = input_ids.shape[1]

    for pos in range(prompt_len):
        token_id = input_ids[:, pos : pos + 1]
        last_logits = model.decode_one(
            token_id=token_id,
            cache=cache,
        )

    if last_logits is None:
        raise RuntimeError("Prompt must contain at least one token")

    # ------------------------------------------------------------
    # Generate new tokens.
    # ------------------------------------------------------------
    for _ in range(max_new_tokens):
        next_token_logits = last_logits[:, -1, :]

        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        generated_ids = torch.cat(
            [generated_ids, next_token_id],
            dim=1,
        )

        if eos_token_id is not None:
            if torch.all(next_token_id.squeeze(-1) == eos_token_id):
                break

        # Feed the generated token to update cache and get next logits.
        if generated_ids.shape[1] >= model.config.max_seq_len:
            break

        last_logits = model.decode_one(
            token_id=next_token_id,
            cache=cache,
        )

    return generated_ids