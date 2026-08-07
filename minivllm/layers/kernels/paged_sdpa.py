"""Reference paged-KV attention built on top of PyTorch SDPA.

This is deliberately a correctness-first backend.  It converts only the
requested pages of the KV cache to dense K/V tensors, then lets PyTorch choose
the best CPU/CUDA SDPA implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _block_size(context) -> int:
    try:
        return int(context.block_size)
    except AttributeError as exc:
        raise RuntimeError(
            "Paged attention requires context.block_size. Add the scheduler's "
            "KV-cache block size to AttentionContext."
        ) from exc


def gather_paged_kv(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Return a dense ``[seq_len, num_kv_heads, head_dim]`` cache view."""
    num_blocks = math.ceil(seq_len / block_size)
    block_ids = block_table[:num_blocks].to(device=cache.device, dtype=torch.long)
    offsets = torch.arange(block_size, device=cache.device)
    slots = (block_ids[:, None] * block_size + offsets).reshape(-1)[:seq_len]
    return cache.index_select(0, slots)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Causal attention where the query is right-aligned with the KV sequence."""
    q_len, num_heads, _ = q.shape
    k_len, num_kv_heads, _ = k.shape
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads for GQA")

    # SDPA's GQA flag is not available in every PyTorch version. Repeating KV is
    # safe and is acceptable for this fallback; the Triton T4 decode path avoids
    # this expansion.
    if num_heads != num_kv_heads:
        groups = num_heads // num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    q = q.transpose(0, 1).unsqueeze(0)  # [1, Hq, Q, D]
    k = k.transpose(0, 1).unsqueeze(0)  # [1, Hq, K, D]
    v = v.transpose(0, 1).unsqueeze(0)

    # A decode query is the final token(s) in the sequence. is_causal=True is
    # upper-left aligned for non-square matrices, so build the right-aligned
    # causal mask explicitly instead.
    query_positions = torch.arange(k_len - q_len, k_len, device=q.device)[:, None]
    key_positions = torch.arange(k_len, device=q.device)[None, :]
    mask = key_positions <= query_positions
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale
    )
    return out.squeeze(0).transpose(0, 1)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run causal SDPA on flattened, variable-length Q/K/V inputs."""
    out = torch.empty_like(q)
    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    k_offsets = cu_seqlens_k.detach().cpu().tolist()
    for q_start, q_end, k_start, k_end in zip(
        q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:]
    ):
        out[q_start:q_end] = _sdpa(q[q_start:q_end], k[k_start:k_end], v[k_start:k_end], scale)
    return out


def paged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context,
    scale: float,
) -> torch.Tensor:
    """Reference fallback for prefill, prefix-cache prefill, and decode."""
    if context.is_prefill and context.block_tables is None:
        return varlen_attention(q, k, v, context.cu_seqlens_q, context.cu_seqlens_k, scale)

    block_size = _block_size(context)
    block_tables = context.block_tables
    if context.is_prefill:
        k_offsets = context.cu_seqlens_k.detach().cpu().tolist()
        seq_lens = [end - start for start, end in zip(k_offsets[:-1], k_offsets[1:])]
    else:
        seq_lens = context.context_lens.detach().cpu().tolist()
    q_offsets = context.cu_seqlens_q.detach().cpu().tolist() if context.is_prefill else list(range(q.shape[0] + 1))
    out = torch.empty_like(q)
    for i, seq_len in enumerate(seq_lens):
        q_start, q_end = q_offsets[i], q_offsets[i + 1]
        dense_k = gather_paged_kv(k_cache, block_tables[i], seq_len, block_size)
        dense_v = gather_paged_kv(v_cache, block_tables[i], seq_len, block_size)
        out[q_start:q_end] = _sdpa(q[q_start:q_end], dense_k, dense_v, scale)
    return out
