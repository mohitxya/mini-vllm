"""Attention dispatch: FlashAttention, Turing paged attention, or CPU SDPA."""

from __future__ import annotations

import torch
from torch import nn

from minivllm.layers.kernels import paged_sdpa, t4_attention
from minivllm.utils.context import get_context

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = flash_attn_with_kvcache = None


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write flattened K/V tokens to their physical paged-cache slots.

    Tensor indexing is portable to CPU and CUDA. A dedicated write kernel can
    replace this later without changing the attention-backend interface.
    """
    if slot_mapping.numel() != key.shape[0]:
        raise ValueError("slot_mapping must contain one slot per K/V token")
    valid = slot_mapping >= 0
    slots = slot_mapping[valid].to(device=k_cache.device, dtype=torch.long)
    k_cache[slots] = key[valid]
    v_cache[slots] = value[valid]


def _supports_flash_attn(q: torch.Tensor) -> bool:
    """FA2's CUDA implementation supports Ampere and newer NVIDIA GPUs."""
    return (
        flash_attn_varlen_func is not None
        and q.is_cuda
        and torch.cuda.get_device_capability(q.device) >= (8, 0)
    )


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _flash_attention(self, q, k, v, k_cache, v_cache, context):
        if context.is_prefill:
            if context.block_tables is not None:  # prefix-cache prefill
                k, v = k_cache, v_cache
            return flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        return flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
        ).squeeze(1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if _supports_flash_attn(q):
            return self._flash_attention(q, k, v, k_cache, v_cache, context)
        if q.is_cuda:
            # On T4 this dispatches to direct paged-KV Triton decode. Other
            # CUDA devices use its correct SDPA fallback until a backend is
            # added for that architecture.
            return t4_attention.paged_attention(
                q, k, v, k_cache, v_cache, context, self.scale
            )
        return paged_sdpa.paged_attention(q, k, v, k_cache, v_cache, context, self.scale)
