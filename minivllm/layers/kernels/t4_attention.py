"""Turing (SM75) paged attention.

Prefill uses PyTorch SDPA, whose CUDA dispatcher can choose an efficient
memory-efficient kernel. Decode uses the Triton kernel below, which reads
directly from the paged KV cache and therefore does not materialize a dense
context for every generated token.
"""

from __future__ import annotations

import torch

from . import paged_sdpa

try:
    import triton
    import triton.language as tl
except ImportError:  # Keeps CPU-only installations importable.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _paged_decode_kernel(
        q_ptr, k_cache_ptr, v_cache_ptr, block_tables_ptr, seq_lens_ptr, out_ptr,
        q_stride_b, q_stride_h, cache_stride_slot, cache_stride_h,
        table_stride_b, out_stride_b, out_stride_h,
        scale,
        NUM_KV_HEADS: tl.constexpr, GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr, MAX_SEQ_LEN: tl.constexpr,
    ):
        batch = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // GROUP_SIZE
        d = tl.arange(0, BLOCK_D)
        d_mask = d < HEAD_DIM
        q = tl.load(q_ptr + batch * q_stride_b + q_head * q_stride_h + d,
                    mask=d_mask, other=0.0)
        seq_len = tl.load(seq_lens_ptr + batch)
        m = -float("inf")
        l = 0.0
        acc = tl.zeros((BLOCK_D,), tl.float32)

        for start in range(0, MAX_SEQ_LEN, BLOCK_N):
            pos = start + tl.arange(0, BLOCK_N)
            valid = pos < seq_len
            block = pos // BLOCK_SIZE
            block_id = tl.load(block_tables_ptr + batch * table_stride_b + block,
                               mask=valid, other=0)
            slots = block_id * BLOCK_SIZE + (pos % BLOCK_SIZE)
            k_ptrs = k_cache_ptr + slots[:, None] * cache_stride_slot + kv_head * cache_stride_h + d[None, :]
            v_ptrs = v_cache_ptr + slots[:, None] * cache_stride_slot + kv_head * cache_stride_h + d[None, :]
            k = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0)
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(valid, scores, -float("inf"))
            next_m = tl.maximum(m, tl.max(scores, axis=0))
            p = tl.exp(scores - next_m)
            alpha = tl.exp(m - next_m)
            v = tl.load(v_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = next_m

        tl.store(out_ptr + batch * out_stride_b + q_head * out_stride_h + d,
                 acc / l, mask=d_mask)


def _can_run(q: torch.Tensor) -> bool:
    return triton is not None and q.is_cuda and torch.cuda.get_device_capability(q.device) == (7, 5)


def paged_attention(q, k, v, k_cache, v_cache, context, scale):
    """Run the T4 decode kernel, falling back to SDPA for prefill or no Triton."""
    if context.is_prefill or not _can_run(q):
        return paged_sdpa.paged_attention(q, k, v, k_cache, v_cache, context, scale)

    if q.ndim != 3:
        raise ValueError("expected flattened decode queries [batch, heads, head_dim]")
    block_size = paged_sdpa._block_size(context)
    if q.shape[-1] > 256:
        # The educational T4 kernel is specialized for common LLM head sizes.
        return paged_sdpa.paged_attention(q, k, v, k_cache, v_cache, context, scale)

    max_seq_len = int(context.context_lens.max().item())
    block_d = triton.next_power_of_2(q.shape[-1])
    max_seq_len = triton.cdiv(max_seq_len, 32) * 32
    group_size = q.shape[1] // k_cache.shape[1]
    out = torch.empty_like(q)
    _paged_decode_kernel[(q.shape[0], q.shape[1])](
        q, k_cache, v_cache, context.block_tables, context.context_lens, out,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1),
        context.block_tables.stride(0), out.stride(0), out.stride(1), scale,
        NUM_KV_HEADS=k_cache.shape[1], GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[-1], BLOCK_D=block_d, BLOCK_SIZE=block_size, BLOCK_N=32,
        MAX_SEQ_LEN=max_seq_len, num_warps=4,
    )
    return out
