from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm_real.cache.contiguous_kv_cache import ContiguousKVCache
from mini_vllm_real.model.config import TinyGPTConfig


class CausalSelfAttentionKV(nn.Module):
    """
    GPT-style self-attention with optional contiguous KV cache support.

    This class supports:

    1. full_forward(x)
        Used for normal full-sequence attention.

    2. decode_one(x, cache, layer_idx)
        Used for one-token decode with KV cache.

    We separate the two paths for clarity.
    """

    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len

        self.qkv_proj = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=True,
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=True,
        )

        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len)
        )

        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence attention.

        This is equivalent to your previous CausalSelfAttention.
        """

        batch_size, seq_len, d_model = x.shape

        if batch_size != 1:
            # We can support batch later.
            # For cache correctness, start with one sequence.
            pass

        if d_model != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, got {d_model}"
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        q, k, v = self._project_qkv(x)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(self.head_dim)

        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf"),
        )

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = self._merge_heads(context)

        out = self.out_proj(context)

        return out

    def decode_one(
        self,
        x: torch.Tensor,
        cache: ContiguousKVCache,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Decode exactly one token using contiguous KV cache.

        Args:
            x:
                Hidden state for one token.
                Shape [1, 1, d_model]

            cache:
                ContiguousKVCache storing previous tokens.

            layer_idx:
                Current transformer layer.

        Returns:
            Output hidden state for one token.
            Shape [1, 1, d_model]
        """

        if x.shape[0] != 1 or x.shape[1] != 1:
            raise ValueError(
                f"decode_one expects x shape [1, 1, d_model], got {tuple(x.shape)}"
            )

        q, k_new, v_new = self._project_qkv(x)

        # q/k/v shape:
        # [1, n_heads, 1, head_dim]

        # Remove batch dimension for cache append:
        # [n_heads, 1, head_dim]
        k_to_store = k_new[0]
        v_to_store = v_new[0]

        cache.append(
            layer_idx=layer_idx,
            key=k_to_store,
            value=v_to_store,
        )

        # Cache currently contains only OLD tokens because current_len has not
        # advanced yet. But we just wrote new K/V at index current_len.
        #
        # We need attention over old + new tokens.
        old_len = cache.current_len
        total_len = old_len + 1

        k_all = cache.keys[layer_idx, :, :total_len, :]
        v_all = cache.values[layer_idx, :, :total_len, :]

        # k_all/v_all shape:
        # [n_heads, total_len, head_dim]
        #
        # Add batch dimension:
        # [1, n_heads, total_len, head_dim]
        k_all = k_all.unsqueeze(0)
        v_all = v_all.unsqueeze(0)

        # q shape:
        # [1, n_heads, 1, head_dim]
        scores = torch.matmul(
            q,
            k_all.transpose(-2, -1),
        )

        # scores shape:
        # [1, n_heads, 1, total_len]
        scores = scores / math.sqrt(self.head_dim)

        # In decode mode, the current token is always allowed to attend to
        # all cached previous tokens plus itself, so no future mask is needed.
        attn_probs = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_probs, v_all)

        # context shape:
        # [1, n_heads, 1, head_dim]
        context = self._merge_heads(context)

        out = self.out_proj(context)

        return out

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project x into Q, K, V and split into heads.

        Returns:
            q, k, v each with shape:
            [batch, n_heads, seq_len, head_dim]
        """

        qkv = self.qkv_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        return q, k, v

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2)

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = x.shape

        x = x.transpose(1, 2)
        x = x.contiguous()

        x = x.view(
            batch_size,
            seq_len,
            n_heads * head_dim,
        )

        return x