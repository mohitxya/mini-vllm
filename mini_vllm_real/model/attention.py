from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm_real.model.config import TinyGPTConfig


class CausalSelfAttention(nn.Module):
    """
    Standard GPT-style causal self-attention.

    This version does NOT use KV cache yet.

    Input:
        x shape: [batch, seq_len, d_model]

    Output:
        out shape: [batch, seq_len, d_model]
    """

    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )

        # One projection creates Q, K, and V together.
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

        # Lower-triangular causal mask.
        # Shape before view: [max_seq_len, max_seq_len]
        mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len)
        )

        # Final shape: [1, 1, max_seq_len, max_seq_len]
        # This broadcasts over batch and heads.
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run full-sequence causal self-attention.

        Args:
            x:
                Shape [batch, seq_len, d_model]

        Returns:
            Shape [batch, seq_len, d_model]
        """

        batch_size, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, got {d_model}"
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        # ------------------------------------------------------------
        # 1. Compute Q, K, V.
        # ------------------------------------------------------------
        qkv = self.qkv_proj(x)

        # qkv shape:
        # [batch, seq_len, 3 * d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # q/k/v shape before splitting heads:
        # [batch, seq_len, d_model]

        # ------------------------------------------------------------
        # 2. Split into attention heads.
        # ------------------------------------------------------------
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # q/k/v shape:
        # [batch, n_heads, seq_len, head_dim]

        # ------------------------------------------------------------
        # 3. Compute scaled dot-product attention scores.
        # ------------------------------------------------------------
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        # scores shape:
        # [batch, n_heads, seq_len, seq_len]

        scores = scores / math.sqrt(self.head_dim)

        # ------------------------------------------------------------
        # 4. Apply causal mask.
        # ------------------------------------------------------------
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf"),
        )

        # ------------------------------------------------------------
        # 5. Softmax over keys.
        # ------------------------------------------------------------
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # ------------------------------------------------------------
        # 6. Weighted sum of values.
        # ------------------------------------------------------------
        context = torch.matmul(attn_probs, v)

        # context shape:
        # [batch, n_heads, seq_len, head_dim]

        # ------------------------------------------------------------
        # 7. Merge heads.
        # ------------------------------------------------------------
        context = self._merge_heads(context)

        # context shape:
        # [batch, seq_len, d_model]

        # ------------------------------------------------------------
        # 8. Final output projection.
        # ------------------------------------------------------------
        out = self.out_proj(context)

        return out

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert:
            [batch, seq_len, d_model]

        into:
            [batch, n_heads, seq_len, head_dim]
        """

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
        """
        Convert:
            [batch, n_heads, seq_len, head_dim]

        into:
            [batch, seq_len, d_model]
        """

        batch_size, n_heads, seq_len, head_dim = x.shape

        x = x.transpose(1, 2)
        x = x.contiguous()

        x = x.view(
            batch_size,
            seq_len,
            n_heads * head_dim,
        )

        return x