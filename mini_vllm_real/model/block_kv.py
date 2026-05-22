from __future__ import annotations

import torch
import torch.nn as nn

from mini_vllm_real.cache.contiguous_kv_cache import ContiguousKVCache
from mini_vllm_real.model.attention_kv import CausalSelfAttentionKV
from mini_vllm_real.model.config import TinyGPTConfig
from mini_vllm_real.model.mlp import MLP


class TransformerBlockKV(nn.Module):
    """
    Transformer block with KV-cache-aware attention.
    """

    def __init__(
        self,
        config: TinyGPTConfig,
        layer_idx: int,
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttentionKV(config)

        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-sequence forward path.
        """

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

    def decode_one(
        self,
        x: torch.Tensor,
        cache: ContiguousKVCache,
    ) -> torch.Tensor:
        """
        One-token decode path using KV cache.
        """

        x = x + self.attn.decode_one(
            x=self.ln_1(x),
            cache=cache,
            layer_idx=self.layer_idx,
        )

        x = x + self.mlp(self.ln_2(x))

        return x