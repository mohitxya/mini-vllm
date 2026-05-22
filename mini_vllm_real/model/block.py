from __future__ import annotations

import torch
import torch.nn as nn

from mini_vllm_real.model.attention import CausalSelfAttention
from mini_vllm_real.model.config import TinyGPTConfig
from mini_vllm_real.model.mlp import MLP


class TransformerBlock(nn.Module):
    """
    One GPT-style transformer block.

    Uses pre-layer normalization:

        x = x + attention(layer_norm(x))
        x = x + mlp(layer_norm(x))

    Residual connections help gradients and preserve information.
    """

    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer with residual connection.
        x = x + self.attn(self.ln_1(x))

        # MLP sublayer with residual connection.
        x = x + self.mlp(self.ln_2(x))

        return x