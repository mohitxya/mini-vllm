from __future__ import annotations

import torch
import torch.nn as nn

from mini_vllm_real.model.config import TinyGPTConfig


class MLP(nn.Module):
    """
    Feed-forward network inside a transformer block.

    GPT-style block usually has:

        Linear(d_model -> 4 * d_model)
        GELU
        Linear(4 * d_model -> d_model)

    This gives the model nonlinear transformation capacity after attention.
    """

    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)