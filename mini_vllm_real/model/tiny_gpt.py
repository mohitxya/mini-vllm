from __future__ import annotations

import torch
import torch.nn as nn

from mini_vllm_real.model.block import TransformerBlock
from mini_vllm_real.model.config import TinyGPTConfig


class TinyGPT(nn.Module):
    """
    Tiny GPT-style language model.

    This is the base model for our actual mini-vLLM path.

    It supports full-sequence forward pass first.

    Later milestones will add:
        - generation loop
        - contiguous KV cache
        - paged KV cache
        - paged attention
    """

    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.config = config

        # Converts token IDs into vectors.
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        # Learned positional embeddings.
        #
        # Position 0 has a vector.
        # Position 1 has a vector.
        # ...
        # Position max_seq_len - 1 has a vector.
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(config.d_model)

        # Language modeling head.
        #
        # Converts hidden vector at each position into vocabulary logits.
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights in a GPT-like way.

        This is not critical for inference mechanics, but helps keep values sane.
        """

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            input_ids:
                Shape [batch, seq_len]

        Returns:
            logits:
                Shape [batch, seq_len, vocab_size]
        """

        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}"
            )

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        device = input_ids.device

        # positions shape:
        #   [seq_len]
        positions = torch.arange(
            0,
            seq_len,
            device=device,
            dtype=torch.long,
        )

        # token_emb shape:
        #   [batch, seq_len, d_model]
        token_emb = self.token_embedding(input_ids)

        # pos_emb shape:
        #   [seq_len, d_model]
        pos_emb = self.position_embedding(positions)

        # Broadcasting:
        #   token_emb: [batch, seq_len, d_model]
        #   pos_emb:   [seq_len, d_model]
        #
        # Result:
        #   [batch, seq_len, d_model]
        x = token_emb + pos_emb

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)

        logits = self.lm_head(x)

        return logits