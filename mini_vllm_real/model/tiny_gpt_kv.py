from __future__ import annotations

import torch
import torch.nn as nn

from mini_vllm_real.cache.contiguous_kv_cache import ContiguousKVCache
from mini_vllm_real.model.block_kv import TransformerBlockKV
from mini_vllm_real.model.config import TinyGPTConfig


class TinyGPTKV(nn.Module):
    """
    Tiny GPT model with contiguous KV-cache support.

    Supports:
        - full forward pass
        - decode_one with KV cache
    """

    def __init__(self, config: TinyGPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlockKV(
                    config=config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.n_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
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
        Full-sequence forward.

        Same behavior as TinyGPT, but implemented with KV-capable blocks.
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

        positions = torch.arange(
            0,
            seq_len,
            device=device,
            dtype=torch.long,
        )

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        return logits

    def decode_one(
        self,
        token_id: torch.Tensor,
        cache: ContiguousKVCache,
    ) -> torch.Tensor:
        """
        Decode exactly one token using contiguous KV cache.

        Args:
            token_id:
                Shape [1, 1]

            cache:
                ContiguousKVCache

        Returns:
            logits:
                Shape [1, 1, vocab_size]
        """

        if token_id.shape != (1, 1):
            raise ValueError(
                f"decode_one expects token_id shape [1, 1], got {tuple(token_id.shape)}"
            )

        if cache.current_len >= self.config.max_seq_len:
            raise RuntimeError("Cannot decode: cache is full")

        device = token_id.device

        # Position of the new token is current cache length.
        position_id = torch.tensor(
            [[cache.current_len]],
            device=device,
            dtype=torch.long,
        )

        token_emb = self.token_embedding(token_id)
        pos_emb = self.position_embedding(position_id)

        x = token_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block.decode_one(
                x=x,
                cache=cache,
            )

        x = self.final_ln(x)

        logits = self.lm_head(x)

        # All layers have written K/V for this token.
        # Now it is safe to advance cache length.
        cache.advance()

        return logits

    def create_cache(
        self,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ContiguousKVCache:
        """
        Create a fresh contiguous KV cache for one request.
        """

        return ContiguousKVCache(
            config=self.config,
            device=device,
            dtype=dtype,
        )