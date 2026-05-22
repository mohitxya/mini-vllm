from __future__ import annotations

import torch

from mini_vllm_real.model.config import TinyGPTConfig


class ContiguousKVCache:
    """
    Real contiguous KV cache for one sequence/request.

    This stores actual K/V tensors.

    Layout:

        keys:
            [n_layers, n_heads, max_seq_len, head_dim]

        values:
            [n_layers, n_heads, max_seq_len, head_dim]

    current_len:
        Number of token positions currently stored in the cache.

    This cache is "contiguous" because token position i is stored at index i.
    Later, paged KV cache will store logical token positions across physical blocks.
    """

    def __init__(
        self,
        config: TinyGPTConfig,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype

        self.keys = torch.empty(
            (
                config.n_layers,
                config.n_heads,
                config.max_seq_len,
                config.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        self.values = torch.empty(
            (
                config.n_layers,
                config.n_heads,
                config.max_seq_len,
                config.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        self.current_len = 0

    def reset(self) -> None:
        """
        Clear the cache logically.

        We do not zero memory because current_len determines what is valid.
        """

        self.current_len = 0

    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Append one token's K/V for a given layer.

        Args:
            layer_idx:
                Which transformer layer this K/V belongs to.

            key:
                Shape [n_heads, 1, head_dim] or [n_heads, head_dim]

            value:
                Shape [n_heads, 1, head_dim] or [n_heads, head_dim]

        Important:
            current_len is advanced outside this method, after all layers
            have written the token at the same position.

        Why?
            For one generated token, every layer must write to the same
            sequence position.
        """

        if layer_idx < 0 or layer_idx >= self.config.n_layers:
            raise IndexError(f"Invalid layer_idx={layer_idx}")

        if self.current_len >= self.config.max_seq_len:
            raise RuntimeError("KV cache is full")

        if key.dim() == 3:
            # [n_heads, 1, head_dim] -> [n_heads, head_dim]
            key = key[:, 0, :]

        if value.dim() == 3:
            value = value[:, 0, :]

        expected_shape = (
            self.config.n_heads,
            self.config.head_dim,
        )

        if tuple(key.shape) != expected_shape:
            raise ValueError(
                f"Expected key shape {expected_shape}, got {tuple(key.shape)}"
            )

        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Expected value shape {expected_shape}, got {tuple(value.shape)}"
            )

        self.keys[layer_idx, :, self.current_len, :] = key
        self.values[layer_idx, :, self.current_len, :] = value

    def advance(self) -> None:
        """
        Advance cache length by one token position.

        Call this after all layers have appended K/V for the current token.
        """

        if self.current_len >= self.config.max_seq_len:
            raise RuntimeError("Cannot advance: KV cache is full")

        self.current_len += 1

    def get_layer_kv(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return valid K/V tensors for one layer.

        Returns:
            key:
                [n_heads, current_len, head_dim]

            value:
                [n_heads, current_len, head_dim]
        """

        if layer_idx < 0 or layer_idx >= self.config.n_layers:
            raise IndexError(f"Invalid layer_idx={layer_idx}")

        key = self.keys[layer_idx, :, : self.current_len, :]
        value = self.values[layer_idx, :, : self.current_len, :]

        return key, value