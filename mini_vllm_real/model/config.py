from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TinyGPTConfig:
    """
    Configuration for our tiny GPT model.

    This is intentionally small so we can understand and debug every tensor.

    vocab_size:
        Number of possible token IDs.

        For a real tokenizer, this may be 50,000+.
        For our tiny model, we start with 256.

    d_model:
        Width of the model. Every token becomes a vector of this size.

    n_heads:
        Number of attention heads.

    n_layers:
        Number of transformer blocks.

    max_seq_len:
        Maximum sequence length supported by learned positional embeddings.

    dropout:
        Dropout probability.

        For inference-only experiments, dropout should be 0.0.
        If you later train the tiny model, you can use 0.1.
    """

    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.d_model <= 0:
            raise ValueError("d_model must be positive")

        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")

        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads. "
                f"Got d_model={self.d_model}, n_heads={self.n_heads}"
            )

    @property
    def head_dim(self) -> int:
        """
        Dimension of one attention head.

        Example:
            d_model = 128
            n_heads = 4
            head_dim = 32
        """

        return self.d_model // self.n_heads