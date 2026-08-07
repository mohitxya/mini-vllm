from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # This matches Qwen's Hugging Face implementation exactly. In particular,
    # cos/sin span the full head dimension as [freqs, freqs].
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("cos_sin_cache", torch.empty(0), persistent=False)
        self.rebuild_cache()

    def rebuild_cache(self) -> None:
        """Recreate the non-parameter RoPE buffer after ``to_empty()``."""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim)
        )
        positions = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cache = torch.stack((emb.cos(), emb.sin()), dim=0)
        self.cos_sin_cache = cache.to(self.cos_sin_cache.device)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin_cache[:, positions]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
