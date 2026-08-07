"""Per-forward attention metadata set by the scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_size: int = 0


_context = Context()


def set_context(context: Context) -> None:
    global _context
    _context = context


def get_context() -> Context:
    return _context
