from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class SamplingStrategy(str, Enum):
    """
    Supported token selection strategies.

    GREEDY:
        Always choose the highest-logit token.

    TEMPERATURE:
        Sample from the full probability distribution after temperature scaling.

    TOP_K:
        Keep only the top-k highest-logit tokens, then sample among them.
    """

    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"


@dataclass
class SamplingConfig:
    """
    Configuration for token sampling.

    strategy:
        Which sampling method to use.

    temperature:
        Controls randomness.

        Lower temperature:
            sharper distribution, more deterministic.

        Higher temperature:
            flatter distribution, more random.

        temperature = 1.0:
            original model distribution.

    top_k:
        Number of highest-probability tokens to keep for top-k sampling.

    seed:
        Optional random seed for reproducible sampling.
    """

    strategy: SamplingStrategy = SamplingStrategy.GREEDY
    temperature: float = 1.0
    top_k: int = 50
    seed: int | None = None


def sample_next_token(
    logits: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    """
    Select the next token from logits.

    Args:
        logits:
            Shape [batch_size, vocab_size]

        config:
            Sampling configuration.

    Returns:
        next_token_id:
            Shape [batch_size, 1]
    """

    if logits.dim() != 2:
        raise ValueError(
            f"Expected logits shape [batch_size, vocab_size], got {tuple(logits.shape)}"
        )

    if config.seed is not None:
        torch.manual_seed(config.seed)

    if config.strategy == SamplingStrategy.GREEDY:
        return greedy_sample(logits)

    if config.strategy == SamplingStrategy.TEMPERATURE:
        return temperature_sample(
            logits=logits,
            temperature=config.temperature,
        )

    if config.strategy == SamplingStrategy.TOP_K:
        return top_k_sample(
            logits=logits,
            temperature=config.temperature,
            top_k=config.top_k,
        )

    raise ValueError(f"Unknown sampling strategy: {config.strategy}")


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy decoding.

    Chooses the token with highest logit.

    Deterministic:
        same logits -> same token
    """

    return torch.argmax(
        logits,
        dim=-1,
        keepdim=True,
    )


def temperature_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Temperature sampling.

    Steps:
        1. Divide logits by temperature.
        2. Convert logits to probabilities using softmax.
        3. Randomly sample from the distribution.

    temperature < 1:
        more confident / less random

    temperature > 1:
        more random

    temperature = 1:
        unchanged distribution
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    return torch.multinomial(
        probs,
        num_samples=1,
    )


def top_k_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Top-k sampling.

    Instead of sampling from the entire vocabulary, keep only the k highest-logit
    tokens and sample from those.

    Example:
        vocab size = 50,257
        top_k = 50

    Then only the 50 most likely tokens are considered.

    This prevents extremely unlikely weird tokens from being sampled.
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    vocab_size = logits.shape[-1]

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    top_k = min(top_k, vocab_size)

    scaled_logits = logits / temperature

    # values shape:  [batch_size, top_k]
    # indices shape: [batch_size, top_k]
    values, indices = torch.topk(
        scaled_logits,
        k=top_k,
        dim=-1,
    )

    probs = F.softmax(values, dim=-1)

    sampled_position_in_topk = torch.multinomial(
        probs,
        num_samples=1,
    )

    next_token_id = torch.gather(
        indices,
        dim=-1,
        index=sampled_position_in_topk,
    )

    return next_token_id