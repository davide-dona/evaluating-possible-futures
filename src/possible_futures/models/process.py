from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from possible_futures.datasets.encoding import PrefixTokens
from possible_futures.datasets.vocabulary import SpecialToken, Vocabulary
from possible_futures.models.decoder import Decoder

# The pseudo-activity that ends a case.
END = '<END>'


class Process(ABC):
    """A process giving the distribution of the next activity after a prefix."""

    @abstractmethod
    def next_activity_probabilities(self, prefix: tuple[str, ...]) -> dict[str, float]:
        """The probability of every activity, or of `END`, right after `prefix`."""

    def sample_suffix(
        self, prefix: tuple[str, ...], *, rng: np.random.Generator, max_steps: int
    ) -> tuple[str, ...]:
        """Draw the activities following `prefix` one at a time until `END`, at most
        `max_steps` of them."""
        return self._decode(
            prefix, pick=lambda options: _choose(options, rng=rng), max_steps=max_steps
        )

    def greedy_suffix(self, prefix: tuple[str, ...], *, max_steps: int) -> tuple[str, ...]:
        """Take the most probable activity following the prefix at every step, until `END`,
        at most `max_steps` of them."""
        return self._decode(
            prefix, pick=lambda options: max(options, key=options.get), max_steps=max_steps
        )

    def _decode(
        self,
        prefix: tuple[str, ...],
        *,
        pick: Callable[[dict[str, float]], str],
        max_steps: int,
    ) -> tuple[str, ...]:
        """Extend the prefix with the activity `pick` chooses at every step, until `END`."""
        suffix = []
        while len(suffix) < max_steps:
            activity = pick(self.next_activity_probabilities((*prefix, *suffix)))
            if activity == END:
                break
            suffix.append(activity)
        return tuple(suffix)


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingDecoder(Decoder):
    """Draws every suffix from a process one activity at a time.

    `vocabulary` must hold every activity the process emits.
    """

    process: Process
    vocabulary: Vocabulary
    rng: np.random.Generator

    def draw(self, prefixes: PrefixTokens, *, num_draws: int, max_steps: int) -> torch.Tensor:
        return _continue_prefixes(
            prefixes,
            decode=lambda prefix: self.process.sample_suffix(
                prefix, rng=self.rng, max_steps=max_steps
            ),
            vocabulary=self.vocabulary,
            num_draws=num_draws,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GreedyDecoder(Decoder):
    """Continues every prefix with the most probable activity of a process at every step, so
    all draws of a prefix are the same suffix.

    `vocabulary` must hold every activity the process emits.
    """

    process: Process
    vocabulary: Vocabulary

    def draw(self, prefixes: PrefixTokens, *, num_draws: int, max_steps: int) -> torch.Tensor:
        return _continue_prefixes(
            prefixes,
            decode=lambda prefix: self.process.greedy_suffix(prefix, max_steps=max_steps),
            vocabulary=self.vocabulary,
            num_draws=num_draws,
        )


def _choose(options: dict[str, float], *, rng: np.random.Generator) -> str:
    """Draw one of the options with its probability."""
    threshold = rng.random()
    for option, probability in options.items():
        threshold -= probability
        if threshold < 0.0:
            return option
    # Rounding can leave the threshold just above zero after the last option.
    return option


def _continue_prefixes(
    prefixes: PrefixTokens,
    *,
    decode: Callable[[tuple[str, ...]], tuple[str, ...]],
    vocabulary: Vocabulary,
    num_draws: int,
) -> torch.Tensor:
    """Continue every prefix `num_draws` times with `decode`, which maps the activities of a
    prefix to the activities of a suffix.

    Returns:
        The tokens `[B, S, L]` on the device of `prefixes`, PAD past each draw's end.
    """
    labels = vocabulary.labels
    tokens = prefixes.tokens.tolist()
    draws = [
        vocabulary.encode(decode(tuple(labels[token] for token in row[:length])))
        for row, length in zip(tokens, prefixes.lengths.tolist(), strict=True)
        for _ in range(num_draws)
    ]
    width = max((len(draw) for draw in draws), default=0)
    padded = np.full(shape=(len(draws), width), fill_value=SpecialToken.PAD, dtype=np.int64)
    for row, draw in zip(padded, draws, strict=True):
        row[: len(draw)] = draw
    # [B * S, L] -> [B, S, L]
    return torch.from_numpy(padded).view(len(tokens), num_draws, width).to(prefixes.lengths.device)
