from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from possible_futures.models.decoder import Decoder

# The pseudo-activity that ends a case.
END = '<END>'


class Process(ABC):
    """A process giving the distribution of the next activity after a prefix."""

    @abstractmethod
    def next_activity_probabilities(self, prefix: tuple[str, ...]) -> dict[str, float]:
        """The probability of every activity, or of `END`, right after `prefix`.

        Args:
            prefix: The activities so far.

        Returns:
            dict[str, float]: The probability of every activity that can follow, keyed by it.
        """

    def sample_suffix(
        self, prefix: tuple[str, ...], *, rng: np.random.Generator, max_steps: int
    ) -> tuple[str, ...]:
        """Draw the activities following `prefix` one at a time until `END`.

        Args:
            prefix: The activities the suffix follows.
            rng: Draws every activity.
            max_steps: The most activities the suffix holds.

        Returns:
            tuple[str, ...]: The activities of the suffix.
        """
        return self._decode(
            prefix, pick=lambda options: _choose(options, rng=rng), max_steps=max_steps
        )

    def greedy_suffix(self, prefix: tuple[str, ...], *, max_steps: int) -> tuple[str, ...]:
        """Take the most probable activity following the prefix at every step, until `END`.

        Args:
            prefix: The activities the suffix follows.
            max_steps: The most activities the suffix holds.

        Returns:
            tuple[str, ...]: The activities of the suffix.
        """
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
        """Extend the prefix with the activity `pick` chooses at every step, until `END`.

        Args:
            prefix: The activities the suffix follows.
            pick: Chooses the next activity from the probability of every activity.
            max_steps: The most activities the suffix holds.

        Returns:
            tuple[str, ...]: The activities of the suffix.
        """
        suffix = []
        while len(suffix) < max_steps:
            activity = pick(self.next_activity_probabilities((*prefix, *suffix)))
            if activity == END:
                break
            suffix.append(activity)
        return tuple(suffix)


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingDecoder(Decoder):
    """Draws every suffix from a process one activity at a time."""

    process: Process
    rng: np.random.Generator

    def draw(
        self, prefixes: Sequence[tuple[str, ...]], *, num_draws: int, max_steps: int
    ) -> list[list[tuple[str, ...]]]:
        return [
            [
                self.process.sample_suffix(prefix, rng=self.rng, max_steps=max_steps)
                for _ in range(num_draws)
            ]
            for prefix in prefixes
        ]


@dataclass(frozen=True, slots=True, kw_only=True)
class GreedyDecoder(Decoder):
    """Continues every prefix with the most probable activity of a process at every step, so
    all draws of a prefix are the same suffix."""

    process: Process

    def draw(
        self, prefixes: Sequence[tuple[str, ...]], *, num_draws: int, max_steps: int
    ) -> list[list[tuple[str, ...]]]:
        return [
            [self.process.greedy_suffix(prefix, max_steps=max_steps)] * num_draws
            for prefix in prefixes
        ]


def _choose(options: dict[str, float], *, rng: np.random.Generator) -> str:
    """Draw one of the options with its probability.

    Args:
        options: The probability of every option, keyed by it.
        rng: Draws the option.

    Returns:
        str: The option drawn.
    """
    threshold = rng.random()
    for option, probability in options.items():
        threshold -= probability
        if threshold < 0.0:
            return option
    # Rounding can leave the threshold just above zero after the last option.
    return option
