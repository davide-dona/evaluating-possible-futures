from collections.abc import Callable
from typing import Protocol

import numpy as np

# The pseudo-activity that ends a case.
END = '<END>'


class Process(Protocol):
    """A source of the distribution of the activity that follows a prefix."""

    def next_activity_probabilities(self, prefix: tuple[str, ...]) -> dict[str, float]:
        """The probability of every activity, or of `END`, right after `prefix`."""
        ...

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
        """Extend the prefix with the activity `pick` chooses among the next ones, until `END`."""
        suffix = []
        while len(suffix) < max_steps:
            activity = pick(self.next_activity_probabilities((*prefix, *suffix)))
            if activity == END:
                break
            suffix.append(activity)
        return tuple(suffix)


def _choose(options: dict[str, float], *, rng: np.random.Generator) -> str:
    """Draw one of the options with its probability."""
    threshold = rng.random()
    for option, probability in options.items():
        threshold -= probability
        if threshold < 0.0:
            return option
    # Rounding can leave the threshold just above zero after the last option.
    return option
