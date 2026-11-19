from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

# The pseudo-activity a process gives as the next one when the case is over, the activity-level
# counterpart of the EOS token.
END = '<END>'


class Process(ABC):
    """A model of how cases unfold: the probability of every possible next activity after a
    prefix. Suffixes follow from applying it repeatedly."""

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
        """Sample the activities following `prefix` one at a time until `END`.

        Args:
            prefix: The activities the suffix follows.
            rng: Samples every activity.
            max_steps: The most activities the suffix holds.

        Returns:
            tuple[str, ...]: The activities of the suffix.
        """
        return self._suffix(
            prefix,
            choose=lambda options: _sample_activity(options, rng=rng),
            max_steps=max_steps,
        )

    def greedy_suffix(self, prefix: tuple[str, ...], *, max_steps: int) -> tuple[str, ...]:
        """Take the most likely next activity at every step, until `END`. This is the greedy
        suffix: the most likely activity at each step, not necessarily the most likely suffix
        as a whole.

        Args:
            prefix: The activities the suffix follows.
            max_steps: The most activities the suffix holds.

        Returns:
            tuple[str, ...]: The activities of the suffix.
        """
        return self._suffix(
            prefix, choose=lambda options: max(options, key=options.get), max_steps=max_steps
        )

    def _suffix(
        self,
        prefix: tuple[str, ...],
        *,
        choose: Callable[[dict[str, float]], str],
        max_steps: int,
    ) -> tuple[str, ...]:
        """Extend the prefix with the activity `choose` picks at every step, until `END`.

        Args:
            prefix: The activities the suffix follows.
            choose: Picks the next activity from the probability of every activity.
            max_steps: The most activities the suffix holds.

        Returns:
            tuple[str, ...]: The activities of the suffix.
        """
        suffix = []
        while len(suffix) < max_steps:
            activity = choose(self.next_activity_probabilities((*prefix, *suffix)))
            if activity == END:
                break
            suffix.append(activity)
        return tuple(suffix)


def _sample_activity(options: dict[str, float], *, rng: np.random.Generator) -> str:
    """Sample one activity, each with its probability.

    Args:
        options: The probability of every activity, keyed by it.
        rng: Samples the activity.

    Returns:
        str: The activity sampled.
    """
    threshold = rng.random()
    for option, probability in options.items():
        threshold -= probability
        if threshold < 0.0:
            return option
    # Rounding can leave the threshold just above zero after the last option.
    return option
