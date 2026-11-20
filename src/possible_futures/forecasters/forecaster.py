from collections.abc import Sequence
from typing import Protocol

from possible_futures.cases import Activities


class Forecaster(Protocol):
    """Imagines possible futures of prefixes: draws suffixes that could follow each one."""

    def draw(
        self, prefixes: Sequence[Activities], *, num_draws: int, max_steps: int
    ) -> list[list[Activities]]:
        """Draw `num_draws` suffixes per prefix, each at most `max_steps` activities. A draw is
        one possible future; a deterministic forecaster returns the same suffix every time.

        Args:
            prefixes: The activities of every prefix.
            num_draws: The number of suffixes drawn for each prefix.
            max_steps: The most activities a suffix holds.

        Returns:
            list[list[Activities]]: The activities of every suffix drawn, `num_draws` per
                prefix, in the order of `prefixes`.
        """
        ...
