from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from possible_futures.cases import Case
from possible_futures.forecasters.forecaster import Forecaster


@dataclass(frozen=True, slots=True, kw_only=True)
class UnconditionalForecaster(Forecaster):
    """Ignores the prefix: every draw is the suffix of a train prefix, drawn uniformly.

    The train suffixes are never longer than the longest train case, so draws are not truncated
    at `max_steps`.
    """

    suffixes: tuple[tuple[str, ...], ...]  # the suffix of every train prefix
    rng: np.random.Generator

    @classmethod
    def fit(cls, cases: Sequence[Case], *, rng: np.random.Generator) -> Self:
        """Collect the suffix of every prefix of the cases.

        Args:
            cases: The train cases.
            rng: Draws the suffixes.

        Returns:
            UnconditionalForecaster: The forecaster drawing among the suffixes, case after case and
                shortest prefix first within a case.
        """
        suffixes = tuple(case.suffix(length) for case in cases for length in case.prefix_lengths)
        return cls(suffixes=suffixes, rng=rng)

    def draw(
        self, prefixes: Sequence[tuple[str, ...]], *, num_draws: int, max_steps: int
    ) -> list[list[tuple[str, ...]]]:
        drawn = self.rng.integers(len(self.suffixes), size=len(prefixes) * num_draws).tolist()
        return [
            [self.suffixes[index] for index in drawn[start : start + num_draws]]
            for start in range(0, len(drawn), num_draws)
        ]
