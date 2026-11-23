from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from possible_futures.cases import Activities
from possible_futures.forecasters.forecaster import Forecaster
from possible_futures.next_activity import NextActivityDistribution


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingForecaster(Forecaster):
    """Draws every suffix by sampling a distribution one activity at a time.

    Attributes:
        distribution: The distribution sampled.
        rng: Samples the activities.
    """

    distribution: NextActivityDistribution
    rng: np.random.Generator

    def draw(
        self, prefixes: Sequence[Activities], *, num_draws: int, max_steps: int
    ) -> list[list[Activities]]:
        return [
            [
                self.distribution.sample_suffix(prefix, rng=self.rng, max_steps=max_steps)
                for _ in range(num_draws)
            ]
            for prefix in prefixes
        ]


@dataclass(frozen=True, slots=True, kw_only=True)
class GreedyForecaster(Forecaster):
    """Continues every prefix with the most likely activity of a distribution at every step, so
    all draws of a prefix are the same suffix.

    Attributes:
        distribution: The distribution followed.
    """

    distribution: NextActivityDistribution

    def draw(
        self, prefixes: Sequence[Activities], *, num_draws: int, max_steps: int
    ) -> list[list[Activities]]:
        return [
            [self.distribution.greedy_suffix(prefix, max_steps=max_steps)] * num_draws
            for prefix in prefixes
        ]
