from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from possible_futures.forecasters.forecaster import Forecaster
from possible_futures.processes.process import Process


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingForecaster(Forecaster):
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
class GreedyForecaster(Forecaster):
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
