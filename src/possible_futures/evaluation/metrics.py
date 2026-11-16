from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from possible_futures.distances import Distance, NormalizedDamerauLevenshtein


class Metric(Protocol):
    """A score of the suffixes drawn for one prefix against its observed suffix."""

    def __call__(self, *, draws: Sequence[tuple[int, ...]], observed: tuple[int, ...]) -> float:
        """The score of the draws, each the vocabulary indices of its activities."""
        ...


@dataclass(frozen=True, slots=True)
class EnergyScore(Metric):
    """The fair energy score of the draws, `1/K sum_i d(x_i, y) - 1/(2 K (K - 1)) sum_{i != j}
    d(x_i, x_j)`, whose second term is 0 for a single draw. Lower is better.

    Equal draws are measured once and weighted by how many times they were drawn.
    """

    distance: Distance

    def __call__(self, *, draws: Sequence[tuple[int, ...]], observed: tuple[int, ...]) -> float:
        counts = Counter(draws)
        unique = list(counts)
        weights = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        total = weights.sum()
        # [U, 1] -> [U]
        to_observed = self.distance(left=unique, right=[observed])[:, 0]
        distance_to_observed = float(weights @ to_observed) / total
        if total < 2:
            return distance_to_observed
        between_draws = self.distance(left=unique, right=unique)
        distance_between_draws = float(weights @ between_draws @ weights) / (
            2.0 * total * (total - 1.0)
        )
        return distance_to_observed - distance_between_draws


class MeanDLSimilarity(Metric):
    """The mean Damerau-Levenshtein similarity of the draws to the observed suffix, 1 minus
    their normalized Damerau-Levenshtein distance. Higher is better."""

    def __call__(self, *, draws: Sequence[tuple[int, ...]], observed: tuple[int, ...]) -> float:
        distances = NormalizedDamerauLevenshtein()(left=draws, right=[observed])  # [K, 1]
        return 1.0 - float(distances.mean())
