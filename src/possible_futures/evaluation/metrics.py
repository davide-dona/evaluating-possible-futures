from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from possible_futures.cases import Activities
from possible_futures.distances import Distance, NormalizedDamerauLevenshtein


class Metric(Protocol):
    """A score of the suffixes drawn for one prefix against the suffix that actually followed.

    Attributes:
        name: Labels the mean score in the scores files and tables.
    """

    name: str

    def __call__(self, *, draws: Sequence[Activities], observed: Activities) -> float:
        """Score the draws of one prefix.

        Args:
            draws: The `S` suffixes drawn for the prefix, each a tuple of activities.
            observed: The suffix that actually followed the prefix.

        Returns:
            float: The score of the draws.
        """
        ...


@dataclass(frozen=True, slots=True)
class EnergyScore(Metric):
    """The fair energy score of the draws, lower is better:
    `1/S sum_i d(x_i, y) - 1/(2 S (S - 1)) sum_{i != j} d(x_i, x_j)`.

    The first term is the mean distance of the draws to the observed suffix `y`, which rewards
    accuracy. The second is half the mean distance between two draws, which rewards draws that
    spread as widely as the real futures do. Without it, the score would favour always drawing
    the single most typical suffix. With a single draw the second term is 0.

    Attributes:
        name: Labels the mean score in the scores files and tables.
        distance: The distance between two suffixes, `d`.
    """

    name: str
    distance: Distance

    def __call__(self, *, draws: Sequence[Activities], observed: Activities) -> float:
        counts = Counter(draws)
        unique = list(counts)
        weights = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        total = weights.sum()
        to_observed = self.distance(left=unique, right=[observed])[:, 0]  # [U, 1] -> [U]
        distance_to_observed = float(weights @ to_observed) / total
        if total < 2:
            return distance_to_observed
        between_draws = self.distance(left=unique, right=unique)
        distance_between_draws = float(weights @ between_draws @ weights) / (
            2.0 * total * (total - 1.0)
        )
        return distance_to_observed - distance_between_draws


@dataclass(frozen=True, slots=True)
class MeanDLSimilarity(Metric):
    """How close the draws are to the observed suffix on average: 1 minus their mean normalized
    Damerau-Levenshtein distance to it. Higher is better.

    Attributes:
        name: Labels the mean score in the scores files and tables.
    """

    name: str

    def __call__(self, *, draws: Sequence[Activities], observed: Activities) -> float:
        distances = NormalizedDamerauLevenshtein()(left=draws, right=[observed])  # [S, 1]
        return 1.0 - float(distances.mean())
