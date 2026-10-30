from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import DamerauLevenshtein

from improper_suffix_metrics.markov_process import Trace

# Predicts the suffix of a prefix as one or more drawn suffixes.
Forecaster = Callable[[Trace], list[Trace]]


def dls(predicted: Trace, observed: Trace) -> float:
    """Return the normalized Damerau-Levenshtein similarity of the activity sequences.

    It is 1 when the sequences are equal and 0 when they share nothing, so higher is better.
    """
    return DamerauLevenshtein.normalized_similarity(predicted.activities, observed.activities)


def length_error(predicted: Trace, observed: Trace) -> float:
    """Return the absolute difference between the numbers of activities of the suffixes."""
    return abs(len(predicted) - len(observed))


def remaining_time_error(predicted: Trace, observed: Trace) -> float:
    """Return the absolute difference in days between the durations of the suffixes."""
    return abs(predicted.duration - observed.duration)


@dataclass(frozen=True, kw_only=True)
class Prediction:
    """The suffixes a forecaster drew for a prefix of a test case, next to the observed suffix."""

    prefix: Trace
    draws: list[Trace]
    observed: Trace


def predict(forecaster: Forecaster, cases: list[Trace]) -> list[Prediction]:
    """Run a forecaster on every prefix of every test case.

    Each case of length `n` gives the prefixes of length 1 to `n - 1`, so that every prefix leaves
    a non-empty suffix. The forecaster predicts each prefix once.

    Args:
        forecaster: The model to run.
        cases: The test cases.

    Returns:
        The predictions, case by case and by increasing prefix length within a case.
    """
    return [
        Prediction(prefix=case[:cut], draws=forecaster(case[:cut]), observed=case[cut:])
        for case in cases
        for cut in range(1, len(case))
    ]


def evaluate(predictions: list[Prediction]) -> dict[str, float]:
    """Score predictions with the averaged point metrics.

    Each point metric compares every drawn suffix with the observed one, is averaged over the draws
    of each prediction, and then over all predictions.

    Args:
        predictions: The predictions to score.

    Returns:
        The mean DLS (higher is better), and the mean absolute errors of the suffix length and of
        the remaining time in days (lower is better), by name.
    """

    def average(metric: Callable[[Trace, Trace], float]) -> float:
        return np.mean([np.mean([metric(d, p.observed) for d in p.draws]) for p in predictions])

    return {
        'DLS': average(dls),
        'Length MAE': average(length_error),
        'Remaining Time MAE': average(remaining_time_error),
    }
