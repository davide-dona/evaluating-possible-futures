from collections.abc import Callable

import numpy as np
from rapidfuzz.distance import DamerauLevenshtein

from improper_suffix_metrics.markov_process import MarkovProcess, Trace

# Draws `draw_count` suffixes for a prefix.
Forecaster = Callable[[Trace, int, np.random.Generator], list[Trace]]

METRICS = ('dls', 'length_mae', 'remaining_time_mae')


def oracle(process: MarkovProcess) -> Forecaster:
    """Return a forecaster that samples from the true suffix distribution."""

    def forecast(prefix: Trace, draw_count: int, rng: np.random.Generator) -> list[Trace]:
        return [process.sample_suffix(prefix.activities[-1], rng) for _ in range(draw_count)]

    return forecast


def mode_model(process: MarkovProcess) -> Forecaster:
    """Return a forecaster that always predicts the most probable suffix."""

    def forecast(prefix: Trace, draw_count: int, rng: np.random.Generator) -> list[Trace]:
        return [process.mode_suffix(prefix.activities[-1])] * draw_count

    return forecast


def score(draws: list[Trace], observed: Trace) -> tuple[float, float, float]:
    """Return the mean DLS, suffix length error, and remaining time error of the draws."""
    similarity = DamerauLevenshtein.normalized_similarity
    remaining_time = sum(observed.times)
    return (
        np.mean([similarity(d.activities, observed.activities) for d in draws]),
        np.mean([abs(len(d) - len(observed)) for d in draws]),
        np.mean([abs(sum(d.times) - remaining_time) for d in draws]),
    )


def evaluate(
    forecaster: Forecaster, cases: list[Trace], draw_count: int, rng: np.random.Generator
) -> dict[str, float]:
    """Return each metric averaged over every prefix of every case that leaves a suffix."""
    scores = [
        score(forecaster(case[:cut], draw_count, rng), case[cut:])
        for case in cases
        for cut in range(1, len(case))
    ]
    return dict(zip(METRICS, np.mean(scores, axis=0).tolist(), strict=True))


def compare(
    process: MarkovProcess, case_count: int, draw_count: int, rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    """Simulate test cases from the process and evaluate the oracle and the mode model on them."""
    cases = [process.sample(rng) for _ in range(case_count)]
    return {
        'oracle': evaluate(oracle(process), cases, draw_count, rng),
        'mode': evaluate(mode_model(process), cases, draw_count, rng),
    }
