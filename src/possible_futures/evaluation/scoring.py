import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from possible_futures.cases import Case
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.evaluation.metrics import Metric
from possible_futures.inference.draws import Decoding


def mean_scores(
    decoding: Decoding,
    *,
    test: Sequence[Case],
    vocabulary: Vocabulary,
    metrics: Sequence[Metric],
    workers: int,
) -> dict[str, float]:
    """Score every forecast of a decoding against its observed suffix and average the scores.

    Args:
        decoding: The forecasts scored.
        test: The test cases, whose suffixes the forecasts are scored against.
        vocabulary: Maps the activities of the test cases to the tokens of the draws.
        metrics: The metrics. Each must pickle, to reach the worker processes.
        workers: Processes scoring the forecasts in parallel.

    Returns:
        dict[str, float]: The mean score of the forecasts under every metric, keyed by its
            name.
    """
    cases = {case.case_id: case for case in test}
    forecasts = decoding.forecasts
    draws = [forecast.draws for forecast in forecasts]
    observed = [
        tuple(vocabulary.encode(cases[forecast.case_id].suffix(forecast.prefix_length)).tolist())
        for forecast in forecasts
    ]
    chunksize = max(1, len(forecasts) // (16 * workers))
    score = partial(_scores, metrics=metrics)
    # One torch thread per process, so the processes do not oversubscribe the cores.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=torch.set_num_threads, initargs=(1,)
    ) as pool:
        scores = np.array(list(pool.map(score, draws, observed, chunksize=chunksize)))  # [N, M]
    means = scores.mean(axis=0).tolist()
    return {metric.name: mean for metric, mean in zip(metrics, means, strict=True)}


def _scores(
    draws: Sequence[tuple[int, ...]], observed: tuple[int, ...], *, metrics: Sequence[Metric]
) -> list[float]:
    return [metric(draws=draws, observed=observed) for metric in metrics]


def save_scores(means: Mapping[str, float], path: Path) -> None:
    """Write the mean scores of one decoding as JSON.

    Args:
        means: The mean score of the decoding under every metric, keyed by metric name.
        path: The JSON file written. Creates the parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(means), indent=2) + '\n')


def load_scores(path: Path) -> dict[str, float]:
    """Read the mean scores of one decoding from a JSON file.

    Args:
        path: The JSON file read.

    Returns:
        dict[str, float]: The mean score of the decoding under every metric, keyed by metric
            name.
    """
    return json.loads(path.read_text())
