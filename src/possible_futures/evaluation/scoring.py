import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from possible_futures.cases import Case
from possible_futures.datasets.tokenizer import Tokenizer
from possible_futures.evaluation.metrics import Metric
from possible_futures.forecasts import Forecast


def mean_scores(
    forecasts: Sequence[Forecast],
    *,
    test: Sequence[Case],
    tokenizer: Tokenizer,
    metrics: Sequence[Metric],
    workers: int,
) -> dict[str, float]:
    """Score every forecast against its observed suffix and average the scores.

    Args:
        forecasts: The forecasts scored.
        test: The test cases, whose suffixes the forecasts are scored against.
        tokenizer: The tokenizer of the train split. Every observed activity it does not know
            reads as `<UNK>`, the one unknown activity a model can draw.
        metrics: The metrics. Each must pickle, to reach the worker processes.
        workers: Processes scoring the forecasts in parallel.

    Returns:
        dict[str, float]: The mean score of the forecasts under every metric, keyed by its
            name.
    """
    cases = {case.case_id: case for case in test}
    draws = [forecast.draws for forecast in forecasts]
    observed = [
        tokenizer.known(cases[forecast.case_id].suffix(forecast.prefix_length))
        for forecast in forecasts
    ]
    chunksize = max(1, len(forecasts) // (16 * workers))
    score = partial(_scores, metrics=metrics)
    # One torch thread per process, so the processes do not oversubscribe the cores.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=torch.set_num_threads, initargs=(1,)
    ) as pool:
        # One row per forecast, one column per metric
        scores = np.array(list(pool.map(score, draws, observed, chunksize=chunksize)))
    means = scores.mean(axis=0).tolist()
    return {metric.name: mean for metric, mean in zip(metrics, means, strict=True)}


def _scores(
    draws: Sequence[tuple[str, ...]], observed: tuple[str, ...], *, metrics: Sequence[Metric]
) -> list[float]:
    """Score the draws of one prefix against its observed suffix under every metric.

    Args:
        draws: The activities of every draw.
        observed: The activities of the observed suffix.
        metrics: The metrics.

    Returns:
        list[float]: The score under every metric, in the order of `metrics`.
    """
    return [metric(draws=draws, observed=observed) for metric in metrics]


def save_scores(means: Mapping[str, float], path: Path) -> None:
    """Write the mean scores of one set of forecasts as JSON.

    Args:
        means: The mean score of the forecasts under every metric, keyed by metric name.
        path: The JSON file written. Creates the parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(means), indent=2) + '\n')


def load_scores(path: Path) -> dict[str, float]:
    """Read the mean scores of one set of forecasts from a JSON file.

    Args:
        path: The JSON file read.

    Returns:
        dict[str, float]: The mean score of the forecasts under every metric, keyed by metric
            name.
    """
    return json.loads(path.read_text())
