from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from possible_futures.evaluation.metrics import Metric
from possible_futures.inference.draws import Decoding

# The scores file has one row per decoding, indexed by its name in this column, and one column
# per metric.
_DECODING_COLUMN = 'decoding'


def mean_scores(
    decoding: Decoding,
    *,
    traces: Mapping[str, tuple[int, ...]],
    metrics: Mapping[str, Metric],
    workers: int,
) -> dict[str, float]:
    """Score every forecast of a decoding against its observed suffix and average the scores.

    Args:
        decoding: The forecasts scored.
        traces: The tokens of every test case, keyed by its case id.
        metrics: The metrics, keyed by their name. Each must pickle, to reach the worker
            processes.
        workers: Processes scoring the forecasts in parallel.

    Returns:
        dict[str, float]: The mean score of the forecasts under every metric, keyed like
            `metrics`.
    """
    forecasts = decoding.forecasts
    draws = [forecast.draws for forecast in forecasts]
    observed = [traces[forecast.case_id][forecast.prefix_length :] for forecast in forecasts]
    chunksize = max(1, len(forecasts) // (16 * workers))
    score = partial(_scores, metrics=list(metrics.values()))
    # One torch thread per process, so the processes do not oversubscribe the cores.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=torch.set_num_threads, initargs=(1,)
    ) as pool:
        scores = np.array(list(pool.map(score, draws, observed, chunksize=chunksize)))  # [N, M]
    return dict(zip(metrics, scores.mean(axis=0).tolist(), strict=True))


def _scores(
    draws: Sequence[tuple[int, ...]], observed: tuple[int, ...], *, metrics: Sequence[Metric]
) -> list[float]:
    return [metric(draws=draws, observed=observed) for metric in metrics]


def save_scores(means: Mapping[str, Mapping[str, float]], path: Path) -> None:
    """Write the mean scores of every decoding as CSV.

    Args:
        means: The mean score of every decoding under every metric, keyed by decoding name and
            then by metric name.
        path: The CSV file written. Creates the parent directories if they do not exist.
    """
    table = pd.DataFrame.from_dict(means, orient='index').rename_axis(_DECODING_COLUMN)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)


def load_scores(path: Path) -> pd.DataFrame:
    """Read the mean scores of every decoding from a CSV file.

    Args:
        path: The CSV file read.

    Returns:
        pd.DataFrame: One row per decoding, indexed by its name, one column per metric.
    """
    return pd.read_csv(path, index_col=_DECODING_COLUMN)
