from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from possible_futures.evaluation.distances import DISTANCES
from possible_futures.inference.generations import (
    CASE_COLUMN,
    PREFIX_LENGTH_COLUMN,
    read_generations,
)
from possible_futures.logs.keys import ACTIVITY_KEY, CASE_KEY


def prefix_scores(draws: list[str], observed: str) -> dict[str, float]:
    """The accuracy and spread of the draws of one prefix under every distance.

    The accuracy is `1/K sum_i d(x_i, y)` and the spread `1/(2 K (K - 1)) sum_{i != j}
    d(x_i, x_j)`, 0 for a single draw, so the fair energy score is accuracy minus spread. Equal
    draws are measured once and weighted by how many times they were drawn.

    Args:
        draws: The `K` drawn suffixes, one character per activity.
        observed: The observed suffix.

    Returns:
        `<distance>_accuracy` and `<distance>_spread` for every distance.
    """
    unique, counts = np.unique(np.array(draws, dtype=object), return_counts=True)
    weights = counts.astype(np.float64)
    total = weights.sum()
    scores = {}
    for name, distance in DISTANCES.items():
        matrix = distance([*unique, observed])  # [U + 1, U + 1]
        scores[f'{name}_accuracy'] = float(weights @ matrix[:-1, -1]) / total
        pairs = float(weights @ matrix[:-1, :-1] @ weights)
        scores[f'{name}_spread'] = 0.0 if total < 2 else pairs / (2.0 * total * (total - 1.0))
    return scores


def score_generations(generations: Path, test: pd.DataFrame, *, workers: int) -> dict[str, float]:
    """Score a generations file against the observed suffixes, every prefix weighed equally.

    Args:
        generations: A file in the `SCHEMA` of `possible_futures.inference.generations`.
        test: The test split, read by `read_split`.
        workers: Processes scoring the prefixes in parallel.

    Returns:
        The mean DLS, as `DLS`, and the fair energy score under every distance, as
        `ES <distance>`.
    """
    draws = read_generations(generations, other_names=test[ACTIVITY_KEY].unique())
    activities = test[ACTIVITY_KEY].map(draws.characters)
    traces = activities.groupby(test[CASE_KEY], sort=False).agg(''.join)
    cases = draws.table[CASE_COLUMN].to_numpy(zero_copy_only=False)
    lengths = draws.table[PREFIX_LENGTH_COLUMN].to_numpy()
    bounds = list(draws.prefixes())
    drawn = [draws.suffixes[start:stop] for start, stop in bounds]
    observed = [traces[cases[start]][lengths[start] :] for start, _ in bounds]
    # One torch thread per process, so the processes do not oversubscribe the cores.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=torch.set_num_threads, initargs=(1,)
    ) as pool:
        scores = pool.map(
            prefix_scores, drawn, observed, chunksize=max(1, len(bounds) // (16 * workers))
        )
        means = pd.DataFrame(list(scores)).mean()
    return {'DLS': 1.0 - means['dl_accuracy']} | {
        f'ES {distance}': means[f'{distance}_accuracy'] - means[f'{distance}_spread']
        for distance in DISTANCES
    }
