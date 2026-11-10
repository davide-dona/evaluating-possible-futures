from pathlib import Path

import numpy as np
import pandas as pd

from possible_futures.evaluation.distances import DISTANCES
from possible_futures.inference.generate import CASE_COLUMN, PREFIX_LENGTH_COLUMN, read_generations
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


def score_generations(generations: Path, test: pd.DataFrame) -> pd.DataFrame:
    """Score every prefix of a generations file against its observed suffix.

    Args:
        generations: A file in the `SCHEMA` of `possible_futures.inference.generate`.
        test: The test split, read by `read_split`.

    Returns:
        One row per prefix: its case, its length, and the `prefix_scores` columns.
    """
    draws = read_generations(generations, other_names=test[ACTIVITY_KEY].unique())
    activities = test[ACTIVITY_KEY].map(draws.characters)
    traces = activities.groupby(test[CASE_KEY], sort=False).agg(''.join)
    cases = draws.table[CASE_COLUMN].to_numpy(zero_copy_only=False)
    lengths = draws.table[PREFIX_LENGTH_COLUMN].to_numpy()
    rows = []
    for start, stop in draws.prefixes():
        case, length = cases[start], int(lengths[start])
        observed = traces[case][length:]
        rows.append(
            {
                CASE_COLUMN: case,
                PREFIX_LENGTH_COLUMN: length,
                **prefix_scores(draws.suffixes[start:stop], observed),
            }
        )
    return pd.DataFrame(rows)


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """The mean DLS and fair energy scores of every decoding, one row each, every prefix weighed
    equally."""
    means = scores.groupby('decoding', sort=False).mean(numeric_only=True)
    columns = {'DLS': 1.0 - means['dl_accuracy']}
    for distance in DISTANCES:
        columns[f'ES {distance}'] = means[f'{distance}_accuracy'] - means[f'{distance}_spread']
    return pd.DataFrame(columns)
