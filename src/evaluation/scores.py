from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.evaluation.distances import DISTANCES
from src.inference.generate import (
    ACTIVITIES_COLUMN,
    CASE_COLUMN,
    DRAW_COLUMN,
    PREFIX_LENGTH_COLUMN,
)
from src.logs.keys import ACTIVITY_KEY, CASE_KEY

# The first character an activity is encoded as, in the Unicode private use area.
_FIRST_CHARACTER = 0xE000


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
        generations: A file written by `src.inference.generate.generate`.
        test: The test split, read by `read_split`.

    Returns:
        One row per prefix: its case, its length, and the `prefix_scores` columns.
    """
    table = pq.read_table(generations).unify_dictionaries().combine_chunks()
    activities = table[ACTIVITIES_COLUMN].chunk(0)
    names = activities.values.dictionary.to_pylist()
    characters = _characters([*names, *test[ACTIVITY_KEY].unique()])
    table_characters = np.array([characters[name] for name in names], dtype=object)
    drawn = table_characters[activities.values.indices.to_numpy(zero_copy_only=False)]
    offsets = activities.offsets.to_numpy()
    suffixes = [''.join(drawn[a:b]) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]

    traces = test[ACTIVITY_KEY].map(characters).groupby(test[CASE_KEY], sort=False).agg(''.join)
    cases = table[CASE_COLUMN].to_numpy(zero_copy_only=False)
    lengths = table[PREFIX_LENGTH_COLUMN].to_numpy()
    starts = np.flatnonzero(pc.equal(table[DRAW_COLUMN], 0).to_numpy(zero_copy_only=False))
    stops = np.append(starts[1:], len(table))
    rows = []
    for start, stop in zip(starts, stops, strict=True):
        case, length = cases[start], int(lengths[start])
        observed = traces[case][length:]
        rows.append(
            {
                CASE_COLUMN: case,
                PREFIX_LENGTH_COLUMN: length,
                **prefix_scores(suffixes[start:stop], observed),
            }
        )
    return pd.DataFrame(rows)


def _characters(names: list[str]) -> dict[str, str]:
    """One distinct character per activity name, in order of first appearance."""
    unique = dict.fromkeys(names)
    return {name: chr(_FIRST_CHARACTER + i) for i, name in enumerate(unique)}
