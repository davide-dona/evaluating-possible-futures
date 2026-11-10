from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.distances import Distance
from src.inference.generate import DRAW_COLUMN, SCHEMA, read_generations


def medoid(draws: Sequence[str], distance: Distance) -> int:
    """The index of the medoid of the draws: the draw with the lowest mean distance to all of
    them, so the one a forecast collapsed on a single draw should keep to stay closest to the
    rest.

    Equal draws are measured once and weighted by how many times they were drawn. Among draws
    tied for the lowest mean, the first in string order is taken, at its first occurrence.
    """
    unique, first, counts = np.unique(
        np.array(draws, dtype=object), return_index=True, return_counts=True
    )
    totals = distance(list(unique)) @ counts  # [U, U] @ [U] -> [U]
    return int(first[totals.argmin()])


def write_medoids(generations: Path, *, path: Path, distance: Distance) -> int:
    """Write the medoid of the draws of every prefix of a generations file, one row per prefix
    in the same format.

    Args:
        generations: A file in the `SCHEMA` of `src.inference.generate`.
        path: The Parquet file to write.
        distance: The distance between suffixes the medoid is taken under.

    Returns:
        The number of prefixes written.
    """
    draws = read_generations(generations)
    rows = [
        start + medoid(draws.suffixes[start:stop], distance) for start, stop in draws.prefixes()
    ]
    medoids = draws.table.take(pa.array(rows, type=pa.int64()))
    medoids = medoids.set_column(
        medoids.schema.get_field_index(DRAW_COLUMN),
        SCHEMA.field(DRAW_COLUMN),
        pa.array(np.zeros(len(rows), dtype=np.int32)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(medoids, path)
    return len(rows)
