from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.datasets.dataset import EncodedLog, every_cut
from src.datasets.vocabulary import PAD, Vocabulary
from src.inference.generate import suffix_table
from src.simulation.processes import NextActivities, greedy_suffix, sample_suffix


def write_baseline(
    source: NextActivities,
    log: EncodedLog,
    vocabulary: Vocabulary,
    *,
    path: Path,
    sampled: bool,
    num_samples: int,
    rng: np.random.Generator,
) -> int:
    """Write the suffixes a known process draws for every scored test prefix, one row per draw,
    in the format of `src.inference.generate.generate`.

    Args:
        source: The process continuing every prefix.
        log: The test split.
        vocabulary: The vocabulary of the trained model, which every activity of `source` is in.
        path: The Parquet file to write.
        sampled: Whether to draw `num_samples` suffixes by `sample_suffix`, or one by
            `greedy_suffix`.
        num_samples: Draws per prefix when sampled.
        rng: Source of the sampled draws.

    Returns:
        The number of prefixes written.
    """
    draws = num_samples if sampled else 1
    max_steps = vocabulary.max_case_length
    index = {name: i for i, name in enumerate(vocabulary.names)}
    cuts = every_cut(log, from_min_prefix=True)
    activities = np.full((len(cuts.case) * draws, max_steps), PAD, dtype=np.int64)
    truncated = np.zeros(len(cuts.case) * draws, dtype=bool)
    row = 0
    for case, length in zip(cuts.case, cuts.length, strict=True):
        start = log.starts[case]
        prefix = tuple(vocabulary.names[i] for i in log.activities[start : start + length])
        for _ in range(draws):
            if sampled:
                suffix, truncated[row] = sample_suffix(source, prefix, rng, max_steps=max_steps)
            else:
                suffix, truncated[row] = greedy_suffix(source, prefix, max_steps=max_steps)
            activities[row, : len(suffix)] = [index[name] for name in suffix]
            row += 1
    table = suffix_table(
        log,
        cuts,
        activities=activities,
        truncated=truncated,
        draws=draws,
        names=pa.array(vocabulary.names, type=pa.string()),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return len(cuts.case)
