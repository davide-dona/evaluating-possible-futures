from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.datasets.dataset import EncodedLog
from src.datasets.vocabulary import PAD, Vocabulary
from src.inference.generate import SCHEMA, suffix_table
from src.simulation.processes import Process

# The most draws held in memory at once while writing the unconditional forecast.
_ROWS_PER_TABLE = 100_000


def write_baseline(
    source: Process,
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
        sampled: Whether to draw `num_samples` suffixes by `Process.sample_suffix`, or one
            by `Process.greedy_suffix`.
        num_samples: Draws per prefix when sampled.
        rng: Source of the sampled draws.

    Returns:
        The number of prefixes written.
    """
    draws = num_samples if sampled else 1
    max_steps = vocabulary.max_case_length
    index = {name: i for i, name in enumerate(vocabulary.names)}
    cuts = log.cuts(from_min_prefix=True)
    activities = np.full((len(cuts.case) * draws, max_steps), PAD, dtype=np.int64)
    truncated = np.zeros(len(cuts.case) * draws, dtype=bool)
    row = 0
    for case, length in zip(cuts.case, cuts.length, strict=True):
        start = log.starts[case]
        prefix = tuple(vocabulary.names[i] for i in log.activities[start : start + length])
        for _ in range(draws):
            if sampled:
                suffix, truncated[row] = source.sample_suffix(prefix, rng, max_steps=max_steps)
            else:
                suffix, truncated[row] = source.greedy_suffix(prefix, max_steps=max_steps)
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


def write_unconditional(
    train: EncodedLog,
    log: EncodedLog,
    vocabulary: Vocabulary,
    *,
    path: Path,
    num_samples: int,
    rng: np.random.Generator,
) -> int:
    """Write a forecast that ignores the prefix: for every scored test prefix, `num_samples`
    suffixes drawn uniformly from the cuts of the train split, in the format of
    `src.inference.generate.generate`.

    Every prefix of train counts once, as every prefix of test does in a mean score, so the
    draws follow the suffix distribution of train over all its prefixes.

    Args:
        train: The train split, whose suffixes are drawn.
        log: The test split.
        vocabulary: The vocabulary both splits are encoded with.
        path: The Parquet file to write.
        num_samples: Draws per prefix.
        rng: Source of the draws.

    Returns:
        The number of prefixes written.
    """
    sources = train.cuts(from_min_prefix=False)
    cuts = log.cuts(from_min_prefix=True)
    names = pa.array(vocabulary.names, type=pa.string())
    batch_size = max(1, _ROWS_PER_TABLE // num_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema=SCHEMA) as writer:
        for first in range(0, len(cuts.case), batch_size):
            selected = cuts.select(np.arange(first, min(first + batch_size, len(cuts.case))))
            rows = len(selected.case) * num_samples
            drawn = sources.select(rng.integers(len(sources.case), size=rows))
            table = suffix_table(
                log,
                selected,
                activities=train.suffixes(drawn),
                truncated=np.zeros(rows, dtype=bool),
                draws=num_samples,
                names=names,
            )
            writer.write_table(table)
    return len(cuts.case)
