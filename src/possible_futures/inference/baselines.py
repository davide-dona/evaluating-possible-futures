from collections.abc import Callable
from pathlib import Path

import numpy as np

from possible_futures.datasets.encoded_log import EncodedLog
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.draws import Decoding, DecodingWriter, PrefixDraws
from possible_futures.processes.decoding import Process

# The most draws held in memory at once while writing the unconditional draws.
_ROWS_PER_TABLE = 100_000


def write_sampled_process_draws(
    process: Process,
    test_log: EncodedLog,
    vocabulary: Vocabulary,
    *,
    path: Path,
    num_draws: int,
    max_steps: int,
    rng: np.random.Generator,
) -> int:
    """Sample `num_draws` suffixes for every test prefix from a process and save them to `path`.

    Args:
        process: The process the suffixes are drawn from.
        test_log: The test split.
        vocabulary: The vocabulary the splits are encoded with, which holds every activity the
            process emits.
        path: The Parquet file to write.
        num_draws: Draws per prefix.
        max_steps: The most events a draw holds.
        rng: Source of the draws.

    Returns:
        The number of prefixes written.
    """
    return _write_process_draws(
        lambda prefix: process.sample_suffix(prefix, rng=rng, max_steps=max_steps),
        test_log=test_log,
        vocabulary=vocabulary,
        path=path,
        num_draws=num_draws,
    )


def write_greedy_process_draws(
    process: Process,
    test_log: EncodedLog,
    vocabulary: Vocabulary,
    *,
    path: Path,
    max_steps: int,
) -> int:
    """Save to `path` the suffix of every test prefix that takes the most probable activity of
    a process at every step.

    Args:
        process: The process the suffixes are decoded from.
        test_log: The test split.
        vocabulary: The vocabulary the splits are encoded with, which holds every activity the
            process emits.
        path: The Parquet file to write.
        max_steps: The most events a suffix holds.

    Returns:
        The number of prefixes written.
    """
    return _write_process_draws(
        lambda prefix: process.greedy_suffix(prefix, max_steps=max_steps),
        test_log=test_log,
        vocabulary=vocabulary,
        path=path,
        num_draws=1,
    )


def _write_process_draws(
    decode: Callable[[tuple[str, ...]], tuple[str, ...]],
    *,
    test_log: EncodedLog,
    vocabulary: Vocabulary,
    path: Path,
    num_draws: int,
) -> int:
    """Continue every test prefix `num_draws` times with `decode`, which maps a prefix of
    activity names to a suffix of them, and save the draws to `path`."""
    cuts = test_log.cuts()
    prefixes = []
    for case_index, prefix_length in zip(cuts.case_indices, cuts.prefix_lengths, strict=True):
        start = test_log.case_starts[case_index]
        events = test_log.activities[start : start + prefix_length]
        prefix = tuple(vocabulary.names[i] for i in events)
        draws = [tuple(vocabulary.encode(decode(prefix)).tolist()) for _ in range(num_draws)]
        prefixes.append(
            PrefixDraws(
                case_id=test_log.case_ids[case_index],
                prefix_length=int(prefix_length),
                draws=draws,
            )
        )
    Decoding(prefixes=prefixes).save(path, vocabulary)
    return len(prefixes)


def write_unconditional_draws(
    train_log: EncodedLog,
    test_log: EncodedLog,
    vocabulary: Vocabulary,
    *,
    path: Path,
    num_draws: int,
    rng: np.random.Generator,
) -> int:
    """Save to `path` draws that ignore the prefix: for every test prefix, `num_draws` suffixes
    drawn uniformly from the cuts of the train split.

    Every prefix of train counts once, as every prefix of test does in a mean score, so the
    draws follow the suffix distribution of train over all its prefixes.

    Args:
        train_log: The train split, whose suffixes are drawn.
        test_log: The test split.
        vocabulary: The vocabulary both splits are encoded with.
        path: The Parquet file to write.
        num_draws: Draws per prefix.
        rng: Source of the draws.

    Returns:
        The number of prefixes written.
    """
    sources = train_log.cuts()
    cuts = test_log.cuts()
    num_prefixes = len(cuts.case_indices)
    batch_size = max(1, _ROWS_PER_TABLE // num_draws)
    with DecodingWriter(path, vocabulary) as writer:
        for first in range(0, num_prefixes, batch_size):
            selected = cuts.select(np.arange(first, min(first + batch_size, num_prefixes)))
            rows = len(selected.case_indices) * num_draws
            drawn = sources.select(rng.integers(len(sources.case_indices), size=rows))
            writer.write(
                case_ids=test_log.case_ids[selected.case_indices],
                prefix_lengths=selected.prefix_lengths,
                activities=train_log.suffixes(drawn),
                num_draws=num_draws,
            )
    return num_prefixes
