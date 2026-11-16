from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
import torch

from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.evaluation.metrics import Metric
from possible_futures.inference.draws import CASE_ID_COLUMN, PREFIX_LENGTH_COLUMN, PrefixDraws
from possible_futures.logs.keys import ACTIVITY_KEY, CASE_KEY


def observed_suffixes(
    prefixes: Sequence[PrefixDraws], test: pd.DataFrame, vocabulary: Vocabulary
) -> list[tuple[int, ...]]:
    """The observed suffix of every prefix, as vocabulary indices.

    Args:
        prefixes: The prefixes, cut from the test split.
        test: The test split, read by `read_split`.
        vocabulary: The vocabulary the draws were written with.
    """
    activities = pd.Series(vocabulary.encode(test[ACTIVITY_KEY]), index=test.index)
    traces = activities.groupby(test[CASE_KEY], sort=False).agg(tuple)
    return [traces[prefix.case_id][prefix.prefix_length :] for prefix in prefixes]


def score_prefixes(
    prefixes: Sequence[PrefixDraws],
    observed: Sequence[tuple[int, ...]],
    *,
    metrics: Mapping[str, Metric],
    workers: int,
) -> pd.DataFrame:
    """Score the draws of every prefix against its observed suffix under every metric.

    Args:
        prefixes: The draws of every prefix.
        observed: The observed suffix of every prefix, by `observed_suffixes`.
        metrics: The metrics, keyed by the column they are written to. Each must pickle, to
            reach the worker processes.
        workers: Processes scoring the prefixes in parallel.

    Returns:
        One row per prefix: its case id, its length, and its score under every metric.
    """
    scores = pd.DataFrame(
        {
            CASE_ID_COLUMN: [prefix.case_id for prefix in prefixes],
            PREFIX_LENGTH_COLUMN: [prefix.prefix_length for prefix in prefixes],
        }
    )
    draws = [prefix.draws for prefix in prefixes]
    chunksize = max(1, len(prefixes) // (16 * workers))
    # One torch thread per process, so the processes do not oversubscribe the cores.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=torch.set_num_threads, initargs=(1,)
    ) as pool:
        for name, metric in metrics.items():
            score = partial(_score, metric=metric)
            scores[name] = list(pool.map(score, draws, observed, chunksize=chunksize))
    return scores


def _score(draws: Sequence[tuple[int, ...]], observed: tuple[int, ...], *, metric: Metric) -> float:
    return metric(draws=draws, observed=observed)


@dataclass(frozen=True, slots=True)
class MeanScores:
    """The mean score of every decoding under every metric."""

    INDEX: ClassVar[str] = 'decoding'

    table: pd.DataFrame  # one row per decoding, indexed by its name, one column per metric

    def save(self, path: Path) -> None:
        """Write the scores as CSV, creating the directory if missing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.rename_axis(self.INDEX).to_csv(path)

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(table=pd.read_csv(path, index_col=cls.INDEX))
