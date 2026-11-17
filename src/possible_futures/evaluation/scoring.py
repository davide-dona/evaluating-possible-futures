from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
import torch

from possible_futures.datasets.cases import Cases
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.evaluation.metrics import Metric
from possible_futures.inference.draws import PrefixDraws


def observed_suffixes(
    prefixes: Sequence[PrefixDraws], test: Cases, vocabulary: Vocabulary
) -> list[tuple[int, ...]]:
    """The observed suffix of every prefix in the test cases, as tokens."""
    traces = {
        case_id: tuple(vocabulary.encode(trace).tolist())
        for case_id, trace in zip(test.case_ids, test.traces, strict=True)
    }
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
        metrics: The metrics, keyed by the column they are written to. Each must pickle, to
            reach the worker processes.
        workers: Processes scoring the prefixes in parallel.

    Returns:
        One row per prefix, in the order of `prefixes`, with its score under every metric.
    """
    scores = pd.DataFrame(index=range(len(prefixes)))
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
