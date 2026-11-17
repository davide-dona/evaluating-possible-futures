from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# One row per case.
CASES_SCHEMA = pa.schema(
    [
        ('case', pa.string()),
        ('activities', pa.list_(pa.string())),
        ('min_prefix_length', pa.int64()),
        ('max_prefix_length', pa.int64()),
    ]
)


class Cuts(NamedTuple):
    """Prefixes of some cases, each as the case it is cut from and the events it holds."""

    case_indices: np.ndarray  # int64, [N]
    prefix_lengths: np.ndarray  # int64, [N]

    def select(self, index: np.ndarray) -> 'Cuts':
        return Cuts(
            case_indices=self.case_indices[index], prefix_lengths=self.prefix_lengths[index]
        )


@dataclass(frozen=True, slots=True)
class Cases:
    """The cases of a split and the range of prefix lengths each is cut at.

    A case whose lowest prefix length exceeds its highest is not cut at all.
    """

    case_ids: np.ndarray  # object, [C]
    traces: tuple[tuple[str, ...], ...]  # the activities of every case, in time order
    min_prefix_lengths: np.ndarray  # int64, [C]
    max_prefix_lengths: np.ndarray  # int64, [C]
    case_lengths: np.ndarray = field(init=False, repr=False, compare=False)  # int64, [C]

    def __post_init__(self) -> None:
        lengths = np.fromiter((len(trace) for trace in self.traces), dtype=np.int64)
        object.__setattr__(self, 'case_lengths', lengths)

    def __len__(self) -> int:
        return len(self.traces)

    @classmethod
    def concat(cls, parts: list[Self]) -> Self:
        return cls(
            case_ids=np.concatenate([part.case_ids for part in parts]),
            traces=tuple(chain.from_iterable(part.traces for part in parts)),
            min_prefix_lengths=np.concatenate([part.min_prefix_lengths for part in parts]),
            max_prefix_lengths=np.concatenate([part.max_prefix_lengths for part in parts]),
        )

    def cuts(self) -> Cuts:
        """Every cut of every case, from its lowest to its highest prefix length."""
        lengths = [
            np.arange(lowest, highest + 1)
            for lowest, highest in zip(
                self.min_prefix_lengths, self.max_prefix_lengths, strict=True
            )
        ]
        return Cuts(
            case_indices=np.repeat(np.arange(len(lengths)), [len(each) for each in lengths]),
            prefix_lengths=np.concatenate(lengths),
        )

    def num_prefixes(self) -> int:
        return len(self.cuts().prefix_lengths)

    def num_variants(self) -> int:
        """The number of distinct traces."""
        return len(set(self.traces))

    def num_activities(self) -> int:
        """The number of distinct activities."""
        return len(set(chain.from_iterable(self.traces)))

    def save(self, path: Path) -> None:
        """Write the cases in `CASES_SCHEMA`, creating the directory if missing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                'case': self.case_ids,
                'activities': [list(trace) for trace in self.traces],
                'min_prefix_length': self.min_prefix_lengths,
                'max_prefix_length': self.max_prefix_lengths,
            },
            schema=CASES_SCHEMA,
        )
        pq.write_table(table, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        table = pq.read_table(path, schema=CASES_SCHEMA)
        return cls(
            case_ids=table['case'].to_numpy().astype(object),
            traces=tuple(tuple(trace) for trace in table['activities'].to_pylist()),
            min_prefix_lengths=table['min_prefix_length'].to_numpy(),
            max_prefix_lengths=table['max_prefix_length'].to_numpy(),
        )


class TrainTestSplit(NamedTuple):
    train: Cases
    test: Cases
