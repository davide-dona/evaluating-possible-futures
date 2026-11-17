from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# One row per case.
CASES_SCHEMA = pa.schema([('case_id', pa.string()), ('activities', pa.list_(pa.string()))])


@dataclass(frozen=True, slots=True)
class Cases:
    """The cases of a split."""

    case_ids: np.ndarray  # object, [C]
    traces: tuple[tuple[str, ...], ...]  # the activities of every case, in time order
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
        )

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
            {'case_id': self.case_ids, 'activities': [list(trace) for trace in self.traces]},
            schema=CASES_SCHEMA,
        )
        pq.write_table(table, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        table = pq.read_table(path, columns=CASES_SCHEMA.names).cast(CASES_SCHEMA)
        return cls(
            case_ids=table['case_id'].to_numpy().astype(object),
            traces=tuple(tuple(trace) for trace in table['activities'].to_pylist()),
        )
