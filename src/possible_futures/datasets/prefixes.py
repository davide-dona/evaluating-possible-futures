from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from possible_futures.datasets.cases import Cases

# One row per prefix.
PREFIXES_SCHEMA = pa.schema([('case_id', pa.string()), ('prefix_length', pa.int64())])


@dataclass(frozen=True, slots=True)
class Prefixes:
    """Prefixes of some cases, each as the id of its case and the number of events it holds.

    The prefixes of a case are consecutive and shortest first.
    """

    case_ids: np.ndarray  # object, [N]
    lengths: np.ndarray  # int64, [N]

    def __len__(self) -> int:
        return len(self.lengths)

    @classmethod
    def within(cls, case_ids: np.ndarray, *, shortest: np.ndarray, longest: np.ndarray) -> Self:
        """Every prefix of every case from its `shortest` to its `longest` length, none for a case
        whose shortest exceeds its longest."""
        counts = np.maximum(longest - shortest + 1, 0)
        # The offset of every prefix within its case, restarting at 0 for each case.
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        return cls(
            case_ids=np.repeat(case_ids, counts), lengths=np.repeat(shortest, counts) + offsets
        )

    @classmethod
    def every(cls, cases: Cases) -> Self:
        """Every prefix of every case that leaves a non-empty suffix."""
        return cls.within(
            cases.case_ids,
            shortest=np.ones(len(cases), dtype=np.int64),
            longest=cases.case_lengths - 1,
        )

    def save(self, path: Path) -> None:
        """Write the prefixes in `PREFIXES_SCHEMA`, creating the directory if missing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {'case_id': self.case_ids, 'prefix_length': self.lengths}, schema=PREFIXES_SCHEMA
        )
        pq.write_table(table, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        table = pq.read_table(path, columns=PREFIXES_SCHEMA.names).cast(PREFIXES_SCHEMA)
        return cls(
            case_ids=table['case_id'].to_numpy().astype(object),
            lengths=table['prefix_length'].to_numpy(),
        )
