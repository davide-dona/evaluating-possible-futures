from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from possible_futures.datasets.cases import CASES_SCHEMA, Case, cases_from_table, cases_table

# One row per case: the case, then the lengths of its shortest and longest scored prefix. The
# scored prefixes of a case are every prefix between the two, both null when it has none.
SCORED_CASES_SCHEMA = pa.schema(
    [
        *CASES_SCHEMA,
        ('shortest_scored_length', pa.int64()),
        ('longest_scored_length', pa.int64()),
    ]
)


@dataclass(frozen=True, slots=True)
class Prefix:
    """The first `length` events of a case."""

    case: Case
    length: int

    @property
    def activities(self) -> tuple[str, ...]:
        return self.case.activities[: self.length]

    @property
    def suffix(self) -> tuple[str, ...]:
        """The activities following the prefix."""
        return self.case.activities[self.length :]

    @property
    def end(self) -> pd.Timestamp:
        """The time of the last event of the prefix."""
        return self.case.timestamps[self.length - 1]


def every_prefix(cases: Sequence[Case]) -> list[Prefix]:
    """Every prefix of every case that leaves a non-empty suffix, shortest first within a case."""
    return [Prefix(case=case, length=length) for case in cases for length in range(1, len(case))]


def save_scored_cases(cases: Sequence[Case], scored: Sequence[Prefix], path: Path) -> None:
    """Write the cases with the range of lengths of their `scored` prefixes, in
    `SCORED_CASES_SCHEMA`, creating the directory if missing.

    The scored prefixes of a case must have consecutive lengths.
    """
    lengths: dict[str, list[int]] = {}
    for prefix in scored:
        lengths.setdefault(prefix.case.case_id, []).append(prefix.length)
    shortest = [min(lengths[case.case_id]) if case.case_id in lengths else None for case in cases]
    longest = [max(lengths[case.case_id]) if case.case_id in lengths else None for case in cases]
    table = (
        cases_table(cases)
        .append_column('shortest_scored_length', pa.array(shortest, type=pa.int64()))
        .append_column('longest_scored_length', pa.array(longest, type=pa.int64()))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def load_scored_prefixes(path: Path) -> list[Prefix]:
    """The scored prefixes of the cases saved at `path`, shortest first within a case."""
    table = pq.read_table(path, columns=SCORED_CASES_SCHEMA.names).cast(SCORED_CASES_SCHEMA)
    return [
        Prefix(case=case, length=length)
        for case, shortest, longest in zip(
            cases_from_table(table),
            table['shortest_scored_length'].to_pylist(),
            table['longest_scored_length'].to_pylist(),
            strict=True,
        )
        if shortest is not None
        for length in range(shortest, longest + 1)
    ]
