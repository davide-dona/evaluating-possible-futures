from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CASES_SCHEMA = pa.schema(
    [
        ('case_id', pa.string()),
        ('activities', pa.list_(pa.string())),
        ('timestamps', pa.list_(pa.timestamp('ns', tz='UTC'))),
        ('shortest_prefix_length', pa.int64()),
        ('longest_prefix_length', pa.int64()),
    ]
)


@dataclass(frozen=True, slots=True)
class Case:
    """The events of one case, in time order, and the lengths of the prefixes taken from it."""

    case_id: str
    activities: tuple[str, ...]
    timestamps: tuple[pd.Timestamp, ...]  # UTC
    prefix_lengths: range

    def __len__(self) -> int:
        return len(self.activities)

    @property
    def start(self) -> pd.Timestamp:
        return self.timestamps[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.timestamps[-1]

    @property
    def duration_days(self) -> float:
        return (self.end - self.start).total_seconds() / 86_400.0


def num_variants(cases: Sequence[Case]) -> int:
    """The number of distinct activity sequences."""
    return len({case.activities for case in cases})


def num_activities(cases: Sequence[Case]) -> int:
    """The number of distinct activities."""
    return len({activity for case in cases for activity in case.activities})


def num_prefixes(cases: Sequence[Case]) -> int:
    """The number of prefixes taken from the cases."""
    return sum(len(case.prefix_lengths) for case in cases)


def save_cases(cases: Sequence[Case], path: Path) -> None:
    """Write the cases in `CASES_SCHEMA`, creating the directory if missing.

    Args:
        cases: The cases to write.
        path: The Parquet file written.
    """
    table = pa.table(
        {
            'case_id': [case.case_id for case in cases],
            'activities': [list(case.activities) for case in cases],
            'timestamps': [list(case.timestamps) for case in cases],
            'shortest_prefix_length': [case.prefix_lengths.start for case in cases],
            'longest_prefix_length': [case.prefix_lengths.stop - 1 for case in cases],
        },
        schema=CASES_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def load_cases(path: Path) -> list[Case]:
    """Read the cases written by `save_cases`.

    Args:
        path: The Parquet file read.

    Returns:
        list[Case]: The cases, in the order they were written.
    """
    table = pq.read_table(path, columns=CASES_SCHEMA.names).cast(CASES_SCHEMA)
    timestamps = [
        tuple(pd.Timestamp(value, tz='UTC') for value in case)
        for case in table['timestamps'].to_numpy(zero_copy_only=False)
    ]
    return [
        Case(
            case_id=case_id,
            activities=tuple(activities),
            timestamps=times,
            prefix_lengths=range(shortest, longest + 1),
        )
        for case_id, activities, times, shortest, longest in zip(
            table['case_id'].to_pylist(),
            table['activities'].to_pylist(),
            timestamps,
            table['shortest_prefix_length'].to_pylist(),
            table['longest_prefix_length'].to_pylist(),
            strict=True,
        )
    ]
