from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# One row per case.
CASES_SCHEMA = pa.schema(
    [
        ('case_id', pa.string()),
        ('activities', pa.list_(pa.string())),
        ('timestamps', pa.list_(pa.timestamp('ns', tz='UTC'))),
    ]
)


@dataclass(frozen=True, slots=True)
class Case:
    """The events of one case, in time order."""

    case_id: str
    activities: tuple[str, ...]
    timestamps: tuple[pd.Timestamp, ...]  # UTC

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


def save_cases(cases: Sequence[Case], path: Path) -> None:
    """Write the cases in `CASES_SCHEMA`, creating the directory if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(cases_table(cases), path)


def load_cases(path: Path) -> list[Case]:
    return cases_from_table(pq.read_table(path, columns=CASES_SCHEMA.names))


def cases_table(cases: Sequence[Case]) -> pa.Table:
    """The cases as a table in `CASES_SCHEMA`."""
    return pa.table(
        {
            'case_id': [case.case_id for case in cases],
            'activities': [list(case.activities) for case in cases],
            'timestamps': [list(case.timestamps) for case in cases],
        },
        schema=CASES_SCHEMA,
    )


def cases_from_table(table: pa.Table) -> list[Case]:
    """The cases of a table holding the columns of `CASES_SCHEMA`."""
    table = table.select(CASES_SCHEMA.names).cast(CASES_SCHEMA)
    timestamps = [
        tuple(pd.Timestamp(value, tz='UTC') for value in case)
        for case in table['timestamps'].to_numpy(zero_copy_only=False)
    ]
    return [
        Case(case_id=case_id, activities=tuple(activities), timestamps=times)
        for case_id, activities, times in zip(
            table['case_id'].to_pylist(), table['activities'].to_pylist(), timestamps, strict=True
        )
    ]
