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
    """One run of the process: its activities in time order, and the prefixes taken from it.

    Attributes:
        case_id: The identifier of the case in its log.
        activities: The activity of every event, in time order.
        timestamps: The UTC time of every event, in the order of `activities`.
        prefix_lengths: The lengths of the prefixes of the case that are trained on or scored.
    """

    case_id: str
    activities: tuple[str, ...]
    timestamps: tuple[pd.Timestamp, ...]
    prefix_lengths: range

    def __len__(self) -> int:
        return len(self.activities)

    @property
    def start(self) -> pd.Timestamp:
        """The time of the first event.

        Returns:
            pd.Timestamp: The UTC timestamp of the first event.
        """
        return self.timestamps[0]

    @property
    def end(self) -> pd.Timestamp:
        """The time of the last event.

        Returns:
            pd.Timestamp: The UTC timestamp of the last event.
        """
        return self.timestamps[-1]

    @property
    def duration_days(self) -> float:
        """The time from the first to the last event.

        Returns:
            float: The duration of the case in days.
        """
        return (self.end - self.start).total_seconds() / 86_400.0

    def prefix(self, length: int) -> tuple[str, ...]:
        """The first `length` activities of the case.

        Args:
            length: The length of the prefix.

        Returns:
            tuple[str, ...]: The activities of the prefix.
        """
        return self.activities[:length]

    def suffix(self, length: int) -> tuple[str, ...]:
        """The activities following the prefix of `length` activities.

        Args:
            length: The length of the prefix the suffix follows.

        Returns:
            tuple[str, ...]: The activities of the suffix.
        """
        return self.activities[length:]


def num_variants(cases: Sequence[Case]) -> int:
    """The number of distinct variants within the cases.

    Args:
        cases: The cases counted.

    Returns:
        int: The number of distinct variants.
    """
    return len({case.activities for case in cases})


def num_events(cases: Sequence[Case]) -> int:
    """The number of events within the cases.

    Args:
        cases: The cases counted.

    Returns:
        int: The number of events.
    """
    return sum(len(case) for case in cases)


def num_activities(cases: Sequence[Case]) -> int:
    """The number of distinct activities within the cases.

    Args:
        cases: The cases counted.

    Returns:
        int: The number of distinct activities.
    """
    return len({activity for case in cases for activity in case.activities})


def num_prefixes(cases: Sequence[Case]) -> int:
    """The number of prefixes taken from the cases.

    Args:
        cases: The cases counted.

    Returns:
        int: The number of prefixes.
    """
    return sum(len(case.prefix_lengths) for case in cases)


def save_cases(cases: Sequence[Case], path: Path) -> None:
    """Write the cases to a Parquet file.

    Args:
        cases: The cases to write.
        path: The Parquet file written. Creates the parent directories if they do not exist.
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
    """Read the cases from a Parquet file.

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
