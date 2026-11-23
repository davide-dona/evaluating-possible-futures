from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

Activities = tuple[str, ...]
"""The activities of a sequence of events, in time order."""

CASES_SCHEMA = pa.schema(
    [
        ('case_id', pa.string()),
        ('activities', pa.list_(pa.string())),
        ('timestamps', pa.list_(pa.timestamp('ns', tz='UTC'))),
    ]
)

SCORED_SCHEMA = pa.schema(
    [
        ('case_id', pa.string()),
        ('shortest_prefix_length', pa.int64()),
        ('longest_prefix_length', pa.int64()),
    ]
)

ScoredLengths = dict[str, range]
"""The lengths of the prefixes scored for each case, keyed by case id. A case with no scored
prefix is absent."""


@dataclass(frozen=True, slots=True)
class Case:
    """One run of the process: its activities and their times, in time order.

    Attributes:
        case_id: The identifier of the case in its log.
        activities: The activity of every event, in time order.
        timestamps: The UTC time of every event, in the order of `activities`.
    """

    case_id: str
    activities: Activities
    timestamps: tuple[pd.Timestamp, ...]

    def __len__(self) -> int:
        return len(self.activities)

    @property
    def prefix_lengths(self) -> range:
        """The lengths of every prefix of the case that leaves a non-empty suffix.

        Returns:
            range: The lengths, from 1 to one less than the length of the case.
        """
        return range(1, len(self))

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

    def prefix(self, length: int) -> Activities:
        """The first `length` activities of the case.

        Args:
            length: The length of the prefix.

        Returns:
            Activities: The activities of the prefix.
        """
        return self.activities[:length]

    def suffix(self, length: int) -> Activities:
        """The activities following the prefix of `length` activities.

        Args:
            length: The length of the prefix the suffix follows.

        Returns:
            Activities: The activities of the suffix.
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
    """The number of prefixes of the cases that leave a non-empty suffix.

    Args:
        cases: The cases counted.

    Returns:
        int: The number of prefixes.
    """
    return sum(len(case.prefix_lengths) for case in cases)


def num_scored_prefixes(scored: ScoredLengths) -> int:
    """The number of scored prefixes.

    Args:
        scored: The lengths of the scored prefixes of every case.

    Returns:
        int: The number of scored prefixes.
    """
    return sum(len(lengths) for lengths in scored.values())


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
        )
        for case_id, activities, times in zip(
            table['case_id'].to_pylist(),
            table['activities'].to_pylist(),
            timestamps,
            strict=True,
        )
    ]


def save_scored(scored: ScoredLengths, path: Path) -> None:
    """Write the lengths of the scored prefixes to a Parquet file.

    Args:
        scored: The lengths of the scored prefixes of every case.
        path: The Parquet file written. Creates the parent directories if they do not exist.
    """
    table = pa.table(
        {
            'case_id': list(scored),
            'shortest_prefix_length': [lengths.start for lengths in scored.values()],
            'longest_prefix_length': [lengths.stop - 1 for lengths in scored.values()],
        },
        schema=SCORED_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def load_scored(path: Path) -> ScoredLengths:
    """Read the lengths of the scored prefixes from a Parquet file.

    Args:
        path: The Parquet file read.

    Returns:
        ScoredLengths: The lengths of every case, in the order they were written.
    """
    table = pq.read_table(path).cast(SCORED_SCHEMA)
    return {
        case_id: range(shortest, longest + 1)
        for case_id, shortest, longest in zip(
            table['case_id'].to_pylist(),
            table['shortest_prefix_length'].to_pylist(),
            table['longest_prefix_length'].to_pylist(),
            strict=True,
        )
    }
