from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from possible_futures.datasets.cases import Case


def read_log(
    path: Path, *, case_key: str, activity_key: str, timestamp_key: str, separator: str
) -> list[Case]:
    """Read a CSV log file and return the cases in time order.

    Args:
        path: The path to the CSV file.
        case_key: The name of the column containing the case IDs.
        activity_key: The name of the column containing the activity names.
        timestamp_key: The name of the column containing the timestamps.
        separator: The separator used in the CSV file.

    Returns:
        A list of `Case` objects, each representing a case with its activities and timestamps,
        and every prefix that leaves a non-empty suffix.
    """
    events = pd.read_csv(
        path,
        sep=separator,
        usecols=[case_key, activity_key, timestamp_key],
        dtype={case_key: str, activity_key: str},
    )
    events[timestamp_key] = pd.to_datetime(events[timestamp_key], format='mixed', utc=True)
    events = events.sort_values(by=timestamp_key, kind='stable')
    cases = events.groupby(case_key, sort=False)[[activity_key, timestamp_key]].agg(tuple)
    return [
        Case(
            case_id=case_id,
            activities=activities,
            timestamps=timestamps,
            prefix_lengths=range(1, len(activities)),
        )
        for case_id, activities, timestamps in zip(
            cases.index, cases[activity_key], cases[timestamp_key], strict=True
        )
    ]


def write_log(
    cases: Sequence[Case],
    path: Path,
    *,
    case_key: str,
    activity_key: str,
    timestamp_key: str,
    separator: str,
) -> None:
    """Write the events of the cases to a CSV log file, one row per event, case after case.

    Args:
        cases: The cases written.
        path: The CSV file written. Creates the parent directories if they do not exist.
        case_key: The name of the column of the case IDs.
        activity_key: The name of the column of the activity names.
        timestamp_key: The name of the column of the timestamps.
        separator: The separator of the CSV file.
    """
    events = pd.DataFrame(
        {
            case_key: [case.case_id for case in cases for _ in case.activities],
            activity_key: [activity for case in cases for activity in case.activities],
            timestamp_key: [timestamp for case in cases for timestamp in case.timestamps],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(path, sep=separator, index=False)


def drop_outliers(
    cases: Sequence[Case], *, max_length: int, max_duration_days: float
) -> list[Case]:
    """Drop the outlier cases: the ones too long, in events or in time.

    Args:
        cases: The list of cases to filter.
        max_length: The maximum length of a case.
        max_duration_days: The maximum duration of a case in days.

    Returns:
        A list of `Case` objects, filtered to exclude outliers.
    """
    return [
        case
        for case in cases
        if len(case) <= max_length and case.duration_days <= max_duration_days
    ]
