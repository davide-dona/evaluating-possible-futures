from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from possible_futures.preprocessing.cases import Case


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


def drop_outliers(
    cases: Sequence[Case], *, max_length: int, max_duration_days: float
) -> list[Case]:
    """Drop outliers cases, based on length and duration constraints.

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
