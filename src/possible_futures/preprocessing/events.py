from pathlib import Path

import pandas as pd

CASE_COLUMN = 'case'
ACTIVITY_COLUMN = 'activity'
TIMESTAMP_COLUMN = 'timestamp'


def read_original_log(
    path: Path, *, case_key: str, activity_key: str, timestamp_key: str, separator: str
) -> pd.DataFrame:
    """Read the case, activity, and timestamp of every event of a raw CSV log."""
    columns = {
        case_key: CASE_COLUMN,
        activity_key: ACTIVITY_COLUMN,
        timestamp_key: TIMESTAMP_COLUMN,
    }
    events = pd.read_csv(
        path, sep=separator, usecols=list(columns), dtype={case_key: str, activity_key: str}
    ).rename(columns=columns)
    events[TIMESTAMP_COLUMN] = pd.to_datetime(events[TIMESTAMP_COLUMN], format='mixed')
    return events[list(columns.values())]


def num_cases(events: pd.DataFrame) -> int:
    return events[CASE_COLUMN].nunique()


def case_lengths(events: pd.DataFrame) -> pd.Series:
    """The number of events of every case, indexed by case."""
    return events.groupby(CASE_COLUMN).size()


def case_durations_days(events: pd.DataFrame) -> pd.Series:
    """The days from the first to the last event of every case, indexed by case."""
    edges = events.groupby(CASE_COLUMN)[TIMESTAMP_COLUMN].agg(['min', 'max'])
    return (edges['max'] - edges['min']).dt.total_seconds() / 86_400.0


def drop_outlier_cases(
    events: pd.DataFrame, *, max_duration_days: float, max_length: int
) -> pd.DataFrame:
    """Keep the cases lasting at most `max_duration_days` and at most `max_length` events long."""
    kept = (case_durations_days(events) <= max_duration_days) & (case_lengths(events) <= max_length)
    return events[events[CASE_COLUMN].isin(kept.index[kept])]
