from pathlib import Path

import pandas as pd

from possible_futures.logs.keys import ACTIVITY_KEY, CASE_KEY, TIMESTAMP_KEY


def read_original_log(
    path: Path, *, case_key: str, activity_key: str, timestamp_key: str, separator: str
) -> pd.DataFrame:
    """Read the case, activity, and timestamp of every event of a raw log.

    Args:
        path: The raw CSV log.
        case_key: The raw column holding the case.
        activity_key: The raw column holding the activity.
        timestamp_key: The raw column holding the timestamp.
        separator: The field separator.

    Returns:
        One row per event, with the canonical column names and the timestamp parsed.
    """
    columns = {case_key: CASE_KEY, activity_key: ACTIVITY_KEY, timestamp_key: TIMESTAMP_KEY}
    log = pd.read_csv(
        path, sep=separator, usecols=list(columns), dtype={case_key: str, activity_key: str}
    ).rename(columns=columns)
    log[TIMESTAMP_KEY] = pd.to_datetime(log[TIMESTAMP_KEY], format='mixed')
    return log[list(columns.values())]


def write_split(split: pd.DataFrame, path: Path) -> None:
    """Write a preprocessed split, creating its directory if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)


def read_split(path: Path) -> pd.DataFrame:
    """Read a split written by `write_split`: case, activity, timestamp, and lowest and highest
    cut of every event, in log order."""
    return pd.read_csv(
        path,
        dtype={CASE_KEY: str, ACTIVITY_KEY: str},
        parse_dates=[TIMESTAMP_KEY],
        date_format='ISO8601',
    )


def write_max_case_length(max_case_length: int, path: Path) -> None:
    """Write the length of the longest case preprocessing keeps."""
    path.write_text(f'{max_case_length}\n')


def read_max_case_length(path: Path) -> int:
    """Read the length written by `write_max_case_length`."""
    return int(path.read_text())
