from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from possible_futures.logs.keys import ACTIVITY_KEY, CASE_KEY, MIN_PREFIX_KEY, TIMESTAMP_KEY


def read_original_log(path: Path, data_config: DictConfig) -> pd.DataFrame:
    """Read the case, activity, and timestamp of every event of a raw log.

    Args:
        path: The raw CSV log.
        data_config: The `data` section, naming the separator and the raw columns holding the
            case, the activity, and the timestamp.

    Returns:
        One row per event, with the canonical column names and the timestamp parsed.
    """
    columns = {
        data_config.case_key: CASE_KEY,
        data_config.activity_key: ACTIVITY_KEY,
        data_config.timestamp_key: TIMESTAMP_KEY,
    }
    log = pd.read_csv(
        path,
        sep=data_config.separator,
        usecols=list(columns),
        dtype={data_config.case_key: str, data_config.activity_key: str},
    ).rename(columns=columns)
    log[TIMESTAMP_KEY] = pd.to_datetime(log[TIMESTAMP_KEY], format='mixed')
    return log[list(columns.values())]


def write_split(split: pd.DataFrame, path: Path) -> None:
    """Write a preprocessed split, creating its directory if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)


def read_split(path: Path) -> pd.DataFrame:
    """Read a split written by `write_split`: case, activity, and lowest cut of every event, in
    log order."""
    return pd.read_csv(
        path,
        usecols=[CASE_KEY, ACTIVITY_KEY, MIN_PREFIX_KEY],
        dtype={CASE_KEY: str, ACTIVITY_KEY: str},
    )
