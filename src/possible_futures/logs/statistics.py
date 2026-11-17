import pandas as pd

from possible_futures.logs.keys import (
    ACTIVITY_KEY,
    CASE_KEY,
    MAX_PREFIX_KEY,
    MIN_PREFIX_KEY,
    TIMESTAMP_KEY,
)


def case_durations_days(log: pd.DataFrame) -> pd.Series:
    """The duration of every case in days, first event to last, indexed by case."""
    edges = log.groupby(CASE_KEY)[TIMESTAMP_KEY].agg(['min', 'max'])
    return (edges['max'] - edges['min']).dt.total_seconds() / 86_400.0


def case_lengths(log: pd.DataFrame) -> pd.Series:
    """The length of every case in events, indexed by case."""
    return log.groupby(CASE_KEY).size()


def num_prefixes(split: pd.DataFrame) -> int:
    """The number of cuts of a split read by `read_split`, from the lowest to the highest cut of
    every case."""
    bounds = split.groupby(CASE_KEY)[[MIN_PREFIX_KEY, MAX_PREFIX_KEY]].first()
    return int((bounds[MAX_PREFIX_KEY] - bounds[MIN_PREFIX_KEY] + 1).clip(lower=0).sum())


def num_variants(log: pd.DataFrame) -> int:
    """The number of distinct activity sequences among the cases, each read in row order."""
    return log.groupby(CASE_KEY, sort=False)[ACTIVITY_KEY].agg(tuple).nunique()
