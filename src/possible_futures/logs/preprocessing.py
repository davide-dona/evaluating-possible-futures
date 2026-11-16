from typing import NamedTuple

import numpy as np
import pandas as pd

from possible_futures.logs.keys import CASE_KEY, MAX_PREFIX_KEY, MIN_PREFIX_KEY, TIMESTAMP_KEY
from possible_futures.logs.statistics import case_durations_days, case_lengths


class TrainTestSplit(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame


def drop_outlier_cases(
    log: pd.DataFrame, *, max_duration_days: float, max_length: int
) -> pd.DataFrame:
    """Keep the cases running at most `max_duration_days`, first event to last, and holding at
    most `max_length` events.

    A case spanning decades is a broken timestamp rather than a slow process.
    """
    kept = (case_durations_days(log) <= max_duration_days) & (case_lengths(log) <= max_length)
    return log[log[CASE_KEY].isin(kept.index[kept])]


def debias_log_end(log: pd.DataFrame, *, window_days: float) -> pd.DataFrame:
    """Drop every prefix ending in the last `window_days` of a log, the end-of-dataset debiasing
    of Weytjens and De Weerdt (2021).

    A log is a snapshot: a case still running when it was extracted ends at its last recorded
    event, which would then be scored as the true end of the trace. With `window_days` the
    longest a case lasts, only a case starting in the window can be one of them. Without those
    cases the window holds just the late events of older cases, so the prefixes of the kept
    cases that end in it go too. Selecting cases on start time rather than on how they end
    keeps the selection independent of the scored suffixes.

    Returns:
        The cases starting before the window, with `MAX_PREFIX_KEY` added: the highest cut
        that ends before the window and leaves a non-empty suffix.
    """
    window_start = log[TIMESTAMP_KEY].max() - pd.Timedelta(days=window_days)
    # A cut of k ends at the k-th event in time order, so the events before the window count
    # the cuts ending before it.
    before = (log[TIMESTAMP_KEY] <= window_start).groupby(log[CASE_KEY]).transform('sum')
    length = log.groupby(CASE_KEY)[CASE_KEY].transform('size')
    log = log.assign(**{MAX_PREFIX_KEY: np.minimum(before, length - 1)})
    return log[before > 0]


def sort_log(log: pd.DataFrame) -> pd.DataFrame:
    """Order a log by case start time, and the events of a case by their own timestamp.

    Returns:
        A reindexed copy. The sort is stable, so events sharing a timestamp keep the order the
        log gave them, which is the order cuts count events in.
    """
    case_start = log.groupby(CASE_KEY)[TIMESTAMP_KEY].min()
    rank = pd.Series(
        data=range(len(case_start)), index=case_start.sort_values(kind='mergesort').index
    )
    log = log.assign(_case_rank=log[CASE_KEY].map(rank))
    log = log.sort_values(by=['_case_rank', TIMESTAMP_KEY], kind='mergesort')
    return log.drop(columns='_case_rank').reset_index(drop=True)


def out_of_time_split(log: pd.DataFrame, *, test_fraction: float) -> TrainTestSplit:
    """Split a log out of time into train and test.

    The separation is the start of the case at the `1 - test_fraction` quantile of case starts.
    Train keeps the cases finished before it, test the cases still running at it. A test case
    that started before the separation keeps only the cuts whose prefix reaches past it, since
    its earlier prefixes would be predicted with training data from their future.

    Args:
        log: The debiased log, sorted by case start and then by timestamp.
        test_fraction: Fraction of all cases the separation time is placed at.

    Returns:
        Train and test, row subsets of `log` with `MIN_PREFIX_KEY` added: 1 everywhere
        except on the crossing test cases.
    """
    log = log.assign(**{MIN_PREFIX_KEY: 1})
    edges = log.groupby(CASE_KEY)[TIMESTAMP_KEY].agg(['min', 'max'])
    start_times, stop_times = edges['min'], edges['max']

    first_test_case = int(len(edges) * (1 - test_fraction))
    separation_time = np.sort(start_times.values)[first_test_case]

    train_cases = stop_times.index[stop_times.values < separation_time]
    train = log[log[CASE_KEY].isin(train_cases)]

    test_cases = stop_times.index[stop_times.values >= separation_time]
    crossing = test_cases.difference(start_times.sort_values().index[first_test_case:])
    test = _bound_crossing_prefixes(
        log[log[CASE_KEY].isin(test_cases)], separation_time=separation_time, crossing=crossing
    )
    return TrainTestSplit(train=train, test=test)


def _bound_crossing_prefixes(
    split: pd.DataFrame, *, separation_time: np.datetime64, crossing: pd.Index
) -> pd.DataFrame:
    """Raise `MIN_PREFIX_KEY` of the crossing cases to their first cut whose prefix holds an
    event after the separation."""
    split = split.copy()
    rows = split[split[CASE_KEY].isin(crossing)]
    # A cut of k takes the first k events, so a bound is the 1-based position of the event the
    # prefix has to reach.
    positions = rows.groupby(CASE_KEY).cumcount() + 1
    # Compared as numpy values, since a tz-aware column and a numpy timestamp are not comparable
    # as pandas objects.
    selected = rows[TIMESTAMP_KEY].values > separation_time
    bounds = positions[selected].groupby(rows[CASE_KEY][selected]).min()
    split[MIN_PREFIX_KEY] = split[CASE_KEY].map(bounds).fillna(split[MIN_PREFIX_KEY]).astype(int)
    return split
