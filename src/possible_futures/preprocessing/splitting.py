import numpy as np
import pandas as pd

from possible_futures.datasets.cases import Cases, TrainTestSplit
from possible_futures.preprocessing.events import (
    ACTIVITY_COLUMN,
    CASE_COLUMN,
    TIMESTAMP_COLUMN,
    case_lengths,
)


def out_of_time_split(
    events: pd.DataFrame, *, test_fraction: float, end_window_days: float
) -> TrainTestSplit:
    """Drop the cases starting in the last `end_window_days` of the log, then split the rest at
    the start of the case at the `1 - test_fraction` quantile of case starts.

    Train keeps the cases finished before the separation, test the cases still running at it.
    Every case is cut up to its last prefix that ends before the end window and leaves a
    non-empty suffix. A test case started before the separation is cut from its first prefix
    reaching past it.
    """
    events = _sort(events)
    window_start = events[TIMESTAMP_COLUMN].max() - pd.Timedelta(days=end_window_days)
    # A cut of k ends at the k-th event in time order, so the events before the window count
    # the cuts ending before it.
    before_window = (events[TIMESTAMP_COLUMN] <= window_start).groupby(events[CASE_COLUMN]).sum()
    max_prefix_lengths = np.minimum(before_window, case_lengths(events) - 1)
    events = events[events[CASE_COLUMN].isin(before_window.index[before_window > 0])]

    edges = events.groupby(CASE_COLUMN)[TIMESTAMP_COLUMN].agg(['min', 'max'])
    start_times, stop_times = edges['min'], edges['max']
    first_test_case = int(len(edges) * (1 - test_fraction))
    separation_time = np.sort(start_times.values)[first_test_case]
    train_cases = stop_times.index[stop_times.values < separation_time]
    test_cases = stop_times.index[stop_times.values >= separation_time]
    crossing = test_cases.difference(start_times.sort_values().index[first_test_case:])

    min_prefix_lengths = pd.Series(1, index=edges.index)
    first_cuts = _first_cuts_after(
        events[events[CASE_COLUMN].isin(crossing)], separation_time=separation_time
    )
    min_prefix_lengths.loc[first_cuts.index] = first_cuts
    return TrainTestSplit(
        train=_cases(
            events[events[CASE_COLUMN].isin(train_cases)],
            min_prefix_lengths=min_prefix_lengths,
            max_prefix_lengths=max_prefix_lengths,
        ),
        test=_cases(
            events[events[CASE_COLUMN].isin(test_cases)],
            min_prefix_lengths=min_prefix_lengths,
            max_prefix_lengths=max_prefix_lengths,
        ),
    )


def _sort(events: pd.DataFrame) -> pd.DataFrame:
    """Sort cases by start time and the events of each case by timestamp, ties kept in order."""
    case_start = events.groupby(CASE_COLUMN)[TIMESTAMP_COLUMN].min()
    rank = pd.Series(
        data=range(len(case_start)), index=case_start.sort_values(kind='mergesort').index
    )
    events = events.assign(_case_rank=events[CASE_COLUMN].map(rank))
    events = events.sort_values(by=['_case_rank', TIMESTAMP_COLUMN], kind='mergesort')
    return events.drop(columns='_case_rank').reset_index(drop=True)


def _first_cuts_after(events: pd.DataFrame, *, separation_time: np.datetime64) -> pd.Series:
    """The first cut of every case whose prefix holds an event after the separation."""
    # A cut of k takes the first k events, so a cut is the 1-based position of the event the
    # prefix has to reach.
    positions = events.groupby(CASE_COLUMN).cumcount() + 1
    # Compared as numpy values, since a tz-aware column and a numpy timestamp are not comparable
    # as pandas objects.
    selected = events[TIMESTAMP_COLUMN].values > separation_time
    return positions[selected].groupby(events[CASE_COLUMN][selected]).min()


def _cases(
    events: pd.DataFrame, *, min_prefix_lengths: pd.Series, max_prefix_lengths: pd.Series
) -> Cases:
    """The cases of sorted events, in order of their first event."""
    traces = events.groupby(CASE_COLUMN, sort=False)[ACTIVITY_COLUMN].agg(tuple)
    return Cases(
        case_ids=traces.index.to_numpy(dtype=object),
        traces=tuple(traces),
        min_prefix_lengths=min_prefix_lengths.loc[traces.index].to_numpy(np.int64),
        max_prefix_lengths=max_prefix_lengths.loc[traces.index].to_numpy(np.int64),
    )
