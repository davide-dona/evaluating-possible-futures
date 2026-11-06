import numpy as np
import pandas as pd

from src.logs.keys import CASE_KEY, MIN_PREFIX_KEY, TIMESTAMP_KEY


def case_durations(log: pd.DataFrame) -> pd.Series:
    """The duration of every case in days, first event to last, indexed by case."""
    edges = log.groupby(CASE_KEY)[TIMESTAMP_KEY].agg(['min', 'max'])
    return (edges['max'] - edges['min']).dt.total_seconds() / 86_400.0


def case_lengths(log: pd.DataFrame) -> pd.Series:
    """The length of every case in events, indexed by case."""
    return log.groupby(CASE_KEY).size()


def drop_outlier_cases(
    log: pd.DataFrame, *, max_duration_days: float, max_length: int
) -> pd.DataFrame:
    """Keep the cases running at most `max_duration_days`, first event to last, and holding at
    most `max_length` events.

    A case spanning decades is a broken timestamp rather than a slow process.
    """
    kept = (case_durations(log) <= max_duration_days) & (case_lengths(log) <= max_length)
    return log[log[CASE_KEY].isin(kept.index[kept])]


def drop_cases_starting_late(log: pd.DataFrame, *, min_observed_days: float) -> pd.DataFrame:
    """Keep the cases starting at least `min_observed_days` before the last event of the log.

    A log is a snapshot: a case still running when it was extracted ends at its last recorded
    event, which would then be scored as the true end of the trace. Selecting on start time
    rather than on how a case ends keeps the selection independent of the scored suffixes.
    """
    cutoff = log[TIMESTAMP_KEY].max() - pd.Timedelta(days=min_observed_days)
    starts = log.groupby(CASE_KEY)[TIMESTAMP_KEY].min()
    return log[log[CASE_KEY].map(starts) <= cutoff]


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


def out_of_time_split(
    log: pd.DataFrame, *, val_frac: float, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a log out of time into train and test, and draw validation at random from train.

    The separation is the start of the case at the `1 - test_frac` quantile of case starts.
    Train keeps the cases finished before it, test the cases still running at it. A test case
    that started before the separation keeps only the cuts whose prefix reaches past it, since
    its earlier prefixes would be predicted with training data from their future.

    Args:
        log: The filtered log, sorted by case start and then by timestamp.
        val_frac: Fraction of all cases drawn from the train cases for validation.
        test_frac: Fraction of all cases the separation time is placed at.
        seed: Seed of the validation draw.

    Returns:
        `(train, val, test)`, row subsets of `log` with `MIN_PREFIX_KEY` added: 1 everywhere
        except on the crossing test cases.
    """
    log = log.assign(**{MIN_PREFIX_KEY: 1})
    edges = log.groupby(CASE_KEY)[TIMESTAMP_KEY].agg(['min', 'max'])
    starts_ts, stops_ts = edges['min'], edges['max']

    first_test_case = int(len(edges) * (1 - test_frac))
    separation_ts = np.sort(starts_ts.values)[first_test_case]

    train_cases = stops_ts.index[stops_ts.values < separation_ts]
    train = log[log[CASE_KEY].isin(train_cases)]

    test_cases = stops_ts.index[stops_ts.values >= separation_ts]
    crossing = test_cases.difference(starts_ts.sort_values().index[first_test_case:])
    test = _bound_crossing_prefixes(
        log[log[CASE_KEY].isin(test_cases)], separation_ts=separation_ts, crossing=crossing
    )

    # The candidates are taken in log order, so the draw depends on the seed alone.
    val_share = val_frac / (1 - test_frac)
    train_cases = train[CASE_KEY].unique()
    rng = np.random.default_rng(seed)
    val_cases = rng.choice(train_cases, size=int(len(train_cases) * val_share), replace=False)
    val = train[train[CASE_KEY].isin(val_cases)]
    train = train[~train[CASE_KEY].isin(val_cases)]
    return train, val, test


def _bound_crossing_prefixes(
    split: pd.DataFrame, *, separation_ts: np.datetime64, crossing: pd.Index
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
    selected = rows[TIMESTAMP_KEY].values > separation_ts
    bounds = positions[selected].groupby(rows[CASE_KEY][selected]).min()
    split[MIN_PREFIX_KEY] = split[CASE_KEY].map(bounds).fillna(split[MIN_PREFIX_KEY]).astype(int)
    return split
