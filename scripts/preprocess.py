import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.logs.io import read_original_log, write_split
from possible_futures.logs.keys import ACTIVITY_KEY, CASE_KEY, MIN_PREFIX_KEY, Split
from possible_futures.logs.preprocessing import (
    case_durations,
    case_lengths,
    drop_cases_starting_late,
    drop_outlier_cases,
    out_of_time_split,
    sort_log,
)
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import original_log, split_file, vocabulary_file


def run(data: DictConfig) -> None:
    """Filter, sort, and split a raw log, and fit the vocabulary on its train split.

    Args:
        data: The `data` section.
    """
    banner(
        f'Preprocessing "{data.name}"',
        {
            'original log': original_log(data.name),
            'split': f'{1 - data.val_split - data.test_split:.0%} train, '
            f'{data.val_split:.0%} val, {data.test_split:.0%} test, '
            'test out of time, val at random',
            'splits': split_file(data.name, Split.TRAIN).parent,
        },
    )
    with step('Reading the original log'):
        log = read_original_log(original_log(data.name), data)
        print(f'  {len(log):,} events over {log[CASE_KEY].nunique():,} cases', flush=True)

    # Compute the values for dropping outlier cases
    max_duration_days = float(np.percentile(case_durations(log), data.max_case_duration_percentile))
    max_length = int(np.ceil(np.percentile(case_lengths(log), data.max_case_length_percentile)))
    with step(
        f'Dropping the cases longer than {max_length} events or running longer than '
        f'{max_duration_days:,.1f} days'
    ):
        # Drop the outlier cases and report how many were dropped
        kept = drop_outlier_cases(log, max_duration_days=max_duration_days, max_length=max_length)
        _report_dropped(before=log, after=kept)
        log = kept
    # Drop the cases starting too close to the end of the log.
    with step(f'Dropping the cases starting within {max_duration_days:,.1f} days of the log end'):
        kept = drop_cases_starting_late(log, min_observed_days=max_duration_days)
        _report_dropped(before=log, after=kept)
        log = kept

    with step('Sorting and splitting the log out of time'):
        train, val, test = out_of_time_split(
            sort_log(log),
            val_frac=data.val_split,
            test_frac=data.test_split,
            seed=data.split_seed,
        )

    with step('Writing the splits and the vocabulary'):
        for split, rows in ((Split.TRAIN, train), (Split.VAL, val), (Split.TEST, test)):
            write_split(rows, split_file(data.name, split))
        # Write vocabulary fitted on the train split, which is used for encoding the splits for training
        vocabulary = Vocabulary.fit(train, max_case_length=max_length)
        vocabulary.save(vocabulary_file(data.name))

    # Report the statistics of the splits.
    summary = pd.DataFrame(
        {name: _describe(rows) for name, rows in (('train', train), ('val', val), ('test', test))}
    ).T
    print(
        f'\nPreprocessed "{data.name}": {len(vocabulary.activities)} train activities, cases of '
        f'at most {max_length} events and {max_duration_days:,.1f} days\n'
        f'{summary.to_string()}\n'
        'Prefixes are the cuts leaving a non-empty suffix; on test, only the scored ones.',
        flush=True,
    )


def _describe(split: pd.DataFrame) -> dict[str, str]:
    """The size of a split and the shape of its cases, formatted for printing."""
    traces = split.groupby(CASE_KEY, sort=False)[ACTIVITY_KEY].agg(tuple)
    lengths = traces.map(len)
    lowest = split.groupby(CASE_KEY, sort=False)[MIN_PREFIX_KEY].first()
    prefixes = (lengths - lowest).clip(lower=0).sum()
    return {
        'cases': f'{len(traces):,}',
        'events': f'{len(split):,}',
        'prefixes': f'{prefixes:,}',
        'variants': f'{traces.nunique():,}',
        'activities': f'{split[ACTIVITY_KEY].nunique()}',
        'case length': f'{lengths.mean():.1f} ± {lengths.std():.1f}',
    }


def _report_dropped(*, before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Report how many cases were dropped between two logs."""
    read = before[CASE_KEY].nunique()
    dropped = read - after[CASE_KEY].nunique()
    print(f'  dropped {dropped:,} of {read:,} cases ({dropped / read:.2%})', flush=True)


@hydra.main(version_base='1.3', config_path='../config', config_name='preprocess')
def main(config: DictConfig) -> None:
    run(config.data)


if __name__ == '__main__':
    main()
