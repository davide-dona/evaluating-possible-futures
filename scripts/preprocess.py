from dataclasses import dataclass

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.datasets.cases import Cases
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.preprocessing.events import (
    case_durations_days,
    case_lengths,
    drop_outlier_cases,
    num_cases,
    read_original_log,
)
from possible_futures.preprocessing.splitting import out_of_time_split
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import Split, original_log_file, split_file, vocabulary_file


@dataclass
class PreprocessConfig:
    data: DataConfig = MISSING


ConfigStore.instance().store(name='preprocess_schema', node=PreprocessConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='preprocess')
def main(config: PreprocessConfig) -> None:
    """Filter, sort, and split a raw log, and fit the vocabulary on its train split."""
    data = config.data
    banner(
        f'Preprocessing "{data.name}"',
        {
            'original log': original_log_file(data.name),
            'split': f'{1 - data.test_fraction:.0%} train, {data.test_fraction:.0%} test, '
            'out of time',
            'splits': split_file(data.name, Split.TRAIN).parent,
        },
    )
    with step('Reading the original log'):
        events = read_original_log(
            original_log_file(data.name),
            case_key=data.case_key,
            activity_key=data.activity_key,
            timestamp_key=data.timestamp_key,
            separator=data.separator,
        )
        print(f'  {len(events):,} events over {num_cases(events):,} cases', flush=True)

    max_duration_days = float(
        np.percentile(case_durations_days(events), data.max_case_duration_percentile)
    )
    max_length = int(np.ceil(np.percentile(case_lengths(events), data.max_case_length_percentile)))
    with step(
        f'Dropping the cases longer than {max_length} events or running longer than '
        f'{max_duration_days:,.1f} days'
    ):
        kept = drop_outlier_cases(
            events, max_duration_days=max_duration_days, max_length=max_length
        )
        _report_dropped(read=num_cases(events), kept=num_cases(kept))

    with step(
        f'Debiasing the last {max_duration_days:,.1f} days and splitting the log out of time'
    ):
        train, test = out_of_time_split(
            kept, test_fraction=data.test_fraction, end_window_days=max_duration_days
        )
        _report_dropped(read=num_cases(kept), kept=len(train) + len(test))

    with step('Writing the splits and the vocabulary'):
        train.save(split_file(data.name, Split.TRAIN))
        test.save(split_file(data.name, Split.TEST))
        vocabulary = Vocabulary.fit(train.traces)
        vocabulary.save(vocabulary_file(data.name))

    summary = pd.DataFrame({'train': _describe(train), 'test': _describe(test)}).T
    print(
        f'\nPreprocessed "{data.name}": {len(vocabulary.activities)} train activities, cases of '
        f'at most {max_length} events and {max_duration_days:,.1f} days\n'
        f'{summary.to_string()}\n'
        'Prefixes are the cuts between the lowest and the highest cut of every case.',
        flush=True,
    )


def _describe(cases: Cases) -> dict[str, str]:
    """The size of a split and the shape of its cases, formatted for printing."""
    lengths = cases.case_lengths
    return {
        'cases': f'{len(cases):,}',
        'events': f'{lengths.sum():,}',
        'prefixes': f'{cases.num_prefixes():,}',
        'variants': f'{cases.num_variants():,}',
        'activities': f'{cases.num_activities()}',
        'case length': f'{lengths.mean():.1f} ± {lengths.std(ddof=1):.1f}',
    }


def _report_dropped(*, read: int, kept: int) -> None:
    """Report how many of the cases read were dropped."""
    dropped = read - kept
    print(f'  dropped {dropped:,} of {read:,} cases ({dropped / read:.2%})', flush=True)


if __name__ == '__main__':
    main()
