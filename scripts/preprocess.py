from dataclasses import dataclass

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.cases import (
    Case,
    num_activities,
    num_events,
    num_prefixes,
    num_scored_prefixes,
    num_variants,
    save_cases,
    save_scored,
)
from possible_futures.datasets.tokenizer import Tokenizer
from possible_futures.preprocessing.log import drop_outliers, read_log
from possible_futures.preprocessing.splitting import OutOfTimeSplit
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    original_log_file,
    scored_file,
    split_file,
    tokenizer_file,
)


@dataclass
class PreprocessConfig:
    """The settings of the preprocess stage.

    Attributes:
        data: The dataset.
    """

    data: DataConfig = MISSING


ConfigStore.instance().store(name='preprocess_schema', node=PreprocessConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='preprocess')
def main(config: PreprocessConfig) -> None:
    """Filter and split a raw log, pick its scored test prefixes, and fit the tokenizer."""
    data = config.data
    split = data.split
    banner(
        f'Preprocessing "{data.name}"',
        {
            'original log': original_log_file(data.name),
            'split': f'{1 - split.test_fraction:.0%} train, {split.test_fraction:.0%} test, '
            'out of time',
            'splits': split_file(data.name, Split.TRAIN).parent,
        },
    )
    with step('Reading the original log'):
        cases = read_log(
            original_log_file(data.name),
            case_key=data.case_key,
            activity_key=data.activity_key,
            timestamp_key=data.timestamp_key,
            separator=data.separator,
        )
        print(f'  {num_events(cases):,} events over {len(cases):,} cases', flush=True)

    max_duration_days = float(
        np.percentile([case.duration_days for case in cases], q=split.max_case_duration_percentile)
    )
    max_length = int(
        np.ceil(np.percentile([len(case) for case in cases], q=split.max_case_length_percentile))
    )
    with step(
        f'Dropping the cases longer than {max_length} events or running longer than '
        f'{max_duration_days:,.1f} days'
    ):
        kept = drop_outliers(cases, max_length=max_length, max_duration_days=max_duration_days)
        _report_dropped(read=len(cases), kept=len(kept))

    with step(
        f'Debiasing the last {max_duration_days:,.1f} days and splitting the log out of time'
    ):
        splits = OutOfTimeSplit.of(
            kept, test_fraction=split.test_fraction, end_window_days=max_duration_days
        )
        _report_dropped(read=len(kept), kept=len(splits.train) + len(splits.test))

    with step('Writing the splits, the scored prefixes and the tokenizer'):
        save_cases(splits.train, path=split_file(data.name, Split.TRAIN))
        save_cases(splits.test, path=split_file(data.name, Split.TEST))
        save_scored(splits.scored, path=scored_file(data.name))
        tokenizer = Tokenizer.fit(splits.train)
        tokenizer.save(tokenizer_file(data.name))

    summary = pd.DataFrame(
        {
            'train': _describe(splits.train, prefixes=num_prefixes(splits.train)),
            'test': _describe(splits.test, prefixes=num_scored_prefixes(splits.scored)),
        }
    ).T
    print(
        f'\nPreprocessed "{data.name}": {len(tokenizer.activities)} train activities, cases of '
        f'at most {max_length} events and {max_duration_days:,.1f} days\n'
        f'{summary.to_string()}\n'
        'Train counts every prefix with a non-empty suffix, test only the scored ones.',
        flush=True,
    )


def _describe(cases: list[Case], *, prefixes: int) -> dict[str, str]:
    """Describe the size of a split, its prefixes, and the shape of its cases, for printing.

    Args:
        cases: The cases of the split.
        prefixes: The number of prefixes of the split, trained on or scored.

    Returns:
        dict[str, str]: Every statistic, formatted, keyed by its name.
    """
    lengths = np.array([len(case) for case in cases])
    return {
        'cases': f'{len(cases):,}',
        'events': f'{num_events(cases):,}',
        'prefixes': f'{prefixes:,}',
        'variants': f'{num_variants(cases):,}',
        'activities': f'{num_activities(cases)}',
        'case length': f'{lengths.mean():.1f} ± {lengths.std(ddof=1):.1f}',
    }


def _report_dropped(*, read: int, kept: int) -> None:
    """Print how many of the cases read were dropped.

    Args:
        read: The number of cases before the drop.
        kept: The number of cases after it.
    """
    dropped = read - kept
    print(f'  dropped {dropped:,} of {read:,} cases ({dropped / read:.2%})', flush=True)


if __name__ == '__main__':
    main()
