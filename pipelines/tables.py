from collections.abc import Mapping

import hydra
import pandas as pd
from omegaconf import DictConfig

from pipelines.helpers.console import banner, step
from pipelines.helpers.paths import OUTPUTS, split_file, stage_dir
from src.evaluation.scores import summary
from src.logs.keys import ACTIVITY_KEY, CASE_KEY, TIMESTAMP_KEY, Split
from src.logs.preprocessing import case_durations, case_lengths

# The columns of `summary` in the order printed, each with its header in the paper's macros and
# whether a higher mean is better.
METRICS = {
    'DLS': (r'$\meanDLS$~$\uparrow$', True),
    'ES dl': (r'\DL~$\downarrow$', False),
    'ES exact': (r'\exact~$\downarrow$', False),
    'ES bigram': (r'\ngram[2]~$\downarrow$', False),
}
DECIMALS = 3


def run(config: DictConfig) -> None:
    """Write the LaTeX tables of the paper: one per entry of `config.scores`, holding the mean
    scores of its decodings on its datasets, and `properties.tex`, describing the preprocessed
    logs."""
    output = OUTPUTS / 'tables'
    banner(
        'Writing the tables of the paper',
        {name: ', '.join(table.datasets) for name, table in config.scores.items()}
        | {'properties': ', '.join(config.properties), 'output': output},
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, table in config.scores.items():
        path = output / f'{name}.tex'
        with step(f'Writing {path}'):
            means = {
                config.datasets[dataset]: summary(
                    pd.read_parquet(stage_dir(dataset, 'evaluate') / 'scores.parquet')
                )
                for dataset in table.datasets
            }
            decodings = {decoding: config.decodings[decoding] for decoding in table.decodings}
            path.write_text(scores_table(means, decodings))
    path = output / 'properties.tex'
    with step(f'Writing {path}'):
        logs = {config.datasets[dataset]: read_log(dataset) for dataset in config.properties}
        path.write_text(properties_table(logs))


def read_log(dataset: str) -> pd.DataFrame:
    """The preprocessed log of a dataset, its three splits together."""
    splits = [
        pd.read_csv(
            split_file(dataset, split),
            usecols=[CASE_KEY, ACTIVITY_KEY, TIMESTAMP_KEY],
            dtype={CASE_KEY: str, ACTIVITY_KEY: str},
        )
        for split in Split
    ]
    log = pd.concat(splits, ignore_index=True)
    log[TIMESTAMP_KEY] = pd.to_datetime(log[TIMESTAMP_KEY], format='mixed')
    return log


def properties_table(logs: Mapping[str, pd.DataFrame]) -> str:
    """A booktabs `tabular` describing every log: its cases, events, variants and activities,
    and the mean and standard deviation of its case length in events and duration in days.

    Args:
        logs: The preprocessed logs, keyed by the name the table prints.

    Returns:
        The table, to `\\input` inside a `table` environment.
    """
    lines = [
        r'\begin{tabular}{@{}lrrrrD{+}{\,\pm\,}{-1}D{+}{\,\pm\,}{-1}@{}}',
        r'\toprule',
        r'Event log & Cases & Events & Variants & Activities & \multicolumn{1}{c}{Case length}'
        r' & \multicolumn{1}{c}{Case duration} \\',
        r'\midrule',
    ]
    for name, log in logs.items():
        lengths, durations = case_lengths(log), case_durations(log)
        variants = log.groupby(CASE_KEY, sort=False)[ACTIVITY_KEY].agg(tuple).nunique()
        cells = [
            name,
            str(len(lengths)),
            str(len(log)),
            str(variants),
            str(log[ACTIVITY_KEY].nunique()),
            f'{lengths.mean():.1f}+{lengths.std():.1f}',
            f'{durations.mean():.2f}+{durations.std():.2f}',
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def scores_table(means: Mapping[str, pd.DataFrame], decodings: Mapping[str, str]) -> str:
    """A booktabs `tabular` of the means of every dataset.

    Args:
        means: The `summary` of every dataset, keyed by the name the table prints. A table of a
            single dataset has no dataset column.
        decodings: The macro of every decoding, in the order of the rows. A decoding a dataset
            was not scored on has no row.

    Returns:
        The table, to `\\input` inside a `table` environment.
    """
    grouped = len(means) > 1
    headers = [*(['Dataset'] if grouped else []), 'Decoding', *(h for h, _ in METRICS.values())]
    columns = 'l' * (len(headers) - len(METRICS)) + 'c' * len(METRICS)
    lines = [
        rf'\begin{{tabular}}{{@{{}}{columns}@{{}}}}',
        r'\toprule',
        ' & '.join(headers) + r' \\',
    ]
    for dataset, scores in means.items():
        rows = scores.loc[[decoding for decoding in decodings if decoding in scores.index]]
        cells = rows[list(METRICS)].round(DECIMALS)
        best = {
            metric: cells[metric].max() if higher else cells[metric].min()
            for metric, (_, higher) in METRICS.items()
        }
        lines.append(r'\midrule')
        if grouped:
            lines.append(rf'\multirow{{{len(cells)}}}{{*}}{{{dataset}}}')
        for decoding, values in cells.iterrows():
            numbers = [
                rf'\textbf{{{value:.{DECIMALS}f}}}'
                if value == best[metric]
                else f'{value:.{DECIMALS}f}'
                for metric, value in values.items()
            ]
            group = [''] if grouped else []
            lines.append(' & '.join([*group, decodings[decoding], *numbers]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


@hydra.main(version_base='1.3', config_path='../config', config_name='tables')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
