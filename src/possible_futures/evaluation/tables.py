from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from possible_futures.datasets.cases import num_activities, num_events, num_prefixes, num_variants
from possible_futures.preprocessing.splitting import OutOfTimeSplit

DECIMALS = 3


@dataclass(frozen=True, slots=True)
class MetricColumn:
    """How a table prints the mean scores of one metric.

    Attributes:
        header: The column header, in LaTeX.
        higher_is_better: Whether the best score of a column is its highest, printed in bold.
    """

    header: str
    higher_is_better: bool


def scores_table(
    means: Mapping[str, pd.DataFrame],
    *,
    forecasts: Mapping[str, str],
    metrics: Mapping[str, MetricColumn],
) -> str:
    """A booktabs `tabular` of the mean scores of every dataset.

    Args:
        means: The mean scores of every set of forecasts, one row each, for every dataset, keyed
            by the name the table prints. A table of a single dataset has no dataset column.
        forecasts: The macro of every set of forecasts, in the order of the rows.
        metrics: The column of every metric, keyed by its name in `means`, in the order of
            the columns.

    Returns:
        The table, to `\\input` inside a `table` environment.
    """
    grouped = len(means) > 1
    labels = [*(['Event log'] if grouped else []), 'Decoding']
    headers = [*labels, *(f'{{{column.header}}}' for column in metrics.values())]
    score = f'S[table-format=1.{DECIMALS}, detect-weight, mode=text]'
    columns = 'l' * len(labels) + score * len(metrics)
    lines = [
        rf'\begin{{tabular}}{{@{{}}{columns}@{{}}}}',
        r'\toprule',
        ' & '.join(headers) + r' \\',
    ]
    for index, (dataset, scores) in enumerate(means.items()):
        cells = scores.loc[list(forecasts), list(metrics)].round(DECIMALS)
        best = {
            metric: cells[metric].max() if column.higher_is_better else cells[metric].min()
            for metric, column in metrics.items()
        }
        lines.append(r'\addlinespace' if index else r'\midrule')
        if grouped:
            lines.append(rf'\multirow{{{len(cells)}}}{{*}}{{{dataset}}}')
        for name, values in cells.iterrows():
            numbers = [
                rf'\bfseries {value:.{DECIMALS}f}'
                if value == best[metric]
                else f'{value:.{DECIMALS}f}'
                for metric, value in values.items()
            ]
            group = [''] if grouped else []
            lines.append(' & '.join([*group, forecasts[name], *numbers]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def properties_table(splits: Mapping[str, OutOfTimeSplit]) -> str:
    """A booktabs `tabular` of the size and case lengths of every log and its test prefixes.

    Args:
        splits: The train and test cases of every preprocessed log, keyed by the name the table
            prints.

    Returns:
        The table, to `\\input` inside a `table` environment.
    """
    lines = [
        r'\begin{tabular}{@{}lrrrrD{+}{\,\pm\,}{-1}r@{}}',
        r'\toprule',
        r'Event log & Cases & Events & Variants & Activities & \multicolumn{1}{c}{Case length}'
        r' & Test prefixes \\',
        r'\midrule',
    ]
    for name, split in splits.items():
        log = split.train + split.test
        lengths = np.array([len(case) for case in log])
        cells = [
            name,
            _grouped_digits(len(log)),
            _grouped_digits(num_events(log)),
            _grouped_digits(num_variants(log)),
            _grouped_digits(num_activities(log)),
            f'{lengths.mean():.1f}+{lengths.std(ddof=1):.1f}',
            _grouped_digits(num_prefixes(split.test)),
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def _grouped_digits(count: int) -> str:
    """A count with its digits grouped in thousands by commas, braced so that math mode adds no
    space after them."""
    return f'{count:,}'.replace(',', '{,}')
