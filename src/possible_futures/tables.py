from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from possible_futures.logs.keys import ACTIVITY_KEY
from possible_futures.logs.statistics import case_lengths, num_prefixes, num_variants

DECIMALS = 3


@dataclass(frozen=True, slots=True)
class MetricColumn:
    """How a table prints the mean scores of one metric."""

    header: str  # LaTeX, in the paper's macros
    higher_is_better: bool


def scores_table(
    means: Mapping[str, pd.DataFrame],
    *,
    decodings: Mapping[str, str],
    metrics: Mapping[str, MetricColumn],
) -> str:
    """A booktabs `tabular` of the mean scores of every dataset.

    Args:
        means: The mean scores of every decoding, one row each, for every dataset, keyed by the
            name the table prints. A table of a single dataset has no dataset column.
        decodings: The macro of every decoding, in the order of the rows. A decoding a dataset
            was not scored on has no row.
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
        rows = scores.loc[[decoding for decoding in decodings if decoding in scores.index]]
        cells = rows[list(metrics)].round(DECIMALS)
        best = {
            metric: cells[metric].max() if column.higher_is_better else cells[metric].min()
            for metric, column in metrics.items()
        }
        lines.append(r'\addlinespace' if index else r'\midrule')
        if grouped:
            lines.append(rf'\multirow{{{len(cells)}}}{{*}}{{{dataset}}}')
        for decoding, values in cells.iterrows():
            numbers = [
                rf'\bfseries {value:.{DECIMALS}f}'
                if value == best[metric]
                else f'{value:.{DECIMALS}f}'
                for metric, value in values.items()
            ]
            group = [''] if grouped else []
            lines.append(' & '.join([*group, decodings[decoding], *numbers]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def properties_table(splits: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]]) -> str:
    """A booktabs `tabular` describing every log: its cases, events, variants and activities,
    the mean and standard deviation of its case length in events, and the number of prefixes of
    its test split, which every mean score is taken over.

    Args:
        splits: The train and test splits of every preprocessed log, as read by `read_split`,
            keyed by the name the table prints.

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
    for name, (train, test) in splits.items():
        log = pd.concat([train, test], ignore_index=True)
        lengths = case_lengths(log)
        cells = [
            name,
            _grouped_digits(len(lengths)),
            _grouped_digits(len(log)),
            _grouped_digits(num_variants(log)),
            _grouped_digits(log[ACTIVITY_KEY].nunique()),
            f'{lengths.mean():.1f}+{lengths.std():.1f}',
            _grouped_digits(num_prefixes(test)),
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def _grouped_digits(count: int) -> str:
    """A count with its digits grouped in thousands by commas, braced so that math mode adds no
    space after them."""
    return f'{count:,}'.replace(',', '{,}')
