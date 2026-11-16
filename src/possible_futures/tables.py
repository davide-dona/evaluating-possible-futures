from collections.abc import Mapping

import pandas as pd

from possible_futures.logs.keys import ACTIVITY_KEY
from possible_futures.logs.statistics import case_durations, case_lengths, num_variants

# The columns of the scores the evaluate stage writes, in the order printed, each with its
# header in the paper's macros and whether a higher mean is better.
METRICS = {
    'DLS': (r'$\meanDLS$~$\uparrow$', True),
    'ES dl': (r'\DL~$\downarrow$', False),
    'ES exact': (r'\exact~$\downarrow$', False),
    'ES bigram': (r'\ngram[2]~$\downarrow$', False),
}
DECIMALS = 3


def scores_table(means: Mapping[str, pd.DataFrame], decodings: Mapping[str, str]) -> str:
    """A booktabs `tabular` of the means of every dataset.

    Args:
        means: The scores the evaluate stage writes for every dataset, keyed by the name the
            table prints. A table of a single dataset has no dataset column.
        decodings: The macro of every decoding, in the order of the rows. A decoding a dataset
            was not scored on has no row.

    Returns:
        The table, to `\\input` inside a `table` environment.
    """
    grouped = len(means) > 1
    labels = [*(['Event log'] if grouped else []), 'Decoding']
    headers = [*labels, *(f'{{{h}}}' for h, _ in METRICS.values())]
    score = f'S[table-format=1.{DECIMALS}, detect-weight, mode=text]'
    columns = 'l' * len(labels) + score * len(METRICS)
    lines = [
        rf'\begin{{tabular}}{{@{{}}{columns}@{{}}}}',
        r'\toprule',
        ' & '.join(headers) + r' \\',
    ]
    for index, (dataset, scores) in enumerate(means.items()):
        rows = scores.loc[[decoding for decoding in decodings if decoding in scores.index]]
        cells = rows[list(METRICS)].round(DECIMALS)
        best = {
            metric: cells[metric].max() if higher else cells[metric].min()
            for metric, (_, higher) in METRICS.items()
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
        cells = [
            name,
            _grouped_digits(len(lengths)),
            _grouped_digits(len(log)),
            _grouped_digits(num_variants(log)),
            _grouped_digits(log[ACTIVITY_KEY].nunique()),
            f'{lengths.mean():.1f}+{lengths.std():.1f}',
            f'{durations.mean():.2f}+{durations.std():.2f}',
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def _grouped_digits(count: int) -> str:
    """A count with its digits grouped in thousands by commas, braced so that math mode adds no
    space after them."""
    return f'{count:,}'.replace(',', '{,}')
