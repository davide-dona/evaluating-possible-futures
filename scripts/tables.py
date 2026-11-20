from dataclasses import dataclass

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.cases import load_cases, load_scored
from possible_futures.evaluation.scoring import load_scores
from possible_futures.evaluation.tables import MetricColumn, properties_table, scores_table
from possible_futures.preprocessing.splitting import OutOfTimeSplit
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import OUTPUTS, Split, scored_file, scores_file, split_file


@dataclass
class ScoresTableConfig:
    """One table of mean scores.

    Attributes:
        datasets: The datasets of the table, one group of rows each.
        forecasts: The forecasts of every dataset, one row each.
    """

    datasets: list[str] = MISSING
    forecasts: list[str] = MISSING


@dataclass
class MetricColumnConfig:
    """How a table prints the mean scores of one metric.

    Attributes:
        header: The column header, in LaTeX.
        higher_is_better: Whether the best score of a column is its highest.
    """

    header: str = MISSING
    higher_is_better: bool = MISSING


@dataclass
class TablesConfig:
    """The settings of the tables stage.

    Attributes:
        datasets: The name every table prints for every dataset.
        scores: Every table of mean scores, keyed by its file stem.
        properties: The datasets of the table of log properties.
        forecasts: The LaTeX macro every table prints for every set of forecasts.
        metrics: The column of every metric, keyed by its name in the scores files.
    """

    datasets: dict[str, str] = MISSING
    scores: dict[str, ScoresTableConfig] = MISSING
    properties: list[str] = MISSING
    forecasts: dict[str, str] = MISSING
    metrics: dict[str, MetricColumnConfig] = MISSING


ConfigStore.instance().store(name='tables_schema', node=TablesConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='tables')
def main(config: TablesConfig) -> None:
    """Write a LaTeX table of mean scores per entry of `config.scores`, and one of the logs."""
    output = OUTPUTS / 'tables'
    banner(
        'Writing the LaTeX tables',
        {name: ', '.join(table.datasets) for name, table in config.scores.items()}
        | {'properties': ', '.join(config.properties), 'output': output},
    )
    metrics = {
        name: MetricColumn(header=column.header, higher_is_better=column.higher_is_better)
        for name, column in config.metrics.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, table in config.scores.items():
        path = output / f'{name}.tex'
        with step(f'Writing {path}'):
            means = {
                config.datasets[dataset]: pd.DataFrame.from_dict(
                    {
                        forecasts_name: load_scores(scores_file(dataset, forecasts_name))
                        for forecasts_name in table.forecasts
                    },
                    orient='index',
                )
                for dataset in table.datasets
            }
            macros = {name: config.forecasts[name] for name in table.forecasts}
            path.write_text(scores_table(means, forecasts=macros, metrics=metrics))
    path = output / 'properties.tex'
    with step(f'Writing {path}'):
        splits = {
            config.datasets[dataset]: OutOfTimeSplit(
                train=load_cases(split_file(dataset, Split.TRAIN)),
                test=load_cases(split_file(dataset, Split.TEST)),
                scored=load_scored(scored_file(dataset)),
            )
            for dataset in config.properties
        }
        path.write_text(properties_table(splits))


if __name__ == '__main__':
    main()
