from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.evaluation.scoring import load_scores
from possible_futures.preprocessing.cases import load_cases, num_prefixes
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import OUTPUTS, Split, scores_file, split_file
from possible_futures.tables import MetricColumn, properties_table, scores_table


@dataclass
class ScoresTableConfig:
    datasets: list[str] = MISSING
    decodings: list[str] = MISSING


@dataclass
class MetricColumnConfig:
    header: str = MISSING
    higher_is_better: bool = MISSING


@dataclass
class TablesConfig:
    datasets: dict[str, str] = MISSING
    scores: dict[str, ScoresTableConfig] = MISSING
    properties: list[str] = MISSING
    decodings: dict[str, str] = MISSING
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
                config.datasets[dataset]: load_scores(scores_file(dataset))
                for dataset in table.datasets
            }
            decodings = {decoding: config.decodings[decoding] for decoding in table.decodings}
            path.write_text(scores_table(means, decodings=decodings, metrics=metrics))
    path = output / 'properties.tex'
    with step(f'Writing {path}'):
        cases, num_test_prefixes = {}, {}
        for dataset in config.properties:
            name = config.datasets[dataset]
            test = load_cases(split_file(dataset, Split.TEST))
            cases[name] = load_cases(split_file(dataset, Split.TRAIN)) + test
            num_test_prefixes[name] = num_prefixes(test)
        path.write_text(properties_table(cases, num_test_prefixes=num_test_prefixes))


if __name__ == '__main__':
    main()
