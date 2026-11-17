import hydra
from hydra.core.config_store import ConfigStore

from possible_futures.evaluation.scoring import MeanScores
from possible_futures.logs.io import read_split
from possible_futures.runs.configs import TablesConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import OUTPUTS, Split, scores_file, split_file
from possible_futures.tables import MetricColumn, properties_table, scores_table

ConfigStore.instance().store(name='tables_schema', node=TablesConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='tables')
def main(config: TablesConfig) -> None:
    """Write the LaTeX tables of the paper: one per entry of `config.scores`, holding the mean
    scores of its decodings on its datasets, and `properties.tex`, describing the preprocessed
    logs."""
    output = OUTPUTS / 'tables'
    banner(
        'Writing the tables of the paper',
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
                config.datasets[dataset]: MeanScores.load(scores_file(dataset)).table
                for dataset in table.datasets
            }
            decodings = {decoding: config.decodings[decoding] for decoding in table.decodings}
            path.write_text(scores_table(means, decodings=decodings, metrics=metrics))
    path = output / 'properties.tex'
    with step(f'Writing {path}'):
        splits = {
            config.datasets[dataset]: (
                read_split(split_file(dataset, Split.TRAIN)),
                read_split(split_file(dataset, Split.TEST)),
            )
            for dataset in config.properties
        }
        path.write_text(properties_table(splits))


if __name__ == '__main__':
    main()
