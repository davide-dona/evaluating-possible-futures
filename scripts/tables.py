import hydra
import pandas as pd
from omegaconf import DictConfig

from possible_futures.evaluation.scores import summary
from possible_futures.logs.io import read_split
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import OUTPUTS, Split, scores_file, split_file
from possible_futures.tables import properties_table, scores_table


@hydra.main(version_base='1.3', config_path='../config', config_name='tables')
def main(config: DictConfig) -> None:
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
                config.datasets[dataset]: summary(pd.read_parquet(scores_file(dataset)))
                for dataset in table.datasets
            }
            decodings = {decoding: config.decodings[decoding] for decoding in table.decodings}
            path.write_text(scores_table(means, decodings))
    path = output / 'properties.tex'
    with step(f'Writing {path}'):
        logs = {
            config.datasets[dataset]: pd.concat(
                [read_split(split_file(dataset, split)) for split in Split], ignore_index=True
            )
            for dataset in config.properties
        }
        path.write_text(properties_table(logs))


if __name__ == '__main__':
    main()
