from dataclasses import dataclass
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from possible_futures.forecasters.forecasts import ForecastWriter, iter_forecasts
from possible_futures.forecasters.medoid import medoids
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import forecasts_file


@dataclass
class CollapseConfig:
    """The settings of the collapse stage.

    Attributes:
        data: The dataset.
        forecasts: The name of the forecasts collapsed, the stem of their file.
        output: The name of the collapsed forecasts, the stem of their file.
        max_rows: The most draws held in memory at once.
        distance: The `Distance` the medoid is taken under, built with
            `hydra.utils.instantiate`.
    """

    data: DataConfig = MISSING
    forecasts: str = MISSING
    output: str = MISSING
    max_rows: int = MISSING
    distance: Any = MISSING


ConfigStore.instance().store(name='collapse_schema', node=CollapseConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='collapse')
def main(config: CollapseConfig) -> None:
    """Keep only the medoid of the sampled suffixes of every test prefix."""
    name = config.data.name
    source = forecasts_file(name, config.forecasts)
    path = forecasts_file(name, config.output)
    distance = instantiate(config.distance)
    banner(
        f'Collapsing the {config.forecasts} suffixes of "{name}"',
        {
            'forecasts': source,
            'distance': type(distance).__name__,
            config.output: f'1 draw per prefix, written to {path}',
        },
    )
    with step('Taking the medoid of every prefix'), ForecastWriter(path) as writer:
        num_prefixes = 0
        for forecasts in iter_forecasts(source, max_rows=config.max_rows):
            writer.write(medoids(forecasts, distance=distance))
            num_prefixes += len(forecasts)
        print(f'  {num_prefixes:,} prefixes', flush=True)


if __name__ == '__main__':
    main()
