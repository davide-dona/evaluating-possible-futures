from dataclasses import dataclass
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.draws import Decoding
from possible_futures.inference.medoid import medoids
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import Stage, sampled_file, stage_dir, vocabulary_file


@dataclass
class CollapseConfig:
    data: DataConfig = MISSING
    # The `Distance` the medoid is taken under, built with `hydra.utils.instantiate`.
    distance: Any = MISSING


ConfigStore.instance().store(name='collapse_schema', node=CollapseConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='collapse')
def main(config: CollapseConfig) -> None:
    """Keep only the medoid of the sampled suffixes of every test prefix."""
    name = config.data.name
    sampled = sampled_file(name)
    path = stage_dir(name, Stage.COLLAPSE) / 'medoid.parquet'
    distance = instantiate(config.distance)
    banner(
        f'Collapsing the sampled suffixes of "{name}"',
        {
            'sampled': sampled,
            'distance': type(distance).__name__,
            'medoid': f'1 draw per prefix, written to {path}',
        },
    )
    with step('Taking the medoid of every prefix'):
        vocabulary = Vocabulary.load(vocabulary_file(name))
        collapsed = medoids(Decoding.load(sampled, vocabulary), distance=distance)
        collapsed.save(path, vocabulary)
        print(f'  {len(collapsed.prefixes):,} prefixes', flush=True)


if __name__ == '__main__':
    main()
