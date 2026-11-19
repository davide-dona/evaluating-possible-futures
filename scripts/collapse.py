from dataclasses import dataclass
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from possible_futures.datasets.tokenizer import Tokenizer
from possible_futures.inference.decoding import load_decoding, save_decoding
from possible_futures.inference.medoid import medoids
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import decoding_file, tokenizer_file


@dataclass
class CollapseConfig:
    data: DataConfig = MISSING
    # The name of the decoding collapsed, the stem of its file.
    decoding: str = MISSING
    # The name of the collapsed decoding, the stem of its file.
    name: str = MISSING
    # The `Distance` the medoid is taken under, built with `hydra.utils.instantiate`.
    distance: Any = MISSING


ConfigStore.instance().store(name='collapse_schema', node=CollapseConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='collapse')
def main(config: CollapseConfig) -> None:
    """Keep only the medoid of the sampled suffixes of every test prefix."""
    name = config.data.name
    source = decoding_file(name, config.decoding)
    path = decoding_file(name, config.name)
    distance = instantiate(config.distance)
    banner(
        f'Collapsing the {config.decoding} suffixes of "{name}"',
        {
            'decoding': source,
            'distance': type(distance).__name__,
            config.name: f'1 draw per prefix, written to {path}',
        },
    )
    with step('Taking the medoid of every prefix'):
        collapsed = medoids(
            load_decoding(source),
            tokenizer=Tokenizer.load(tokenizer_file(name)),
            distance=distance,
        )
        save_decoding(collapsed, path=path)
        print(f'  {len(collapsed):,} prefixes', flush=True)


if __name__ == '__main__':
    main()
