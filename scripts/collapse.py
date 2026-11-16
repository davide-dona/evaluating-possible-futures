import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.medoid import write_medoids
from possible_futures.runs.configs import CollapseConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import Stage, sampled_file, stage_dir, vocabulary_file

ConfigStore.instance().store(name='collapse_schema', node=CollapseConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='collapse')
def main(config: CollapseConfig) -> None:
    """Collapse the sampled suffixes of every scored test prefix on their medoid, the draw with
    the lowest mean distance to all of them."""
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
        num_prefixes = write_medoids(sampled, vocabulary=vocabulary, path=path, distance=distance)
        print(f'  {num_prefixes:,} prefixes', flush=True)


if __name__ == '__main__':
    main()
