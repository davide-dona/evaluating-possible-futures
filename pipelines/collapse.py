import hydra
from omegaconf import DictConfig

from pipelines.helpers.console import banner, step
from pipelines.helpers.paths import stage_dir
from src.evaluation.distances import damerau_levenshtein
from src.inference.medoid import write_medoids


def run(config: DictConfig) -> None:
    """Collapse the sampled suffixes of every scored test prefix on their medoid under the
    Damerau-Levenshtein distance, the draw with the highest mean DLS to all of them."""
    name = config.data.name
    sampled = stage_dir(name, 'generate') / 'sampled.parquet'
    path = stage_dir(name, 'collapse') / 'medoid.parquet'
    banner(
        f'Collapsing the sampled suffixes of "{name}"',
        {'sampled': sampled, 'medoid': f'1 draw per prefix, written to {path}'},
    )
    with step('Taking the medoid of every prefix'):
        prefixes = write_medoids(sampled, path=path, distance=damerau_levenshtein)
        print(f'  {prefixes:,} prefixes', flush=True)


@hydra.main(version_base='1.3', config_path='../config', config_name='collapse')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
