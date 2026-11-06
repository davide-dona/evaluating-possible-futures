import hydra
import numpy as np
from omegaconf import DictConfig

from pipelines.helpers.console import banner, step
from pipelines.helpers.paths import split_file, stage_dir, vocabulary_file
from src.datasets.dataset import EncodedLog
from src.datasets.vocabulary import Vocabulary
from src.inference.baselines import write_baseline
from src.logs.io import read_split
from src.logs.keys import Split
from src.simulation.processes import MarkovChain, SyntheticProcess


def run(config: DictConfig) -> None:
    """Draw the suffixes of every scored test prefix from the true process and from a
    first-order Markov chain fitted on the train split."""
    name = config.data.name
    output = stage_dir(name, 'baselines')
    banner(
        f'Writing the baselines of "{name}"',
        {
            'oracle_sampled': f'{config.num_samples} draws per prefix from the true process',
            'oracle_greedy': '1 draw per prefix, the most probable activity at every step',
            'markov_greedy': 'as oracle_greedy, from a first-order chain fitted on train',
            'output': output,
        },
    )
    with step('Fitting the Markov chain and encoding the test split'):
        vocabulary = Vocabulary.load(vocabulary_file(name))
        chain = MarkovChain.fit(read_split(split_file(name, Split.TRAIN)))
        test_log = EncodedLog.from_split(read_split(split_file(name, Split.TEST)), vocabulary)

    process = SyntheticProcess(**config.data.process)
    rng = np.random.default_rng(config.seed)
    baselines = (
        ('oracle_sampled', process, True),
        ('oracle_greedy', process, False),
        ('markov_greedy', chain, False),
    )
    for decoding, source, sampled in baselines:
        path = output / f'{decoding}.parquet'
        with step(f'Writing {decoding} suffixes to {path}'):
            prefixes = write_baseline(
                source,
                test_log,
                vocabulary,
                path=path,
                sampled=sampled,
                num_samples=config.num_samples,
                rng=rng,
            )
            print(f'  {prefixes:,} prefixes', flush=True)


@hydra.main(version_base='1.3', config_path='../config', config_name='baselines')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
