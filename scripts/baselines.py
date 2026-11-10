import hydra
import numpy as np
from omegaconf import DictConfig

from possible_futures.datasets.dataset import EncodedLog
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.baselines import write_baseline, write_unconditional
from possible_futures.logs.io import read_split
from possible_futures.logs.keys import Split
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import split_file, stage_dir, vocabulary_file
from possible_futures.simulation.processes import MarkovChain, SyntheticProcess


def run(config: DictConfig) -> None:
    """Write the forecasts that need no trained model.

    Every dataset gets the unconditional forecast, drawn from the train split whatever the
    prefix. A dataset simulated from a known process also gets the sampled and greedy decodings
    of that process and of a first-order Markov chain fitted on the train split.
    """
    name = config.data.name
    output = stage_dir(name, 'baselines')
    simulated = 'process' in config.data
    draws = f'{config.num_samples} draws per prefix'
    process_fields = {
        'oracle_sampled': f'{draws} from the true process',
        'oracle_greedy': '1 draw per prefix, the most probable activity at every step',
        'markov_sampled': f'{draws} from a first-order chain fitted on train',
        'markov_greedy': 'as oracle_greedy, from the chain',
    }
    banner(
        f'Writing the baselines of "{name}"',
        (process_fields if simulated else {})
        | {'unconditional': f'{draws} from the suffixes of train', 'output': output},
    )
    with step('Encoding the train and test splits'):
        vocabulary = Vocabulary.load(vocabulary_file(name))
        train = read_split(split_file(name, Split.TRAIN))
        train_log = EncodedLog.from_split(train, vocabulary)
        test_log = EncodedLog.from_split(read_split(split_file(name, Split.TEST)), vocabulary)

    rng = np.random.default_rng(config.seed)
    if simulated:
        process = SyntheticProcess(**config.data.process)
        chain = MarkovChain.fit(train)
        baselines = (
            ('oracle_sampled', process, True),
            ('oracle_greedy', process, False),
            ('markov_sampled', chain, True),
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

    path = output / 'unconditional.parquet'
    with step(f'Writing unconditional suffixes to {path}'):
        prefixes = write_unconditional(
            train_log,
            test_log,
            vocabulary,
            path=path,
            num_samples=config.num_samples,
            rng=rng,
        )
        print(f'  {prefixes:,} prefixes', flush=True)


@hydra.main(version_base='1.3', config_path='../config', config_name='baselines')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
