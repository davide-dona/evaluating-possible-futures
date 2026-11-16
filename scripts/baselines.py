import hydra
import numpy as np
from hydra.core.config_store import ConfigStore

from possible_futures.datasets.encoded_log import EncodedLog
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.baselines import (
    write_greedy_process_draws,
    write_sampled_process_draws,
    write_unconditional_draws,
)
from possible_futures.logs.io import read_max_case_length, read_split
from possible_futures.processes.markov import MarkovChain
from possible_futures.processes.synthetic import SyntheticProcess
from possible_futures.runs.configs import BaselinesConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    Stage,
    max_case_length_file,
    split_file,
    stage_dir,
    vocabulary_file,
)

ConfigStore.instance().store(name='baselines_schema', node=BaselinesConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='baselines')
def main(config: BaselinesConfig) -> None:
    """Write the decodings that need no trained model.

    Every dataset gets the unconditional decoding, drawn from the train split whatever the
    prefix. A dataset simulated from a known process also gets the sampled and greedy decodings
    of that process, as `oracle`, and of a first-order Markov chain fitted on the train split,
    as `markov`.
    """
    name = config.data.name
    output = stage_dir(name, Stage.BASELINES)
    simulated = config.data.process is not None
    draws = f'{config.num_draws} draws per prefix'
    banner(
        f'Writing the baselines of "{name}"',
        (
            {
                'oracle': 'the true process',
                'markov': 'a first-order chain fitted on train',
                'decoders': f'sampled, {draws}; greedy, the most probable activity at every step',
            }
            if simulated
            else {}
        )
        | {'unconditional': f'{draws} from the suffixes of train', 'output': output},
    )
    with step('Encoding the train and test splits'):
        vocabulary = Vocabulary.load(vocabulary_file(name))
        train = read_split(split_file(name, Split.TRAIN))
        train_log = EncodedLog.from_split(train, vocabulary)
        test_log = EncodedLog.from_split(read_split(split_file(name, Split.TEST)), vocabulary)

    rng = np.random.default_rng(config.seed)
    if simulated:
        max_steps = read_max_case_length(max_case_length_file(name))
        processes = {
            'oracle': SyntheticProcess(**config.data.process),
            'markov': MarkovChain.fit(train),
        }
        for process_name, process in processes.items():
            path = output / f'{process_name}_sampled.parquet'
            with step(f'Writing {process_name}_sampled suffixes to {path}'):
                num_prefixes = write_sampled_process_draws(
                    process,
                    test_log=test_log,
                    vocabulary=vocabulary,
                    path=path,
                    num_draws=config.num_draws,
                    max_steps=max_steps,
                    rng=rng,
                )
                print(f'  {num_prefixes:,} prefixes', flush=True)
            path = output / f'{process_name}_greedy.parquet'
            with step(f'Writing {process_name}_greedy suffixes to {path}'):
                num_prefixes = write_greedy_process_draws(
                    process,
                    test_log=test_log,
                    vocabulary=vocabulary,
                    path=path,
                    max_steps=max_steps,
                )
                print(f'  {num_prefixes:,} prefixes', flush=True)

    path = output / 'unconditional.parquet'
    with step(f'Writing unconditional suffixes to {path}'):
        num_prefixes = write_unconditional_draws(
            train_log,
            test_log=test_log,
            vocabulary=vocabulary,
            path=path,
            num_draws=config.num_draws,
            rng=rng,
        )
        print(f'  {num_prefixes:,} prefixes', flush=True)


if __name__ == '__main__':
    main()
