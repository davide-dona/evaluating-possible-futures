from dataclasses import dataclass

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.datasets.cases import Cases
from possible_futures.datasets.encoded_log import EncodedLog
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.decode import write_draws
from possible_futures.models.empirical import EmpiricalSuffixes
from possible_futures.models.markov import MarkovChain
from possible_futures.models.process import GreedyProcessDecoder, ProcessSampler
from possible_futures.models.sampler import SuffixSampler
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    Stage,
    split_file,
    stage_dir,
    vocabulary_file,
)
from possible_futures.synthetic import SyntheticProcess


@dataclass
class BaselinesConfig:
    data: DataConfig = MISSING
    seed: int = MISSING
    num_draws: int = MISSING
    max_rows: int = MISSING


ConfigStore.instance().store(name='baselines_schema', node=BaselinesConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='baselines')
def main(config: BaselinesConfig) -> None:
    """Write the decodings that need no trained model.

    Every dataset gets the unconditional decoding. A simulated dataset also gets the sampled and
    greedy decodings of its true process, as `oracle`, and of a fitted Markov chain, as `markov`.
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
        train = Cases.load(split_file(name, Split.TRAIN))
        test = Cases.load(split_file(name, Split.TEST))
        train_log = EncodedLog.from_cases(train, vocabulary)
        test_log = EncodedLog.from_cases(test, vocabulary)

    def write(sampler_name: str, sampler: SuffixSampler, *, num_draws: int) -> None:
        path = output / f'{sampler_name}.parquet'
        with step(f'Writing {sampler_name} suffixes to {path}'):
            num_prefixes = write_draws(
                sampler,
                test_log=test_log,
                vocabulary=vocabulary,
                path=path,
                num_draws=num_draws,
                max_steps=int(train.case_lengths.max()),
                max_rows=config.max_rows,
                device=torch.device('cpu'),
            )
            print(f'  {num_prefixes:,} prefixes', flush=True)

    rng = np.random.default_rng(config.seed)
    if simulated:
        processes = {
            'oracle': SyntheticProcess(**config.data.process),
            'markov': MarkovChain.fit(train.traces),
        }
        for process_name, process in processes.items():
            write(
                f'{process_name}_sampled',
                ProcessSampler(process=process, vocabulary=vocabulary, rng=rng),
                num_draws=config.num_draws,
            )
            write(
                f'{process_name}_greedy',
                GreedyProcessDecoder(process=process, vocabulary=vocabulary),
                num_draws=1,
            )
    write(
        'unconditional',
        EmpiricalSuffixes(train_log=train_log, rng=rng),
        num_draws=config.num_draws,
    )


if __name__ == '__main__':
    main()
