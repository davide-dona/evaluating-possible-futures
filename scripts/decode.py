from dataclasses import dataclass
from enum import StrEnum

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.datasets.cases import Cases
from possible_futures.datasets.encoding import EncodedPrefixes
from possible_futures.datasets.prefixes import Prefixes
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.decode import write_draws
from possible_futures.models.decoder import Decoder
from possible_futures.models.markov import MarkovChain
from possible_futures.models.process import GreedyDecoder, SamplingDecoder
from possible_futures.models.u_ed_lstm import UEDLSTM
from possible_futures.models.unconditional import UnconditionalDecoder
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    decoding_file,
    model_dir,
    split_file,
    test_prefixes_file,
    vocabulary_file,
)
from possible_futures.synthetic import SyntheticProcess


class DecoderModel(StrEnum):
    """What the draws come from."""

    U_ED_LSTM = 'u_ed_lstm'  # the trained model
    ORACLE = 'oracle'  # the true process of a simulated dataset
    MARKOV = 'markov'  # a first-order chain fitted on train
    UNCONDITIONAL = 'unconditional'  # the suffixes of train, whatever the prefix


class Strategy(StrEnum):
    SAMPLED = 'sampled'
    GREEDY = 'greedy'  # the most probable activity at every step


@dataclass
class DecoderConfig:
    # The name of the decoding, the stem of its file.
    name: str = MISSING
    model: DecoderModel = MISSING
    strategy: Strategy = MISSING


@dataclass
class DecodeConfig:
    data: DataConfig = MISSING
    decoder: DecoderConfig = MISSING
    seed: int = MISSING
    device: str = MISSING
    num_draws: int = MISSING
    max_rows: int = MISSING


ConfigStore.instance().store(name='decode_schema', node=DecodeConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='decode')
def main(config: DecodeConfig) -> None:
    """Draw suffixes for every test prefix from one decoder and write them as a decoding."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    name, decoder_config = config.data.name, config.decoder
    path = decoding_file(name, decoder_config.name)
    banner(
        f'Decoding "{name}" as {decoder_config.name}',
        {
            'model': decoder_config.model,
            'strategy': decoder_config.strategy,
            'device': device,
            'draws': f'{config.num_draws} per prefix',
            'output': path,
        },
    )
    with step('Building the decoder and encoding the test split'):
        vocabulary = Vocabulary.load(vocabulary_file(name))
        train = Cases.load(split_file(name, Split.TRAIN))
        test = EncodedPrefixes.build(
            Cases.load(split_file(name, Split.TEST)),
            Prefixes.load(test_prefixes_file(name)),
            vocabulary=vocabulary,
        )
        decoder = build_decoder(
            decoder_config,
            data=config.data,
            train=train,
            vocabulary=vocabulary,
            rng=rng,
            device=device,
        )

    with step(f'Writing the draws to {path}'):
        num_prefixes = write_draws(
            decoder,
            test=test,
            vocabulary=vocabulary,
            path=path,
            num_draws=config.num_draws,
            max_steps=int(train.case_lengths.max()),
            max_rows=config.max_rows,
            device=device,
        )
        print(f'  {num_prefixes:,} prefixes', flush=True)


def build_decoder(
    config: DecoderConfig,
    *,
    data: DataConfig,
    train: Cases,
    vocabulary: Vocabulary,
    rng: np.random.Generator,
    device: torch.device,
) -> Decoder:
    """The decoder `config` asks for, built from the train split where it needs one."""
    match config.model, config.strategy:
        case DecoderModel.U_ED_LSTM, Strategy.SAMPLED:
            return UEDLSTM.load(model_dir(data.name), device=device)
        case DecoderModel.UNCONDITIONAL, Strategy.SAMPLED:
            prefixes = EncodedPrefixes.build(train, Prefixes.every(train), vocabulary=vocabulary)
            return UnconditionalDecoder(train=prefixes, rng=rng)
        case DecoderModel.ORACLE | DecoderModel.MARKOV, strategy:
            if config.model is DecoderModel.MARKOV:
                process = MarkovChain.fit(train.traces)
            elif data.process is None:
                raise ValueError(f'"{data.name}" is not simulated, so it has no oracle')
            else:
                process = SyntheticProcess(**data.process)
            if strategy is Strategy.SAMPLED:
                return SamplingDecoder(process=process, vocabulary=vocabulary, rng=rng)
            return GreedyDecoder(process=process, vocabulary=vocabulary)
    raise ValueError(f'The {config.model} decoder has no {config.strategy} strategy')


if __name__ == '__main__':
    main()
