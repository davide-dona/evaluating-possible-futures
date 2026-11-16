import hydra
import torch
from hydra.core.config_store import ConfigStore

from possible_futures.datasets.encoded_log import EncodedLog
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.sampled import write_sampled_draws
from possible_futures.logs.io import read_max_case_length, read_split
from possible_futures.models.u_ed_lstm import UEDLSTM
from possible_futures.runs.configs import GenerateConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    max_case_length_file,
    model_dir,
    sampled_file,
    split_file,
    vocabulary_file,
)

ConfigStore.instance().store(name='generate_schema', node=GenerateConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='generate')
def main(config: GenerateConfig) -> None:
    """Draw sampled suffixes for every scored test prefix from the trained model."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    name = config.data.name
    model_path, path = model_dir(name), sampled_file(name)
    banner(
        f'Generating suffixes on "{name}"',
        {
            'model': model_path,
            'device': device,
            'sampled': f'{config.num_draws} draws per prefix',
            'output': path,
        },
    )
    with step('Loading the model and encoding the test split'):
        model = UEDLSTM.load(model_path, device=device)
        vocabulary = Vocabulary.load(vocabulary_file(name))
        test_log = EncodedLog.from_split(read_split(split_file(name, Split.TEST)), vocabulary)

    with step(f'Writing sampled suffixes to {path}'):
        num_prefixes = write_sampled_draws(
            model,
            test_log=test_log,
            vocabulary=vocabulary,
            path=path,
            num_draws=config.num_draws,
            max_steps=read_max_case_length(max_case_length_file(name)),
            max_rows=config.max_rows,
            device=device,
        )
        print(f'  {num_prefixes:,} prefixes', flush=True)


if __name__ == '__main__':
    main()
