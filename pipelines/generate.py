import hydra
import torch
from omegaconf import DictConfig

from pipelines.helpers.console import banner, step
from pipelines.helpers.paths import split_file, stage_dir, vocabulary_file
from pipelines.helpers.runtime import deterministic
from src.datasets.dataset import EncodedLog
from src.datasets.vocabulary import Vocabulary
from src.inference.generate import generate
from src.logs.io import read_split
from src.logs.keys import Split
from src.models.u_ed_lstm import UEDLSTM


def run(config: DictConfig) -> None:
    """Draw sampled and greedy suffixes for every scored test prefix from the trained model."""
    deterministic(config.seed)
    device = torch.device(config.device)
    name = config.data.name
    checkpoint = stage_dir(name, 'train') / 'model.pt'
    output = stage_dir(name, 'generate')
    banner(
        f'Generating suffixes on "{name}"',
        {
            'checkpoint': checkpoint,
            'device': device,
            'sampled': f'{config.num_samples} draws per prefix',
            'greedy': '1 draw per prefix, dropout off',
            'output': output,
        },
    )
    with step('Loading the model and encoding the test split'):
        model = UEDLSTM.load(checkpoint, device=device)
        vocabulary = Vocabulary.load(vocabulary_file(name))
        test_log = EncodedLog.from_split(read_split(split_file(name, Split.TEST)), vocabulary)

    for decoding, sampled in (('sampled', True), ('greedy', False)):
        path = output / f'{decoding}.parquet'
        with step(f'Writing {decoding} suffixes to {path}'):
            prefixes = generate(
                model,
                test_log,
                vocabulary,
                path=path,
                sampled=sampled,
                num_samples=config.num_samples,
                max_rows=config.max_rows,
                device=device,
            )
            print(f'  {prefixes:,} prefixes', flush=True)


@hydra.main(version_base='1.3', config_path='../config', config_name='generate')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
