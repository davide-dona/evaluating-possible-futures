import csv
from dataclasses import asdict, fields

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from pipelines.helpers.console import banner, duration, step
from pipelines.helpers.paths import split_file, stage_dir, vocabulary_file
from src.datasets.dataset import EncodedLog
from src.datasets.vocabulary import Vocabulary
from src.logs.io import read_split
from src.logs.keys import Split
from src.models.u_ed_lstm import UEDLSTM
from src.training.train import EpochRecord, train


def run(config: DictConfig) -> None:
    """Train the U-ED-LSTM on a preprocessed dataset and write its final weights."""
    # Set the random seeds for reproducibility
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    # Set the device and directories
    device = torch.device(config.device)
    name, training = config.data.name, config.training
    output = stage_dir(name, 'train')
    banner(
        f'Training the U-ED-LSTM on "{name}"',
        {
            'device': device,
            'epochs': training.epochs,
            'optimizer': f'AdamW, lr {training.lr}, batch {training.batch_size}, '
            f'weight penalty {training.weight_penalty}',
            'output': output,
        },
    )
    with step('Encoding the train and validation splits'):
        vocabulary = Vocabulary.load(vocabulary_file(name))
        train_log = EncodedLog.from_split(read_split(split_file(name, Split.TRAIN)), vocabulary)
        validation_log = EncodedLog.from_split(read_split(split_file(name, Split.VAL)), vocabulary)

    model = UEDLSTM(
        vocabulary_size=vocabulary.size,
        num_activities=len(vocabulary.activities),
        **config.model,
    ).to(device)
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters', flush=True)

    output.mkdir(parents=True, exist_ok=True)
    with (output / 'log.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in fields(EpochRecord)])
        writer.writeheader()

        def on_epoch(record: EpochRecord) -> None:
            writer.writerow(asdict(record))
            file.flush()
            print(
                f'epoch {record.epoch:>3}  teacher forcing {record.teacher_forcing:.3f}  '
                f'train {record.train_loss:.4f}  validation {record.validation_loss:.4f}  '
                f'{duration(record.seconds)}',
                flush=True,
            )

        train(
            model,
            train_log=train_log,
            validation_log=validation_log,
            config=training,
            device=device,
            rng=rng,
            on_epoch=on_epoch,
        )
    model.save(output / 'model.pt')
    print(f'Wrote the final weights to {output / "model.pt"}')


@hydra.main(version_base='1.3', config_path='../config', config_name='train')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
