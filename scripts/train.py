from dataclasses import dataclass
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf

from possible_futures.cases import load_cases
from possible_futures.datasets.prefix_suffix import PrefixSuffixDataset
from possible_futures.datasets.tokenizer import Tokenizer
from possible_futures.forecasters.u_ed_lstm import UEDLSTMForecaster
from possible_futures.models.u_ed_lstm.model import UEDLSTM
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, format_duration, step
from possible_futures.runs.paths import Split, model_dir, split_file, tokenizer_file
from possible_futures.training.train import TeacherForcing, train


@dataclass
class ModelConfig:
    num_layers: int = MISSING
    hidden_size: int = MISSING
    dropout: float = MISSING
    logit_samples: int = MISSING


@dataclass
class TrainingConfig:
    epochs: int = MISSING
    batch_size: int = MISSING
    target_length: int = MISSING
    weight_penalty: float = MISSING
    grad_clip_norm: float = MISSING
    teacher_forcing: TeacherForcing = MISSING


@dataclass
class TrainConfig:
    data: DataConfig = MISSING
    seed: int = MISSING
    device: str = MISSING
    model: ModelConfig = MISSING
    # A partial `torch.optim.Optimizer`, built with `hydra.utils.instantiate` and given the
    # parameters of the model.
    optimizer: Any = MISSING
    training: TrainingConfig = MISSING


ConfigStore.instance().store(name='train_schema', node=TrainConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='train')
def main(config: TrainConfig) -> None:
    """Train the U-ED-LSTM on a preprocessed dataset and write its final weights."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    name, training = config.data.name, config.training
    banner(
        f'Training the U-ED-LSTM on "{name}"',
        {
            'device': device,
            'epochs': training.epochs,
            'optimizer': f'{config.optimizer._target_}, lr {config.optimizer.lr}',
            'batch size': training.batch_size,
            'weight penalty': training.weight_penalty,
            'output': model_dir(name),
        },
    )
    with step('Loading the train prefixes'):
        tokenizer = Tokenizer.load(tokenizer_file(name))
        dataset = PrefixSuffixDataset(
            load_cases(split_file(name, Split.TRAIN)), tokenizer=tokenizer
        )

    model = UEDLSTM(vocabulary_size=tokenizer.size, **config.model).to(device)
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters', flush=True)

    optimizer = instantiate(config.optimizer)(params=model.parameters())

    for record in train(
        model,
        optimizer,
        dataset=dataset,
        epochs=training.epochs,
        batch_size=training.batch_size,
        target_length=training.target_length,
        weight_penalty=training.weight_penalty,
        grad_clip_norm=training.grad_clip_norm,
        teacher_forcing=OmegaConf.to_object(training.teacher_forcing),
        device=device,
        rng=rng,
        generator=torch.Generator().manual_seed(config.seed),
    ):
        print(
            f'epoch {record.epoch:>3}  teacher forcing {record.teacher_forcing:.3f}  '
            f'train {record.train_loss:.4f}  {format_duration(record.seconds)}',
            flush=True,
        )
    UEDLSTMForecaster(model=model, tokenizer=tokenizer).save(model_dir(name))
    print(f'Wrote the final weights and the tokenizer to {model_dir(name)}')


if __name__ == '__main__':
    main()
