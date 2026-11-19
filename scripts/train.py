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
    """The size of the U-ED-LSTM.

    Attributes:
        num_layers: Layers of the encoder and of the decoder.
        hidden_size: Width of every LSTM state, `H`.
        dropout: Probability of dropping a gate input or hidden feature.
        logit_samples: Logit vectors sampled at every position to average the loss, `K`.
    """

    num_layers: int = MISSING
    hidden_size: int = MISSING
    dropout: float = MISSING
    logit_samples: int = MISSING


@dataclass
class TrainingConfig:
    """How the model is trained.

    Attributes:
        epochs: The number of passes over the train prefixes.
        batch_size: The number of prefixes in a batch.
        target_length: The number of tokens following each prefix the model learns to predict.
        weight_penalty: Weight of the sum of squared parameters in the loss.
        grad_clip_norm: The largest norm of the gradient, which is rescaled when longer.
        teacher_forcing: The teacher forcing probability of every epoch.
    """

    epochs: int = MISSING
    batch_size: int = MISSING
    target_length: int = MISSING
    weight_penalty: float = MISSING
    grad_clip_norm: float = MISSING
    teacher_forcing: TeacherForcing = MISSING


@dataclass
class TrainConfig:
    """The settings of the train stage.

    Attributes:
        data: The dataset.
        seed: Seeds the initialization, the shuffling, the dropout and the teacher forcing.
        device: The device the model trains on, such as `cuda:0`.
        model: The size of the model.
        optimizer: A partial `torch.optim.Optimizer`, built with `hydra.utils.instantiate` and
            given the parameters of the model.
        training: How the model is trained.
    """

    data: DataConfig = MISSING
    seed: int = MISSING
    device: str = MISSING
    model: ModelConfig = MISSING
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

    model = UEDLSTM(vocabulary_size=tokenizer.vocabulary_size, **config.model).to(device)
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
