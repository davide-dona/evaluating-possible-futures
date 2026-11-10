import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig

from possible_futures.datasets.dataset import EncodedLog, training_batches, validation_batches
from possible_futures.models.u_ed_lstm import UEDLSTM


@dataclass(frozen=True)
class EpochRecord:
    """The losses of one epoch: cross-entropy per prefix, without the weight penalty."""

    epoch: int
    teacher_forcing: float
    train_loss: float
    validation_loss: float
    seconds: float


def teacher_forcing_ratio(epoch: int, *, start: float, decay: float, every: int) -> float:
    """The teacher forcing probability of a 0-based epoch, multiplied by `decay` every `every`
    epochs."""
    return start * decay ** (epoch // every)


def train(
    model: UEDLSTM,
    *,
    train_log: EncodedLog,
    validation_log: EncodedLog,
    config: DictConfig,
    device: torch.device,
    rng: np.random.Generator,
    on_epoch: Callable[[EpochRecord], None],
) -> None:
    """Train for `config.epochs` epochs, keeping the final weights.

    Args:
        model: The model, on `device`.
        train_log: The train split.
        validation_log: The validation split, whose free-running loss is reported every epoch
            and selects nothing.
        config: The `training` section.
        device: Where batches go.
        rng: Shuffles the training cuts.
        on_epoch: Called with every epoch's record.
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=0.0)
    for epoch in range(config.epochs):
        started = time.perf_counter()
        ratio = teacher_forcing_ratio(
            epoch,
            start=config.teacher_forcing.start,
            decay=config.teacher_forcing.decay,
            every=config.teacher_forcing.every,
        )
        total, rows = 0.0, 0
        for batch in training_batches(
            train_log,
            batch_size=config.batch_size,
            window=config.window,
            rng=rng,
        ):
            batch = batch.to(device)
            means, log_variances = model(batch, teacher_forcing=ratio)
            data_loss = model.loss(means, log_variances, batch.targets)
            loss = data_loss + config.weight_penalty * model.weight_penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=config.grad_clip_norm
            )
            optimizer.step()
            total += data_loss.item() * len(batch.targets)
            rows += len(batch.targets)
        on_epoch(
            EpochRecord(
                epoch=epoch + 1,
                teacher_forcing=ratio,
                train_loss=total / rows,
                validation_loss=validation_loss(
                    model,
                    validation_log,
                    batch_size=config.batch_size,
                    window=config.window,
                    device=device,
                ),
                seconds=time.perf_counter() - started,
            )
        )


@torch.no_grad()
def validation_loss(
    model: UEDLSTM, log: EncodedLog, *, batch_size: int, window: int, device: torch.device
) -> float:
    """The free-running cross-entropy per prefix over every cut of a split, dropout active."""
    total, rows = 0.0, 0
    for batch in validation_batches(log, batch_size=batch_size, window=window):
        batch = batch.to(device)
        means, log_variances = model(batch, teacher_forcing=0.0)
        total += model.loss(means, log_variances, batch.targets).item() * len(batch.targets)
        rows += len(batch.targets)
    return total / rows
