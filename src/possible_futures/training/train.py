import csv
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import torch
from torch import nn

from possible_futures.datasets.encoded_log import EncodedLog
from possible_futures.models.u_ed_lstm import UEDLSTM


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """The loss of one epoch: cross-entropy per prefix, without the weight penalty."""

    epoch: int
    teacher_forcing: float
    train_loss: float
    seconds: float


class EpochLog:
    """A CSV file of `EpochRecord` rows, each written and flushed as its epoch ends."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open('w', newline='')
        self._writer = csv.DictWriter(
            self._file, fieldnames=[field.name for field in fields(EpochRecord)]
        )
        self._writer.writeheader()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._file.close()

    def append(self, record: EpochRecord) -> None:
        self._writer.writerow(asdict(record))
        self._file.flush()


def squared_weights(module: nn.Module) -> torch.Tensor:
    """The sum of the squares of every parameter of `module`."""
    return sum(parameter.square().sum() for parameter in module.parameters())


def train(
    model: UEDLSTM,
    *,
    train_log: EncodedLog,
    epochs: int,
    lr: float,
    batch_size: int,
    window: int,
    weight_penalty: float,
    grad_clip_norm: float,
    teacher_forcing_start: float,
    teacher_forcing_decay: float,
    teacher_forcing_every: int,
    device: torch.device,
    rng: np.random.Generator,
    on_epoch: Callable[[EpochRecord], None],
) -> None:
    """Train for `epochs` epochs, keeping the final weights.

    Args:
        model: The model, on `device`.
        train_log: The train split.
        epochs: Passes over the training cuts.
        lr: The learning rate of AdamW.
        batch_size: Cuts per optimizer step.
        window: Events after the prefix each cut targets.
        weight_penalty: Weight of the sum of squared parameters in the loss.
        grad_clip_norm: The norm the gradient is clipped to.
        teacher_forcing_start: The teacher forcing probability of the first epoch.
        teacher_forcing_decay: The factor the probability is multiplied by every
            `teacher_forcing_every` epochs.
        teacher_forcing_every: Epochs between two decays.
        device: Where batches go.
        rng: Shuffles the training cuts and flips the teacher forcing coins.
        on_epoch: Called with every epoch's record.
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=0.0)
    for epoch in range(epochs):
        started = time.perf_counter()
        ratio = teacher_forcing_start * teacher_forcing_decay ** (epoch // teacher_forcing_every)
        total, rows = 0.0, 0
        for batch in train_log.training_batches(batch_size=batch_size, window=window, rng=rng):
            batch = batch.to(device)
            forced = rng.random(size=window - 1) < ratio
            logits = model(batch, forced=forced.tolist())
            data_loss = model.loss(logits=logits, targets=batch.targets)
            loss = data_loss + weight_penalty * squared_weights(model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total += data_loss.item() * len(batch.targets)
            rows += len(batch.targets)
        on_epoch(
            EpochRecord(
                epoch=epoch + 1,
                teacher_forcing=ratio,
                train_loss=total / rows,
                seconds=time.perf_counter() - started,
            )
        )
