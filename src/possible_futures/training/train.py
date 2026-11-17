import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from possible_futures.datasets.encoded_log import EncodedLog
from possible_futures.models.u_ed_lstm import UEDLSTM


@dataclass(slots=True)
class TeacherForcing:
    """A teacher forcing probability that starts at `start` and is multiplied by `decay` every
    `every` epochs."""

    start: float
    decay: float
    every: int

    def ratio(self, epoch: int) -> float:
        """The probability of the 0-indexed `epoch`."""
        return self.start * self.decay ** (epoch // self.every)


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """The loss of one epoch: cross-entropy per prefix, without the weight penalty."""

    epoch: int
    teacher_forcing: float
    train_loss: float
    seconds: float


def train(
    model: UEDLSTM,
    optimizer: torch.optim.Optimizer,
    *,
    train_log: EncodedLog,
    epochs: int,
    batch_size: int,
    window: int,
    weight_penalty: float,
    grad_clip_norm: float,
    teacher_forcing: TeacherForcing,
    device: torch.device,
    rng: np.random.Generator,
) -> Iterator[EpochRecord]:
    """Train for `epochs` epochs, yielding the record of each as it ends.

    Args:
        model: The model, on `device`.
        optimizer: Steps the parameters of `model`.
        train_log: The train split.
        epochs: Passes over the training cuts.
        batch_size: Cuts per optimizer step.
        window: Events after the prefix each cut targets.
        weight_penalty: Weight of the sum of squared parameters in the loss.
        grad_clip_norm: The norm the gradient is clipped to.
        teacher_forcing: The teacher forcing probability of every epoch.
        device: Where batches go.
        rng: Shuffles the training cuts and flips the teacher forcing coins.
    """
    for epoch in range(epochs):
        started = time.perf_counter()
        ratio = teacher_forcing.ratio(epoch)
        total, rows = 0.0, 0
        for batch in train_log.training_batches(batch_size=batch_size, window=window, rng=rng):
            batch = batch.to(device)
            forced = rng.random(size=window - 1) < ratio
            logits = model(batch, forced=forced.tolist())
            loss = model.loss(logits=logits, targets=batch.targets, weight_penalty=weight_penalty)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total += loss.data * len(batch.targets)
            rows += len(batch.targets)
        yield EpochRecord(
            epoch=epoch + 1,
            teacher_forcing=ratio,
            train_loss=total / rows,
            seconds=time.perf_counter() - started,
        )
