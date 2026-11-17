import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from possible_futures.datasets.encoding import EncodedPrefixes
from possible_futures.models.u_ed_lstm import UEDLSTM


@dataclass(slots=True)
class TeacherForcing:
    """A teacher forcing probability that falls in a straight line from `start` to `end` over the
    first `fraction` of the epochs, then stays at `end`."""

    start: float
    end: float
    fraction: float

    def ratio(self, epoch: int, epochs: int) -> float:
        """The probability of the 0-indexed `epoch` out of `epochs`."""
        progress = epoch / (self.fraction * epochs)
        return max(self.end, self.start - (self.start - self.end) * progress)


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
    prefixes: EncodedPrefixes,
    epochs: int,
    batch_size: int,
    target_length: int,
    weight_penalty: float,
    grad_clip_norm: float,
    teacher_forcing: TeacherForcing,
    device: torch.device,
    rng: np.random.Generator,
) -> Iterator[EpochRecord]:
    """Train for `epochs` epochs, yielding the record of each as it ends.

    Args:
        model: The model, on `device`.
        prefixes: The train prefixes.
        optimizer: Steps the parameters of `model`.
        batch_size: Prefixes per optimizer step.
        target_length: Events following every prefix that label it.
        weight_penalty: Weight of the sum of squared parameters in the loss.
        rng: Shuffles the training prefixes and flips the teacher forcing coins.
    """
    for epoch in range(epochs):
        started = time.perf_counter()
        ratio = teacher_forcing.ratio(epoch=epoch, epochs=epochs)
        total, rows = 0.0, 0
        order = rng.permutation(len(prefixes))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            inputs = prefixes.inputs(index).to(device)
            targets = prefixes.targets(index, width=target_length).to(device)
            forced = rng.random(size=target_length - 1) < ratio
            logits = model(inputs, targets, forced=forced.tolist())
            loss = model.loss(logits=logits, targets=targets, weight_penalty=weight_penalty)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total += loss.data * len(index)
            rows += len(index)
        yield EpochRecord(
            epoch=epoch + 1,
            teacher_forcing=ratio,
            train_loss=total / rows,
            seconds=time.perf_counter() - started,
        )
