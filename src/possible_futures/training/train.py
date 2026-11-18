import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from possible_futures.datasets.collation import PrefixSuffixBatch
from possible_futures.datasets.prefix_suffix import PrefixSuffixDataset
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
    pairs: PrefixSuffixDataset,
    epochs: int,
    batch_size: int,
    target_length: int,
    weight_penalty: float,
    grad_clip_norm: float,
    teacher_forcing: TeacherForcing,
    device: torch.device,
    rng: np.random.Generator,
    generator: torch.Generator,
) -> Iterator[EpochRecord]:
    """Train for `epochs` epochs, yielding the record of each as it ends.

    Args:
        model: The model, on `device`.
        optimizer: Steps the parameters of `model`.
        pairs: The train prefixes and their suffixes, shuffled anew every epoch.
        batch_size: The number of prefixes in a batch.
        target_length: The number of events following each prefix that label it.
        weight_penalty: Weight of the sum of squared parameters in the loss.
        rng: Flips the teacher forcing coins.
        generator: Shuffles the prefixes.
    """
    batches = DataLoader(
        pairs,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PrefixSuffixBatch.collate,
        generator=generator,
    )
    for epoch in range(epochs):
        started = time.perf_counter()
        ratio = teacher_forcing.ratio(epoch=epoch, epochs=epochs)
        total, rows = 0.0, 0
        for batch in batches:
            inputs = batch.prefixes.to(device)
            targets = batch.targets(width=target_length).to(device)
            forced = rng.random(size=targets.shape[1] - 1) < ratio
            logits = model(inputs, targets, forced=forced.tolist())
            loss = model.loss(logits=logits, targets=targets, weight_penalty=weight_penalty)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total += loss.data * len(targets)
            rows += len(targets)
        yield EpochRecord(
            epoch=epoch + 1,
            teacher_forcing=ratio,
            train_loss=total / rows,
            seconds=time.perf_counter() - started,
        )
