import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from possible_futures.datasets.collation import PrefixSuffixBatch
from possible_futures.datasets.prefix_suffix import PrefixSuffixDataset
from possible_futures.models.u_ed_lstm.model import UEDLSTM


@dataclass(slots=True)
class TeacherForcing:
    """How often the decoder reads the true previous token instead of its own prediction, over
    training. The probability falls in a straight line from `start` to `end` over the first
    `fraction` of the epochs, then stays at `end`: the model first learns from correct inputs,
    then from its own mistakes, as when sampling.

    Attributes:
        start: The probability in the first epoch.
        end: The probability once the decline is over.
        fraction: The share of the epochs the decline lasts.
    """

    start: float
    end: float
    fraction: float

    def probability(self, epoch: int, epochs: int) -> float:
        """The teacher forcing probability of one epoch.

        Args:
            epoch: The epoch, counted from 0.
            epochs: The number of epochs of the training.

        Returns:
            float: The probability that a decoding step reads the true previous token.
        """
        progress = epoch / (self.fraction * epochs)
        return max(self.end, self.start - (self.start - self.end) * progress)


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """What happened in one epoch of training.

    Attributes:
        epoch: The epoch, counted from 1.
        teacher_forcing: The teacher forcing probability of the epoch.
        train_loss: The mean cross-entropy per train prefix, without the weight penalty.
        seconds: How long the epoch took.
    """

    epoch: int
    teacher_forcing: float
    train_loss: float
    seconds: float


def train(
    model: UEDLSTM,
    optimizer: torch.optim.Optimizer,
    *,
    dataset: PrefixSuffixDataset,
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
        dataset: The train prefixes and their suffixes, shuffled anew every epoch.
        epochs: The number of passes over the dataset.
        batch_size: The number of prefixes in a batch.
        target_length: The number of tokens following each prefix the model learns to predict.
        weight_penalty: Weight of the sum of squared parameters in the loss.
        grad_clip_norm: The largest norm of the gradient, which is rescaled when longer.
        teacher_forcing: The teacher forcing probability of every epoch.
        device: The device the model is on.
        rng: Decides which decoding steps are teacher forced.
        generator: Shuffles the prefixes.

    Returns:
        Iterator[EpochRecord]: The record of every epoch, as it ends.
    """
    batches = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PrefixSuffixBatch.collate,
        generator=generator,
    )
    for epoch in range(epochs):
        started = time.perf_counter()
        probability = teacher_forcing.probability(epoch=epoch, epochs=epochs)
        total, rows = 0.0, 0
        for batch in batches:
            inputs = batch.prefixes.to(device)
            targets = batch.targets(length=target_length).to(device)
            forced = rng.random(size=targets.shape[1] - 1) < probability
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
            teacher_forcing=probability,
            train_loss=total / rows,
            seconds=time.perf_counter() - started,
        )
