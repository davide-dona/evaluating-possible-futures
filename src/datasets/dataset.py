from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple, Self

import numpy as np
import pandas as pd
import torch

from src.datasets.vocabulary import EOS, PAD, Vocabulary
from src.logs.keys import ACTIVITY_KEY, CASE_KEY, MIN_PREFIX_KEY


@dataclass(frozen=True)
class EncodedLog:
    """A split as flat activity indices, every case a run of consecutive events."""

    activities: np.ndarray  # int64, [E]
    starts: np.ndarray  # int64, the first event of every case, [C]
    lengths: np.ndarray  # int64, [C]
    min_prefix: np.ndarray  # int64, the lowest cut of every case, [C]
    cases: np.ndarray  # object, the case identifiers, [C]

    @classmethod
    def from_split(cls, split: pd.DataFrame, vocabulary: Vocabulary) -> Self:
        """Encode a split read by `read_split`, its cases consecutive and in time order."""
        first = split[CASE_KEY].ne(split[CASE_KEY].shift()).to_numpy()
        starts = np.flatnonzero(first)
        return cls(
            activities=vocabulary.encode(split[ACTIVITY_KEY]),
            starts=starts,
            lengths=np.diff(starts, append=len(split)),
            min_prefix=split[MIN_PREFIX_KEY].to_numpy(np.int64)[starts],
            cases=split[CASE_KEY].to_numpy()[starts],
        )


class Cuts(NamedTuple):
    """Prefixes as the case they are cut from and how many events they hold."""

    case: np.ndarray  # int64, [N]
    length: np.ndarray  # int64, [N]

    def select(self, index: np.ndarray) -> 'Cuts':
        return Cuts(case=self.case[index], length=self.length[index])


class Prefixes(NamedTuple):
    """A batch of prefixes, PAD past each one's length."""

    activities: torch.Tensor  # int64, [B, P]
    length: torch.Tensor  # int64, [B]

    def to(self, device: torch.device) -> 'Prefixes':
        return Prefixes(*(field.to(device) for field in self))


class Batch(NamedTuple):
    """A batch of training prefixes and the events following each of them."""

    prefixes: Prefixes
    targets: torch.Tensor  # int64, the next events, EOS past the case end, [B, W]

    def to(self, device: torch.device) -> 'Batch':
        return Batch(prefixes=self.prefixes.to(device), targets=self.targets.to(device))


def every_cut(log: EncodedLog, *, from_min_prefix: bool) -> Cuts:
    """Every cut leaving a non-empty suffix: `k` from 1, or from each case's `min_prefix` when
    `from_min_prefix`, up to the case length minus one."""
    lowest = log.min_prefix if from_min_prefix else np.ones_like(log.lengths)
    counts = np.clip(log.lengths - lowest, a_min=0, a_max=None)
    case = np.repeat(np.arange(len(log.lengths)), counts)
    # Position of each cut within its case's run of cuts.
    offset = np.arange(len(case)) - np.repeat(np.cumsum(counts) - counts, counts)
    return Cuts(case=case, length=lowest[case] + offset)


def gather_prefixes(log: EncodedLog, cuts: Cuts) -> Prefixes:
    """The events of every cut's prefix, padded to the longest of them."""
    positions = np.arange(cuts.length.max())  # [P]
    index = log.starts[cuts.case, None] + positions[None, :]  # [B, P]
    real = positions[None, :] < cuts.length[:, None]
    activities = np.where(real, log.activities[np.where(real, index, 0)], PAD)
    return Prefixes(activities=torch.from_numpy(activities), length=torch.from_numpy(cuts.length))


def gather_targets(log: EncodedLog, cuts: Cuts, *, window: int) -> torch.Tensor:
    """The `window` events following every cut, EOS from the end of the case on."""
    offsets = cuts.length[:, None] + np.arange(window)[None, :]  # [B, W]
    real = offsets < log.lengths[cuts.case, None]
    index = log.starts[cuts.case, None] + offsets
    return torch.from_numpy(np.where(real, log.activities[np.where(real, index, 0)], EOS))


def training_batches(
    log: EncodedLog,
    *,
    batch_size: int,
    window: int,
    bucket_batches: int,
    rng: np.random.Generator,
) -> Iterator[Batch]:
    """One epoch of shuffled training batches, every cut once, bucketed by prefix length.

    The cuts are shuffled, sorted by prefix length within chunks of `bucket_batches` batches, so
    a batch holds prefixes of similar lengths, and the batches are shuffled again.
    """
    cuts = every_cut(log, from_min_prefix=False)
    order = rng.permutation(len(cuts.case))
    chunk = batch_size * bucket_batches
    for first in range(0, len(order), chunk):
        part = order[first : first + chunk]
        order[first : first + chunk] = part[np.argsort(cuts.length[part], kind='stable')]
    batches = [order[first : first + batch_size] for first in range(0, len(order), batch_size)]
    for index in rng.permutation(len(batches)):
        selected = cuts.select(batches[index])
        yield Batch(
            prefixes=gather_prefixes(log, selected),
            targets=gather_targets(log, selected, window=window),
        )


def validation_batches(log: EncodedLog, *, batch_size: int, window: int) -> Iterator[Batch]:
    """Every cut of a split once, in prefix length order."""
    cuts = every_cut(log, from_min_prefix=False)
    order = np.argsort(cuts.length, kind='stable')
    for first in range(0, len(order), batch_size):
        selected = cuts.select(order[first : first + batch_size])
        yield Batch(
            prefixes=gather_prefixes(log, selected),
            targets=gather_targets(log, selected, window=window),
        )


def test_prefix_batches(log: EncodedLog, *, batch_size: int) -> Iterator[tuple[Cuts, Prefixes]]:
    """Every scored test prefix once, from each case's `min_prefix`, in prefix length order."""
    cuts = every_cut(log, from_min_prefix=True)
    order = np.argsort(cuts.length, kind='stable')
    for first in range(0, len(order), batch_size):
        selected = cuts.select(order[first : first + batch_size])
        yield selected, gather_prefixes(log, selected)
