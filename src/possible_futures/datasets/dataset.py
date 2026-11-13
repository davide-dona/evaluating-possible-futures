from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple, Self

import numpy as np
import pandas as pd
import torch

from possible_futures.datasets.vocabulary import EOS, PAD, Vocabulary
from possible_futures.logs.keys import ACTIVITY_KEY, CASE_KEY, MAX_PREFIX_KEY, MIN_PREFIX_KEY


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

    def repeat_interleave(self, repeats: int) -> 'Prefixes':
        """Repeat every prefix `repeats` times in a row, `[B, P] -> [B * repeats, P]`."""
        return Prefixes(*(field.repeat_interleave(repeats, dim=0) for field in self))

    def last_events(self) -> torch.Tensor:
        """The last real event of every prefix, `[B, P] -> [B]`."""
        index = (self.length - 1).unsqueeze(dim=1)  # [B] -> [B, 1]
        return self.activities.gather(dim=1, index=index).squeeze(dim=1)


class Batch(NamedTuple):
    """A batch of training prefixes and the events following each of them."""

    prefixes: Prefixes
    targets: torch.Tensor  # int64, the next events, EOS past the case end, [B, W]

    def to(self, device: torch.device) -> 'Batch':
        return Batch(prefixes=self.prefixes.to(device), targets=self.targets.to(device))


@dataclass(frozen=True)
class EncodedLog:
    """A split as flat activity indices, every case a run of consecutive events."""

    activities: np.ndarray  # int64, [E]
    starts: np.ndarray  # int64, the first event of every case, [C]
    lengths: np.ndarray  # int64, [C]
    min_prefix: np.ndarray  # int64, the lowest cut of every case, [C]
    max_prefix: np.ndarray  # int64, the highest cut of every case, [C]
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
            max_prefix=split[MAX_PREFIX_KEY].to_numpy(np.int64)[starts],
            cases=split[CASE_KEY].to_numpy()[starts],
        )

    def cuts(self) -> Cuts:
        """Every cut of every case, `k` from its `min_prefix` to its `max_prefix`."""
        counts = np.clip(self.max_prefix - self.min_prefix + 1, a_min=0, a_max=None)
        case = np.repeat(np.arange(len(self.lengths)), counts)
        # Position of each cut within its case's run of cuts.
        offset = np.arange(len(case)) - np.repeat(np.cumsum(counts) - counts, counts)
        return Cuts(case=case, length=self.min_prefix[case] + offset)

    def prefixes(self, cuts: Cuts) -> Prefixes:
        """The events of every cut's prefix, padded to the longest of them."""
        positions = np.arange(cuts.length.max())  # [P]
        index = self.starts[cuts.case, None] + positions[None, :]  # [B, P]
        real = positions[None, :] < cuts.length[:, None]
        activities = np.where(real, self.activities[np.where(real, index, 0)], PAD)
        return Prefixes(
            activities=torch.from_numpy(activities), length=torch.from_numpy(cuts.length)
        )

    def targets(self, cuts: Cuts, *, window: int) -> torch.Tensor:
        """The `window` events following every cut, EOS from the end of the case on."""
        return torch.from_numpy(self._following(cuts, width=window, fill=EOS))

    def suffixes(self, cuts: Cuts) -> np.ndarray:
        """The whole suffix of every cut, PAD past the end of the case, `[B, L]` with `L` the
        length of the longest suffix."""
        width = int((self.lengths[cuts.case] - cuts.length).max())
        return self._following(cuts, width=width, fill=PAD)

    def _following(self, cuts: Cuts, *, width: int, fill: int) -> np.ndarray:
        """The `width` events following every cut, `fill` from the end of the case on."""
        offsets = cuts.length[:, None] + np.arange(width)[None, :]  # [B, W]
        real = offsets < self.lengths[cuts.case, None]
        index = self.starts[cuts.case, None] + offsets
        return np.where(real, self.activities[np.where(real, index, 0)], fill)


def training_batches(
    log: EncodedLog,
    *,
    batch_size: int,
    window: int,
    rng: np.random.Generator,
) -> Iterator[Batch]:
    """One epoch of training batches, every cut once, in a random order."""
    cuts = log.cuts()
    order = rng.permutation(len(cuts.case))
    for first in range(0, len(order), batch_size):
        selected = cuts.select(order[first : first + batch_size])
        yield Batch(
            prefixes=log.prefixes(selected),
            targets=log.targets(selected, window=window),
        )


def test_prefix_batches(log: EncodedLog, *, batch_size: int) -> Iterator[tuple[Cuts, Prefixes]]:
    """Every scored test prefix once, in prefix length order."""
    cuts = log.cuts()
    order = np.argsort(cuts.length, kind='stable')
    for first in range(0, len(order), batch_size):
        selected = cuts.select(order[first : first + batch_size])
        yield selected, log.prefixes(selected)
