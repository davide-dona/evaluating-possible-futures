from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple, Self

import numpy as np
import pandas as pd
import torch

from possible_futures.datasets.batches import PrefixBatch, TrainingBatch
from possible_futures.datasets.vocabulary import EOS, PAD, Vocabulary
from possible_futures.logs.keys import ACTIVITY_KEY, CASE_KEY, MAX_PREFIX_KEY, MIN_PREFIX_KEY


class Cuts(NamedTuple):
    """Prefixes of an `EncodedLog`, each as the case it is cut from and the events it holds."""

    case_indices: np.ndarray  # int64, [N]
    prefix_lengths: np.ndarray  # int64, [N]

    def select(self, index: np.ndarray) -> 'Cuts':
        return Cuts(
            case_indices=self.case_indices[index], prefix_lengths=self.prefix_lengths[index]
        )


@dataclass(frozen=True, slots=True)
class EncodedLog:
    """A split as flat activity indices, every case a run of consecutive events."""

    activities: np.ndarray  # int64, [E]
    case_ids: np.ndarray  # object, [C]
    case_starts: np.ndarray  # int64, the position of the first event of every case, [C]
    case_lengths: np.ndarray  # int64, [C]
    min_prefix_lengths: np.ndarray  # int64, the lowest cut of every case, [C]
    max_prefix_lengths: np.ndarray  # int64, the highest cut of every case, [C]

    @classmethod
    def from_split(cls, split: pd.DataFrame, vocabulary: Vocabulary) -> Self:
        """Encode a split read by `read_split`, its cases consecutive and in time order."""
        first = split[CASE_KEY].ne(split[CASE_KEY].shift()).to_numpy()
        case_starts = np.flatnonzero(first)
        return cls(
            activities=vocabulary.encode(split[ACTIVITY_KEY]),
            case_ids=split[CASE_KEY].to_numpy()[case_starts],
            case_starts=case_starts,
            case_lengths=np.diff(case_starts, append=len(split)),
            min_prefix_lengths=split[MIN_PREFIX_KEY].to_numpy(np.int64)[case_starts],
            max_prefix_lengths=split[MAX_PREFIX_KEY].to_numpy(np.int64)[case_starts],
        )

    def cuts(self) -> Cuts:
        """Every cut of every case, from its lowest to its highest prefix length."""
        counts = np.clip(self.max_prefix_lengths - self.min_prefix_lengths + 1, a_min=0, a_max=None)
        case_indices = np.repeat(np.arange(len(self.case_lengths)), counts)
        # Position of each cut within its case's run of cuts.
        offsets = np.arange(len(case_indices)) - np.repeat(np.cumsum(counts) - counts, counts)
        return Cuts(
            case_indices=case_indices,
            prefix_lengths=self.min_prefix_lengths[case_indices] + offsets,
        )

    def prefixes(self, cuts: Cuts) -> PrefixBatch:
        """The events of every cut's prefix, padded to the longest of them."""
        positions = np.arange(cuts.prefix_lengths.max())  # [P]
        index = self.case_starts[cuts.case_indices, None] + positions[None, :]  # [B, P]
        real = positions[None, :] < cuts.prefix_lengths[:, None]
        activities = np.where(real, self.activities[np.where(real, index, 0)], PAD)
        return PrefixBatch(
            activities=torch.from_numpy(activities),
            lengths=torch.from_numpy(cuts.prefix_lengths),
        )

    def targets(self, cuts: Cuts, *, window: int) -> torch.Tensor:
        """The `window` events following every cut, EOS from the end of the case on."""
        return torch.from_numpy(self._following(cuts, width=window, fill=EOS))

    def suffixes(self, cuts: Cuts) -> np.ndarray:
        """The whole suffix of every cut, PAD past the end of the case, `[B, L]` with `L` the
        length of the longest suffix."""
        width = int((self.case_lengths[cuts.case_indices] - cuts.prefix_lengths).max())
        return self._following(cuts, width=width, fill=PAD)

    def training_batches(
        self, *, batch_size: int, window: int, rng: np.random.Generator
    ) -> Iterator[TrainingBatch]:
        """One epoch of training batches, every cut once, in a random order."""
        cuts = self.cuts()
        order = rng.permutation(len(cuts.case_indices))
        for first in range(0, len(order), batch_size):
            selected = cuts.select(order[first : first + batch_size])
            yield TrainingBatch(
                prefixes=self.prefixes(selected),
                targets=self.targets(selected, window=window),
            )

    def cut_batches_by_length(self, *, batch_size: int) -> Iterator[Cuts]:
        """Every cut once, in batches of `batch_size`, in prefix length order."""
        cuts = self.cuts()
        order = np.argsort(cuts.prefix_lengths, kind='stable')
        for first in range(0, len(order), batch_size):
            yield cuts.select(order[first : first + batch_size])

    def _following(self, cuts: Cuts, *, width: int, fill: int) -> np.ndarray:
        """The `width` events following every cut, `fill` from the end of the case on."""
        offsets = cuts.prefix_lengths[:, None] + np.arange(width)[None, :]  # [B, W]
        real = offsets < self.case_lengths[cuts.case_indices, None]
        index = self.case_starts[cuts.case_indices, None] + offsets
        return np.where(real, self.activities[np.where(real, index, 0)], fill)
