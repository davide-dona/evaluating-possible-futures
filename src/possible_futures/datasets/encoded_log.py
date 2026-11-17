from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Self

import numpy as np
import torch

from possible_futures.datasets.batches import PrefixBatch, TrainingBatch
from possible_futures.datasets.cases import Cases, Cuts
from possible_futures.datasets.vocabulary import SpecialToken, Vocabulary


@dataclass(frozen=True, slots=True)
class EncodedLog:
    """Cases as one row of tokens per case, PAD past the end of the case."""

    cases: Cases
    tokens: np.ndarray  # int64, [C, L] with `L` the length of the longest case

    @classmethod
    def from_cases(cls, cases: Cases, vocabulary: Vocabulary) -> Self:
        lengths = cases.case_lengths
        tokens = np.full(
            shape=(len(cases), lengths.max()), fill_value=SpecialToken.PAD, dtype=np.int64
        )
        rows = np.repeat(np.arange(len(cases)), lengths)  # [E]
        positions = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        tokens[rows, positions] = vocabulary.encode(chain.from_iterable(cases.traces))
        return cls(cases=cases, tokens=tokens)

    def prefixes(self, cuts: Cuts) -> PrefixBatch:
        """The events of every cut's prefix, padded to the longest of them."""
        width = cuts.prefix_lengths.max()
        tokens = self.tokens[cuts.case_indices, :width]  # [B, P]
        real = np.arange(width)[None, :] < cuts.prefix_lengths[:, None]
        return PrefixBatch(
            tokens=torch.from_numpy(np.where(real, tokens, SpecialToken.PAD)),
            lengths=torch.from_numpy(cuts.prefix_lengths),
        )

    def targets(self, cuts: Cuts, *, window: int) -> torch.Tensor:
        """The `window` events following every cut, EOS from the end of the case on."""
        return torch.from_numpy(self._following(cuts, width=window, fill=SpecialToken.EOS))

    def suffixes(self, cuts: Cuts) -> np.ndarray:
        """The whole suffix of every cut, PAD past the end of the case, `[B, L]` with `L` the
        length of the longest suffix."""
        width = int((self.cases.case_lengths[cuts.case_indices] - cuts.prefix_lengths).max())
        return self._following(cuts, width=width, fill=SpecialToken.PAD)

    def training_batches(
        self, *, batch_size: int, window: int, rng: np.random.Generator
    ) -> Iterator[TrainingBatch]:
        """One epoch of training batches, every cut once, in a random order."""
        cuts = self.cases.cuts()
        order = rng.permutation(len(cuts.case_indices))
        for first in range(0, len(order), batch_size):
            selected = cuts.select(order[first : first + batch_size])
            yield TrainingBatch(
                prefixes=self.prefixes(selected),
                targets=self.targets(selected, window=window),
            )

    def cut_batches_by_length(self, *, batch_size: int) -> Iterator[Cuts]:
        """Every cut once, in batches of `batch_size`, in prefix length order."""
        cuts = self.cases.cuts()
        order = np.argsort(cuts.prefix_lengths, kind='stable')
        for first in range(0, len(order), batch_size):
            yield cuts.select(order[first : first + batch_size])

    def _following(self, cuts: Cuts, *, width: int, fill: int) -> np.ndarray:
        """The `width` events following every cut, `fill` from the end of the case on."""
        columns = cuts.prefix_lengths[:, None] + np.arange(width)[None, :]  # [B, W]
        real = columns < self.cases.case_lengths[cuts.case_indices, None]
        # Columns past the last one are past the end of the case, so clipping them only
        # reads values that `real` masks out.
        clipped = np.minimum(columns, self.tokens.shape[1] - 1)
        return np.where(real, self.tokens[cuts.case_indices[:, None], clipped], fill)
