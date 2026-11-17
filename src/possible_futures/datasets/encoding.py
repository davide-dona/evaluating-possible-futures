from dataclasses import dataclass
from itertools import chain
from typing import NamedTuple, Self

import numpy as np
import pandas as pd
import torch

from possible_futures.datasets.cases import Cases
from possible_futures.datasets.prefixes import Prefixes
from possible_futures.datasets.vocabulary import SpecialToken, Vocabulary


class PrefixTokens(NamedTuple):
    """The tokens of some prefixes, padded to the longest of them, PAD past each one's length."""

    tokens: torch.Tensor  # int64, [B, P]
    lengths: torch.Tensor  # int64, [B]

    def to(self, device: torch.device) -> 'PrefixTokens':
        """Move the prefix tokens to a device."""
        return PrefixTokens(*(field.to(device) for field in self))

    def repeat_interleave(self, num_copies: int) -> 'PrefixTokens':
        """Repeat every prefix `num_copies` times in a row, `[B, P] -> [B * num_copies, P]`."""
        return PrefixTokens(*(field.repeat_interleave(num_copies, dim=0) for field in self))

    def last_events(self) -> torch.Tensor:
        """The last real event of every prefix, `[B, P] -> [B]`."""
        index = (self.lengths - 1).unsqueeze(dim=1)  # [B] -> [B, 1]
        return self.tokens.gather(dim=1, index=index).squeeze(dim=1)


@dataclass(frozen=True, slots=True)
class EncodedPrefixes:
    """Prefixes of some cases, with the cases encoded as tokens.

    Every method takes `index`, the positions of some of the prefixes in `prefixes`, `[B]`.
    """

    prefixes: Prefixes
    case_indices: np.ndarray  # int64, the row of `tokens` holding each prefix's case, [N]
    tokens: np.ndarray  # int64, one case per row, PAD past its end, [C, L]
    case_lengths: np.ndarray  # int64, [C]

    def __len__(self) -> int:
        return len(self.prefixes)

    @classmethod
    def build(cls, cases: Cases, prefixes: Prefixes, vocabulary: Vocabulary) -> Self:
        """Encode `cases` and locate the case of every one of `prefixes` among them."""
        case_indices = pd.Index(cases.case_ids).get_indexer(prefixes.case_ids)
        if (case_indices < 0).any():
            missing = prefixes.case_ids[case_indices < 0]
            raise ValueError(f'{len(missing)} prefixes belong to no case, such as {missing[0]!r}')
        return cls(
            prefixes=prefixes,
            case_indices=case_indices.astype(np.int64),
            tokens=_encode(cases, vocabulary),
            case_lengths=cases.case_lengths,
        )

    def inputs(self, index: np.ndarray) -> PrefixTokens:
        """The events of the prefixes, padded to the longest of them."""
        lengths = self.prefixes.lengths[index]
        width = lengths.max()
        tokens = self.tokens[self.case_indices[index], :width]  # [B, P]
        real = np.arange(width)[None, :] < lengths[:, None]
        return PrefixTokens(
            tokens=torch.from_numpy(np.where(real, tokens, SpecialToken.PAD)),
            lengths=torch.from_numpy(lengths),
        )

    def targets(self, index: np.ndarray, *, width: int) -> torch.Tensor:
        """The `width` events following each prefix, its training labels, EOS from the end of the
        case on, `[B, width]`."""
        return torch.from_numpy(self._following(index, width=width, fill=SpecialToken.EOS))

    def suffixes(self, index: np.ndarray) -> np.ndarray:
        """The whole suffix of each prefix, PAD past the end of the case, `[B, L]` with `L` the
        length of the longest suffix."""
        remaining = self.case_lengths[self.case_indices[index]] - self.prefixes.lengths[index]
        return self._following(index, width=int(remaining.max()), fill=SpecialToken.PAD)

    def _following(self, index: np.ndarray, *, width: int, fill: int) -> np.ndarray:
        """The `width` events following each prefix, `fill` from the end of the case on."""
        rows = self.case_indices[index]
        columns = self.prefixes.lengths[index, None] + np.arange(width)[None, :]  # [B, W]
        real = columns < self.case_lengths[rows, None]
        # Columns past the last one are past the end of the case, so clipping them only
        # reads values that `real` masks out.
        clipped = np.minimum(columns, self.tokens.shape[1] - 1)
        return np.where(real, self.tokens[rows[:, None], clipped], fill)


def _encode(cases: Cases, vocabulary: Vocabulary) -> np.ndarray:
    """The tokens of the cases, one case per row, PAD past its end, `[C, L]` with `L` the length
    of the longest case."""
    lengths = cases.case_lengths
    # Fill each row with PAD
    tokens = np.full(shape=(len(cases), lengths.max()), fill_value=SpecialToken.PAD, dtype=np.int64)
    # Fill back the real events with their tokens.
    rows = np.repeat(np.arange(len(cases)), lengths)  # [E]
    positions = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    tokens[rows, positions] = vocabulary.encode(chain.from_iterable(cases.traces))
    return tokens
