from collections.abc import Sequence
from typing import NamedTuple, Self

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from possible_futures.datasets.prefix_suffix import PrefixSuffix
from possible_futures.datasets.vocabulary import SpecialToken


class PrefixTokens(NamedTuple):
    """The tokens of some prefixes, padded to the longest of them, PAD past each one's length."""

    tokens: torch.Tensor  # int64, [B, P]
    lengths: torch.Tensor  # int64, [B]

    @classmethod
    def pad(cls, prefixes: Sequence[torch.Tensor]) -> Self:
        """Pad prefixes to the longest of them.

        Args:
            prefixes: The tokens of every prefix, int64 `[P_i]`.

        Returns:
            PrefixTokens: `[B, P]` with `P` the length of the longest prefix.
        """
        return cls(
            tokens=pad_sequence(prefixes, batch_first=True, padding_value=SpecialToken.PAD),
            lengths=torch.tensor([len(prefix) for prefix in prefixes], dtype=torch.int64),
        )

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


class PrefixSuffixBatch(NamedTuple):
    """Some prefixes and the suffixes following them."""

    prefixes: PrefixTokens
    suffixes: torch.Tensor  # int64, PAD past each one's end, [B, L]

    @classmethod
    def collate(cls, items: Sequence[PrefixSuffix]) -> Self:
        """Pad items into a batch, as the `collate_fn` of a `DataLoader`.

        Args:
            items: The prefixes and suffixes of the batch.

        Returns:
            PrefixSuffixBatch: The prefixes padded to the longest prefix, the suffixes to the
                longest suffix.
        """
        return cls(
            prefixes=PrefixTokens.pad([item.prefix for item in items]),
            suffixes=pad_sequence(
                [item.suffix for item in items], batch_first=True, padding_value=SpecialToken.PAD
            ),
        )

    def targets(self, width: int) -> torch.Tensor:
        """The `width` events following each prefix, EOS past the end of its case.

        Args:
            width: The number of events kept from each suffix.

        Returns:
            torch.Tensor: int64 `[B, width]`.
        """
        suffixes = self.suffixes[:, :width]  # [B, L] -> [B, min(L, width)]
        targets = suffixes.masked_fill(suffixes == SpecialToken.PAD, SpecialToken.EOS)
        return F.pad(targets, pad=(0, width - targets.shape[1]), value=SpecialToken.EOS)
