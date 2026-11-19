from collections.abc import Sequence
from typing import NamedTuple, Self

import torch
import torch.nn.functional as F
from jaxtyping import Int64
from torch.nn.utils.rnn import pad_sequence

from possible_futures.datasets.prefix_suffix import PrefixSuffix
from possible_futures.datasets.tokenizer import SpecialToken


class PrefixTokens(NamedTuple):
    """A batch of prefixes as tokens, each padded with PAD to the length of the longest.

    Attributes:
        tokens: The tokens of every prefix, PAD past its length.
        lengths: The number of real tokens in every prefix.
    """

    tokens: Int64[torch.Tensor, 'B P']
    lengths: Int64[torch.Tensor, ' B']

    @classmethod
    def pad(cls, prefixes: Sequence[Int64[torch.Tensor, ' P']]) -> Self:
        """Pad prefixes to the length of the longest of them.

        Args:
            prefixes: The tokens of every prefix.

        Returns:
            PrefixTokens: The padded prefixes and their lengths.
        """
        return cls(
            tokens=pad_sequence(prefixes, batch_first=True, padding_value=SpecialToken.PAD),
            lengths=torch.tensor([len(prefix) for prefix in prefixes], dtype=torch.int64),
        )

    def to(self, device: torch.device) -> 'PrefixTokens':
        """Move the prefixes to a device.

        Args:
            device: The device moved to.

        Returns:
            PrefixTokens: The same prefixes, on `device`.
        """
        return PrefixTokens(*(field.to(device) for field in self))

    def repeat_interleave(self, num_copies: int) -> 'PrefixTokens':
        """Copy every prefix `num_copies` times, the copies of a prefix next to each other, so
        that each copy can be continued independently.

        Args:
            num_copies: The number of copies of every prefix.

        Returns:
            PrefixTokens: `[B * num_copies, P]`, prefix after prefix.
        """
        return PrefixTokens(*(field.repeat_interleave(num_copies, dim=0) for field in self))

    def last_tokens(self) -> Int64[torch.Tensor, ' B']:
        """The last real token of every prefix, the one its suffix continues from.

        Returns:
            torch.Tensor: The token at position `length - 1` of every prefix.
        """
        index = (self.lengths - 1).unsqueeze(dim=1)  # [B] -> [B, 1]
        return self.tokens.gather(dim=1, index=index).squeeze(dim=1)


class PrefixSuffixBatch(NamedTuple):
    """A training batch: prefixes as tokens and the observed suffix following each.

    Attributes:
        prefixes: The prefixes, padded.
        suffixes: The tokens of every suffix, PAD past its end.
    """

    prefixes: PrefixTokens
    suffixes: Int64[torch.Tensor, 'B L']

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

    def targets(self, length: int) -> Int64[torch.Tensor, 'B T']:
        """The first `length` tokens of every suffix, the tokens the model learns to predict.
        A suffix shorter than `length` continues with EOS, so the model also learns when the
        case ends.

        Args:
            length: The number of tokens kept from each suffix, `T`.

        Returns:
            torch.Tensor: The targets of every prefix.
        """
        suffixes = self.suffixes[:, :length]  # [B, L] -> [B, min(L, T)]
        targets = suffixes.masked_fill(suffixes == SpecialToken.PAD, SpecialToken.EOS)
        return F.pad(targets, pad=(0, length - targets.shape[1]), value=SpecialToken.EOS)
