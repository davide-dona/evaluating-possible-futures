from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from possible_futures.datasets.prefix_dataset import PrefixDataset, PrefixItem
from possible_futures.datasets.vocabulary import SpecialToken


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


class TrainBatch(NamedTuple):
    inputs: PrefixTokens
    targets: torch.Tensor  # int64, the events following each prefix, [B, T]


class DecodeBatch(NamedTuple):
    case_ids: tuple[str, ...]  # [B]
    prefixes: PrefixTokens


def train_loader(
    dataset: PrefixDataset, *, batch_size: int, target_length: int, generator: torch.Generator
) -> DataLoader[TrainBatch]:
    """Shuffled batches of the prefixes, each labeled by the events following it.

    Args:
        dataset: The train prefixes.
        batch_size: The number of prefixes in a batch.
        target_length: The number of events following each prefix that label it.
        generator: Shuffles the prefixes.

    Returns:
        DataLoader: A new shuffle of the prefixes on every iteration.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TrainCollate(target_length=target_length),
        generator=generator,
    )


def decode_loader(dataset: PrefixDataset, *, batch_size: int) -> DataLoader[DecodeBatch]:
    """Batches of the prefixes, shortest first, so prefixes of similar length share a batch and
    little of it is padding.

    Args:
        dataset: The prefixes to decode.
        batch_size: The number of prefixes in a batch.

    Returns:
        DataLoader: The same batches on every iteration.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=dataset.prefix_lengths().argsort(stable=True).tolist(),
        collate_fn=decode_collate,
    )


@dataclass(frozen=True, slots=True)
class TrainCollate:
    """Pads train items into a batch, as the `collate_fn` of a `DataLoader`."""

    target_length: int  # events following each prefix that label it

    def __call__(self, items: list[PrefixItem]) -> TrainBatch:
        return TrainBatch(
            inputs=pad_prefixes(items), targets=pad_targets(items, width=self.target_length)
        )


def decode_collate(items: list[PrefixItem]) -> DecodeBatch:
    """Pads items to decode into a batch, as the `collate_fn` of a `DataLoader`."""
    return DecodeBatch(case_ids=tuple(item.case_id for item in items), prefixes=pad_prefixes(items))


def pad_prefixes(items: Sequence[PrefixItem]) -> PrefixTokens:
    """The tokens of the prefixes, PAD past each one's end.

    Args:
        items: The items whose prefixes are padded.

    Returns:
        PrefixTokens: `[B, P]` with `P` the length of the longest prefix.
    """
    return PrefixTokens(
        tokens=pad_sequence(
            [item.prefix for item in items], batch_first=True, padding_value=SpecialToken.PAD
        ),
        lengths=torch.tensor([len(item.prefix) for item in items], dtype=torch.int64),
    )


def pad_suffixes(items: Sequence[PrefixItem]) -> torch.Tensor:
    """The tokens of the whole suffix of each item, PAD past its end.

    Args:
        items: The items whose suffixes are padded.

    Returns:
        torch.Tensor: int64 `[B, S]` with `S` the length of the longest suffix.
    """
    return pad_sequence(
        [item.suffix for item in items], batch_first=True, padding_value=SpecialToken.PAD
    )


def pad_targets(items: Sequence[PrefixItem], *, width: int) -> torch.Tensor:
    """The tokens of the `width` events following each prefix, EOS past the end of its case.

    Args:
        items: The items whose suffixes label their prefixes.
        width: The number of events kept from each suffix.

    Returns:
        torch.Tensor: int64 `[B, width]`.
    """
    targets = pad_sequence(
        [item.suffix[:width] for item in items], batch_first=True, padding_value=SpecialToken.EOS
    )  # [B, T] with T <= width
    return F.pad(targets, pad=(0, width - targets.shape[1]), value=SpecialToken.EOS)
