from typing import NamedTuple

import torch


class PrefixBatch(NamedTuple):
    """Prefixes padded to the longest of them, PAD past each one's length."""

    tokens: torch.Tensor  # int64, [B, P]
    lengths: torch.Tensor  # int64, [B]

    def to(self, device: torch.device) -> 'PrefixBatch':
        return PrefixBatch(*(field.to(device) for field in self))

    def repeat_interleave(self, repeats: int) -> 'PrefixBatch':
        """Repeat every prefix `repeats` times in a row, `[B, P] -> [B * repeats, P]`."""
        return PrefixBatch(*(field.repeat_interleave(repeats, dim=0) for field in self))

    def last_events(self) -> torch.Tensor:
        """The last real event of every prefix, `[B, P] -> [B]`."""
        index = (self.lengths - 1).unsqueeze(dim=1)  # [B] -> [B, 1]
        return self.tokens.gather(dim=1, index=index).squeeze(dim=1)


class TrainingBatch(NamedTuple):
    """Training prefixes and the events following each of them."""

    prefixes: PrefixBatch
    targets: torch.Tensor  # int64, the next events, EOS past the case end, [B, W]

    def to(self, device: torch.device) -> 'TrainingBatch':
        return TrainingBatch(prefixes=self.prefixes.to(device), targets=self.targets.to(device))
