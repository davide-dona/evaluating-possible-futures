from typing import Protocol

import torch

from possible_futures.datasets.batches import PrefixBatch


class SuffixSampler(Protocol):
    """Draws suffixes that continue a batch of prefixes."""

    def sample(self, prefixes: PrefixBatch, *, num_draws: int, max_steps: int) -> torch.Tensor:
        """Draw `num_draws` suffixes per prefix, each at most `max_steps` events.

        Returns:
            The tokens `[B, S, L]` on the device of `prefixes`, PAD past each draw's end.
        """
        ...
