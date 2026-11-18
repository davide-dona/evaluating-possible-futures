from typing import Protocol

import torch

from possible_futures.datasets.collation import PrefixTokens


class Decoder(Protocol):
    """Continues a batch of prefixes with suffixes."""

    def draw(self, prefixes: PrefixTokens, *, num_draws: int, max_steps: int) -> torch.Tensor:
        """Draw `num_draws` suffixes per prefix, each at most `max_steps` events.

        Returns:
            The tokens `[B, S, L]` on the device of `prefixes`, PAD past each draw's end.
        """
        ...
