from dataclasses import dataclass

import numpy as np
import torch

from possible_futures.datasets.collation import PrefixSuffixBatch, PrefixTokens
from possible_futures.datasets.prefix_suffix import PrefixSuffixDataset
from possible_futures.models.decoder import Decoder


@dataclass(frozen=True, slots=True, kw_only=True)
class UnconditionalDecoder(Decoder):
    """Ignores the prefix: every draw is the suffix of a train prefix, drawn uniformly.

    The train suffixes are never longer than the longest train case, so draws are not truncated
    at `max_steps`.
    """

    train: PrefixSuffixDataset
    rng: np.random.Generator

    def draw(self, prefixes: PrefixTokens, *, num_draws: int, max_steps: int) -> torch.Tensor:
        batch_size = len(prefixes.lengths)
        drawn = self.rng.integers(len(self.train), size=batch_size * num_draws)
        tokens = PrefixSuffixBatch.collate(
            [self.train[index] for index in drawn]
        ).suffixes  # [B * S, L]
        # [B * S, L] -> [B, S, L]
        return tokens.view(batch_size, num_draws, -1).to(prefixes.lengths.device)
