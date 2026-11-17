from dataclasses import dataclass

import numpy as np
import torch

from possible_futures.datasets.batches import PrefixBatch
from possible_futures.datasets.encoded_log import EncodedLog
from possible_futures.models.sampler import SuffixSampler


@dataclass(frozen=True, slots=True, kw_only=True)
class EmpiricalSuffixes(SuffixSampler):
    """Ignores the prefix: every draw is the suffix of a cut of the train split, drawn
    uniformly.

    The train suffixes are never longer than the longest train case, so draws are not cut at
    `max_steps`.
    """

    train_log: EncodedLog
    rng: np.random.Generator

    def sample(self, prefixes: PrefixBatch, *, num_draws: int, max_steps: int) -> torch.Tensor:
        sources = self.train_log.cases.cuts()
        batch_size = len(prefixes.lengths)
        drawn = sources.select(
            self.rng.integers(len(sources.case_indices), size=batch_size * num_draws)
        )
        tokens = torch.from_numpy(self.train_log.suffixes(drawn))  # [B * S, L]
        # [B * S, L] -> [B, S, L]
        return tokens.view(batch_size, num_draws, -1).to(prefixes.lengths.device)
