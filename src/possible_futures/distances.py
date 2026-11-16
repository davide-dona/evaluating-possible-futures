from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein

from possible_futures.datasets.vocabulary import PAD


class Distance(Protocol):
    """A distance between suffixes, each the vocabulary indices of its activities."""

    def __call__(
        self, *, left: Sequence[tuple[int, ...]], right: Sequence[tuple[int, ...]]
    ) -> np.ndarray:
        """The `[N, M]` distances between `N` suffixes and `M` suffixes."""
        ...


class Exact(Distance):
    """0 between equal suffixes and 1 between any two different ones."""

    def __call__(
        self, *, left: Sequence[tuple[int, ...]], right: Sequence[tuple[int, ...]]
    ) -> np.ndarray:
        ids: dict[tuple[int, ...], int] = {}
        left_ids = [ids.setdefault(suffix, len(ids)) for suffix in left]
        right_ids = [ids.setdefault(suffix, len(ids)) for suffix in right]
        return np.not_equal.outer(left_ids, right_ids).astype(np.float64)


class Bigram(Distance):
    """The total variation distance, half the L1 distance, between the bigram frequencies of two
    suffixes, 0 between two empty suffixes."""

    def __call__(
        self, *, left: Sequence[tuple[int, ...]], right: Sequence[tuple[int, ...]]
    ) -> np.ndarray:
        # Mark the start and end of each suffix with PAD, which no suffix holds
        codes = [np.array([PAD, *s, PAD], dtype=np.int64) for s in [*left, *right]]
        # Give each bigram (a, b) a unique id: a * base + b
        base = max(int(c.max()) for c in codes) + 1
        bigram_ids = [c[:-1] * base + c[1:] for c in codes]

        # Count bigrams per suffix in an [N + M, V] matrix
        owner = np.repeat(np.arange(len(codes)), [len(ids) for ids in bigram_ids])
        _, column = np.unique(np.concatenate(bigram_ids), return_inverse=True)
        n, v = len(codes), column.max() + 1
        counts = np.bincount(owner * v + column, minlength=n * v).reshape(n, v)

        # Normalize to frequencies and compare the rows of left with those of right
        shares = torch.from_numpy(counts / counts.sum(axis=1, keepdims=True))
        return 0.5 * torch.cdist(shares[: len(left)], shares[len(left) :], p=1.0).numpy()


class NormalizedDamerauLevenshtein(Distance):
    """The Damerau-Levenshtein distance divided by the length of the longer suffix, 0 between
    two empty suffixes."""

    def __call__(
        self, *, left: Sequence[tuple[int, ...]], right: Sequence[tuple[int, ...]]
    ) -> np.ndarray:
        return process.cdist(
            queries=left,
            choices=right,
            scorer=DamerauLevenshtein.normalized_distance,
            dtype=np.float64,
        )
