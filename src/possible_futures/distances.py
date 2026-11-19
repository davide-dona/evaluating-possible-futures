from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from jaxtyping import Float64
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein


class Distance(Protocol):
    """A distance between suffixes, each a tuple of the ids `activity_ids` gives its
    activities."""

    def __call__(
        self, *, left: Sequence[tuple[int, ...]], right: Sequence[tuple[int, ...]]
    ) -> Float64[np.ndarray, 'N M']:
        """The `[N, M]` distances between `N` suffixes and `M` suffixes."""
        ...


def activity_ids(*groups: Sequence[tuple[str, ...]]) -> list[list[tuple[int, ...]]]:
    """Replace every activity with its rank, from 1, among the activities of all the groups, so
    the ids sort the way the activities do.

    Args:
        groups: Groups of suffixes, each a tuple of activities.

    Returns:
        list[list[tuple[int, ...]]]: The suffixes of every group, each a tuple of ids.
    """
    activities = sorted({a for suffixes in groups for suffix in suffixes for a in suffix})
    ids = {activity: rank for rank, activity in enumerate(activities, start=1)}
    return [[tuple(ids[a] for a in suffix) for suffix in suffixes] for suffixes in groups]


class Exact(Distance):
    """0 between equal suffixes and 1 between any two different ones."""

    def __call__(
        self, *, left: Sequence[tuple[int, ...]], right: Sequence[tuple[int, ...]]
    ) -> Float64[np.ndarray, 'N M']:
        ids: dict[tuple[int, ...], int] = {}
        left_ids = [ids.setdefault(suffix, len(ids)) for suffix in left]
        right_ids = [ids.setdefault(suffix, len(ids)) for suffix in right]
        return np.not_equal.outer(left_ids, right_ids).astype(np.float64)


class Bigram(Distance):
    """The total variation distance, half the L1 distance, between the bigram frequencies of two
    suffixes, 0 between two empty suffixes."""

    def __call__(
        self, *, left: Sequence[tuple[int, ...]], right: Sequence[tuple[int, ...]]
    ) -> Float64[np.ndarray, 'N M']:
        # Mark the start and end of each suffix with 0, which no activity has as its id
        codes = [np.array([0, *s, 0], dtype=np.int64) for s in [*left, *right]]
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
    ) -> Float64[np.ndarray, 'N M']:
        return process.cdist(
            queries=left,
            choices=right,
            scorer=DamerauLevenshtein.normalized_distance,
            dtype=np.float64,
        )
