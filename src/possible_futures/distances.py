from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from jaxtyping import Float64
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein


class Distance(Protocol):
    """A distance between suffixes, each a tuple of activities."""

    def __call__(
        self, *, left: Sequence[tuple[str, ...]], right: Sequence[tuple[str, ...]]
    ) -> Float64[np.ndarray, 'N M']:
        """The distance between every suffix of `left` and every suffix of `right`.

        Args:
            left: `N` suffixes.
            right: `M` suffixes.

        Returns:
            np.ndarray: Row `i`, column `j` holds the distance from `left[i]` to `right[j]`.
        """
        ...


def _codes(suffixes: Sequence[tuple[str, ...]]) -> list[tuple[int, ...]]:
    """Number the activities of some suffixes 0, 1, 2, ... in order of first appearance, since
    comparing integers is several times faster than comparing strings.

    Args:
        suffixes: The suffixes.

    Returns:
        list[tuple[int, ...]]: The code of every activity of every suffix.
    """
    codes: dict[str, int] = {}
    return [tuple(codes.setdefault(a, len(codes)) for a in suffix) for suffix in suffixes]


class Exact(Distance):
    """0 between equal suffixes and 1 between any two different ones."""

    def __call__(
        self, *, left: Sequence[tuple[str, ...]], right: Sequence[tuple[str, ...]]
    ) -> Float64[np.ndarray, 'N M']:
        ids: dict[tuple[str, ...], int] = {}
        left_ids = [ids.setdefault(suffix, len(ids)) for suffix in left]
        right_ids = [ids.setdefault(suffix, len(ids)) for suffix in right]
        return np.not_equal.outer(left_ids, right_ids).astype(np.float64)


class Bigram(Distance):
    """How differently two suffixes order their activities: the total variation distance, half
    the L1 distance, between the frequencies of their consecutive activity pairs. The start and
    the end of a suffix count as pairs too, so the first and last activities matter, and two
    empty suffixes are at distance 0."""

    def __call__(
        self, *, left: Sequence[tuple[str, ...]], right: Sequence[tuple[str, ...]]
    ) -> Float64[np.ndarray, 'N M']:
        codes = _codes([*left, *right])
        # Mark the start and end of each suffix with a code no activity has
        boundary = 1 + max((max(suffix) for suffix in codes if suffix), default=-1)
        sequences = [np.array([boundary, *suffix, boundary], dtype=np.int64) for suffix in codes]
        # Give each pair (a, b) a unique id, a * base + b
        base = boundary + 1
        bigram_ids = [sequence[:-1] * base + sequence[1:] for sequence in sequences]

        # Count the pairs of every suffix in a [N + M, distinct pairs] matrix
        owner = np.repeat(np.arange(len(sequences)), [len(ids) for ids in bigram_ids])
        _, column = np.unique(np.concatenate(bigram_ids), return_inverse=True)
        num_suffixes, num_bigrams = len(sequences), column.max() + 1
        counts = np.bincount(
            owner * num_bigrams + column, minlength=num_suffixes * num_bigrams
        ).reshape(num_suffixes, num_bigrams)

        # Normalize to frequencies and compare the rows of left with those of right. torch.cdist
        # computes the L1 distances without a [N, M, distinct pairs] intermediate.
        shares = torch.from_numpy(counts / counts.sum(axis=1, keepdims=True))
        return 0.5 * torch.cdist(shares[: len(left)], shares[len(left) :], p=1.0).numpy()


class NormalizedDamerauLevenshtein(Distance):
    """The fewest insertions, deletions, substitutions and swaps of neighbours that turn one
    suffix into the other, divided by the length of the longer suffix: 0 for equal suffixes, 1
    for suffixes with nothing in common, and 0 between two empty suffixes."""

    def __call__(
        self, *, left: Sequence[tuple[str, ...]], right: Sequence[tuple[str, ...]]
    ) -> Float64[np.ndarray, 'N M']:
        codes = _codes([*left, *right])
        return process.cdist(
            queries=codes[: len(left)],
            choices=codes[len(left) :],
            scorer=DamerauLevenshtein.normalized_distance,
            dtype=np.float64,
        )
