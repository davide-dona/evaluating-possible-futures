from collections.abc import Callable, Sequence

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein

# A suffix is a string holding one character per activity, so equal suffixes are equal strings
# and rapidfuzz compares them natively.
Distance = Callable[[Sequence[str]], np.ndarray]

# Characters no activity is encoded as, marking the two ends of a suffix in its bigrams.
_START, _END = 0, 1


def exact(suffixes: Sequence[str]) -> np.ndarray:
    """`[N, N]`, 0 between equal suffixes and 1 between any two different ones."""
    left = np.empty(len(suffixes), dtype=object)
    left[:] = list(suffixes)
    return np.not_equal.outer(left, left).astype(np.float64)


def bigram(suffixes: Sequence[str]) -> np.ndarray:
    """`[N, N]`, the multiset Jaccard distance between the bigrams of the suffixes, each padded
    at both ends, so a suffix of `n` activities holds `n + 1` bigrams and never none.

    It is `1 - |A n B| / |A u B|` on multisets, computed from the L1 distance between bigram
    counts as `2 L1 / (|A| + |B| + L1)`.
    """
    codes = [np.array([_START, *map(ord, suffix), _END], dtype=np.int64) for suffix in suffixes]
    width = max(int(code.max()) for code in codes) + 1
    pairs = np.concatenate([code[:-1] * width + code[1:] for code in codes])
    rows = np.repeat(np.arange(len(codes)), [len(code) - 1 for code in codes])
    vocabulary, columns = np.unique(pairs, return_inverse=True)
    counts = np.zeros(shape=(len(codes), len(vocabulary)))
    np.add.at(counts, (rows, columns), 1.0)
    l1 = np.abs(counts[:, None, :] - counts[None, :, :]).sum(axis=-1)  # [N, N, V] -> [N, N]
    sizes = counts.sum(axis=1)
    return 2.0 * l1 / (sizes[:, None] + sizes[None, :] + l1)


def damerau_levenshtein(suffixes: Sequence[str]) -> np.ndarray:
    """`[N, N]`, the Damerau-Levenshtein distance divided by the longer suffix's length, 0
    between two empty suffixes."""
    return process.cdist(
        queries=suffixes,
        choices=suffixes,
        scorer=DamerauLevenshtein.normalized_distance,
        dtype=np.float64,
    )


DISTANCES: dict[str, Distance] = {
    'exact': exact,
    'bigram': bigram,
    'dl': damerau_levenshtein,
}
