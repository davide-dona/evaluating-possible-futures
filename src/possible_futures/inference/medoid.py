from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.distances import Distance
from possible_futures.inference.draws import Decoding, PrefixDraws


def medoid_index(draws: Sequence[tuple[int, ...]], *, distance: Distance) -> int:
    """The index of the medoid of the draws: the draw with the lowest mean distance to all of
    them, so the one a forecast collapsed on a single draw should keep to stay closest to the
    rest.

    Equal draws are measured once and weighted by how many times they were drawn. Among draws
    tied for the lowest mean, the lowest in tuple order is taken, at its first occurrence.
    """
    counts = Counter(draws)
    unique = sorted(counts)
    weights = [counts[suffix] for suffix in unique]
    totals = distance(left=unique, right=unique) @ weights  # [U, U] @ [U] -> [U]
    return draws.index(unique[totals.argmin()])


def write_medoids(sampled: Path, vocabulary: Vocabulary, *, path: Path, distance: Distance) -> int:
    """Save to `path` the medoid of the draws of every prefix of a decoding, one draw per prefix.

    Args:
        sampled: The decoding whose draws are collapsed.
        vocabulary: The vocabulary the decoding was written with.
        path: The Parquet file to write.
        distance: The distance between suffixes the medoid is taken under.

    Returns:
        The number of prefixes written.
    """
    medoids = [
        PrefixDraws(
            case_id=prefix.case_id,
            prefix_length=prefix.prefix_length,
            draws=[prefix.draws[medoid_index(prefix.draws, distance=distance)]],
        )
        for prefix in Decoding.load(sampled, vocabulary).prefixes
    ]
    Decoding(prefixes=medoids).save(path, vocabulary)
    return len(medoids)
