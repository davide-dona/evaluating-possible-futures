from collections import Counter
from collections.abc import Sequence

from possible_futures.distances import Distance
from possible_futures.inference.draws import Decoding, Forecast


def medoid_index(draws: Sequence[tuple[int, ...]], *, distance: Distance) -> int:
    """The index of the draw with the lowest mean distance to all draws, ties going to the
    lowest in tuple order."""
    counts = Counter(draws)
    unique = sorted(counts)
    weights = [counts[suffix] for suffix in unique]
    totals = distance(left=unique, right=unique) @ weights  # [U, U] @ [U] -> [U]
    return draws.index(unique[totals.argmin()])


def medoids(decoding: Decoding, *, distance: Distance) -> Decoding:
    """Keep only the medoid of the draws of every prefix."""
    return Decoding(
        forecasts=[
            Forecast(
                case_id=forecast.case_id,
                prefix_length=forecast.prefix_length,
                draws=[forecast.draws[medoid_index(forecast.draws, distance=distance)]],
            )
            for forecast in decoding.forecasts
        ]
    )
