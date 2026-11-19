from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from possible_futures.distances import Distance
from possible_futures.forecasters.forecasts import Forecast


def medoid_index(draws: Sequence[tuple[str, ...]], *, distance: Distance) -> int:
    """The index of the most central draw: the one with the lowest mean distance to all draws,
    ties going to the lowest in tuple order.

    Args:
        draws: The activities of every draw.
        distance: The distance the mean is taken under.

    Returns:
        int: The index of the first occurrence of the medoid in `draws`.
    """
    counts = Counter(draws)
    unique = sorted(counts)
    weights = [counts[suffix] for suffix in unique]
    totals = distance(left=unique, right=unique) @ weights  # [U, U] @ [U] -> [U]
    return draws.index(unique[totals.argmin()])


def medoids(forecasts: Sequence[Forecast], *, distance: Distance) -> list[Forecast]:
    """Collapse every forecast to a single draw, its medoid.

    Args:
        forecasts: The forecasts collapsed.
        distance: The distance the medoid is taken under.

    Returns:
        list[Forecast]: One forecast per forecast of `forecasts`, holding only its medoid.
    """
    return [
        replace(forecast, draws=[forecast.draws[medoid_index(forecast.draws, distance=distance)]])
        for forecast in forecasts
    ]
