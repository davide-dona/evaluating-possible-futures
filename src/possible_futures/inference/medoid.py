from collections import Counter
from collections.abc import Sequence

from possible_futures.distances import Distance, activity_ids
from possible_futures.inference.decoding import Forecast


def medoid_index(draws: Sequence[tuple[str, ...]], *, distance: Distance) -> int:
    """The index of the draw with the lowest mean distance to all draws, ties going to the
    lowest in tuple order."""
    (ids,) = activity_ids(draws)
    counts = Counter(ids)
    unique = sorted(counts)
    weights = [counts[suffix] for suffix in unique]
    totals = distance(left=unique, right=unique) @ weights  # [U, U] @ [U] -> [U]
    return ids.index(unique[totals.argmin()])


def medoids(forecasts: Sequence[Forecast], *, distance: Distance) -> list[Forecast]:
    """Keep only the medoid of the draws of every forecast.

    Args:
        forecasts: The forecasts collapsed.
        distance: The distance the medoid is taken under.

    Returns:
        list[Forecast]: One forecast per forecast of `forecasts`, holding only its medoid.
    """
    return [
        Forecast(
            case_id=forecast.case_id,
            prefix_length=forecast.prefix_length,
            draws=[forecast.draws[medoid_index(forecast.draws, distance=distance)]],
        )
        for forecast in forecasts
    ]
