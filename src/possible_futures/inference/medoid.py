from collections import Counter
from collections.abc import Sequence

from possible_futures.datasets.tokenizer import Tokenizer
from possible_futures.distances import Distance
from possible_futures.inference.decoding import Forecast


def medoid_index(draws: Sequence[tuple[int, ...]], *, distance: Distance) -> int:
    """The index of the draw with the lowest mean distance to all draws, ties going to the
    lowest in tuple order.

    Args:
        draws: The tokens of every draw.
        distance: The distance the mean is taken under.

    Returns:
        int: The index of the first occurrence of the medoid in `draws`.
    """
    counts = Counter(draws)
    unique = sorted(counts)
    weights = [counts[suffix] for suffix in unique]
    totals = distance(left=unique, right=unique) @ weights  # [U, U] @ [U] -> [U]
    return draws.index(unique[totals.argmin()])


def medoids(
    forecasts: Sequence[Forecast], *, tokenizer: Tokenizer, distance: Distance
) -> list[Forecast]:
    """Keep only the medoid of the draws of every forecast.

    Args:
        forecasts: The forecasts collapsed.
        tokenizer: Maps the activities of the draws to tokens.
        distance: The distance the medoid is taken under.

    Returns:
        list[Forecast]: One forecast per forecast of `forecasts`, holding only its medoid.
    """
    collapsed = []
    for forecast in forecasts:
        tokens = [tuple(tokenizer.encode(draw).tolist()) for draw in forecast.draws]
        medoid = forecast.draws[medoid_index(tokens, distance=distance)]
        collapsed.append(
            Forecast(case_id=forecast.case_id, prefix_length=forecast.prefix_length, draws=[medoid])
        )
    return collapsed
