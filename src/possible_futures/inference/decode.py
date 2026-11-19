from collections.abc import Sequence
from pathlib import Path

from possible_futures.cases import Case
from possible_futures.inference.decoding import DecodingWriter, Forecast
from possible_futures.models.decoder import Decoder


def write_decoding(
    decoder: Decoder,
    *,
    test: Sequence[Case],
    path: Path,
    num_draws: int,
    max_steps: int,
    max_rows: int,
) -> int:
    """Draw suffixes for every prefix of the test cases and save them to `path`.

    Args:
        decoder: The decoder the suffixes are drawn from.
        test: The test cases, each with the lengths of its scored prefixes.
        path: The Parquet file the forecasts are written to.
        num_draws: The number of suffixes drawn for each prefix.
        max_steps: The most activities a suffix holds.
        max_rows: The most suffixes drawn at once.

    Returns:
        int: The number of prefixes written.
    """
    # Shortest first, so prefixes of similar length share a batch and little of it is padding.
    prefixes = sorted(
        ((case, length) for case in test for length in case.prefix_lengths),
        key=lambda prefix: prefix[1],
    )
    batch_size = max(1, max_rows // num_draws)
    with DecodingWriter(path) as writer:
        for start in range(0, len(prefixes), batch_size):
            batch = prefixes[start : start + batch_size]
            draws = decoder.draw(
                [case.prefix(length) for case, length in batch],
                num_draws=num_draws,
                max_steps=max_steps,
            )
            writer.write(
                [
                    Forecast(case_id=case.case_id, prefix_length=length, draws=suffixes)
                    for (case, length), suffixes in zip(batch, draws, strict=True)
                ]
            )
    return len(prefixes)
