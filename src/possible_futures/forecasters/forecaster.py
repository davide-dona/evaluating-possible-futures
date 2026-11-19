from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from possible_futures.cases import Case
from possible_futures.forecasts import Forecast, ForecastWriter


class Forecaster(Protocol):
    """Imagines possible futures of prefixes: draws suffixes that could follow each one."""

    def draw(
        self, prefixes: Sequence[tuple[str, ...]], *, num_draws: int, max_steps: int
    ) -> list[list[tuple[str, ...]]]:
        """Draw `num_draws` suffixes per prefix, each at most `max_steps` activities. A draw is
        one possible future; a deterministic forecaster returns the same suffix every time.

        Args:
            prefixes: The activities of every prefix.
            num_draws: The number of suffixes drawn for each prefix.
            max_steps: The most activities a suffix holds.

        Returns:
            list[list[tuple[str, ...]]]: The activities of every suffix drawn, `num_draws` per
                prefix, in the order of `prefixes`.
        """
        ...


def write_forecasts(
    forecaster: Forecaster,
    *,
    test: Sequence[Case],
    path: Path,
    num_draws: int,
    max_steps: int,
    max_rows: int,
) -> int:
    """Draw suffixes for every scored prefix of the test cases and save them to `path`.

    Args:
        forecaster: The forecaster the suffixes are drawn from.
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
    with ForecastWriter(path) as writer:
        for start in range(0, len(prefixes), batch_size):
            batch = prefixes[start : start + batch_size]
            draws = forecaster.draw(
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
