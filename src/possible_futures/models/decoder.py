from collections.abc import Sequence
from typing import Protocol


class Decoder(Protocol):
    """Continues prefixes with suffixes."""

    def draw(
        self, prefixes: Sequence[tuple[str, ...]], *, num_draws: int, max_steps: int
    ) -> list[list[tuple[str, ...]]]:
        """Draw `num_draws` suffixes per prefix, each at most `max_steps` activities.

        Args:
            prefixes: The activities of every prefix.
            num_draws: The number of suffixes drawn for each prefix.
            max_steps: The most activities a suffix holds.

        Returns:
            list[list[tuple[str, ...]]]: The activities of every suffix drawn, `num_draws` per
                prefix, in the order of `prefixes`.
        """
        ...
