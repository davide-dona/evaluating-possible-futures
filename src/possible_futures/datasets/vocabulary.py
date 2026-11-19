import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Self

import numpy as np
from jaxtyping import Int64

from possible_futures.cases import Case


class SpecialToken(IntEnum):
    """The tokens the vocabulary reserves ahead of the activities."""

    # Padding token, fills sequences to the same length for batching.
    PAD = 0
    # Unknown token, activity not in the train split.
    UNK = 1
    # End-of-sequence token, marks the end of a sequence for the decoder.
    EOS = 2

    @property
    def label(self) -> str:
        return f'<{self.name}>'


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Defines the mapping from activities to tokens, including special tokens."""

    # The activities of the train split, sorted.
    activities: tuple[str, ...]
    # Mapping from label to token, special tokens included.
    _tokens: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize the mapping from label to token."""
        object.__setattr__(
            self, '_tokens', {label: token for token, label in enumerate(self.labels)}
        )

    @classmethod
    def fit(cls, cases: Sequence[Case]) -> Self:
        """Collect the activities of the cases.

        Args:
            cases: The train cases.

        Returns:
            Vocabulary: The vocabulary of every activity the cases hold.
        """
        return cls(activities=tuple(sorted({a for case in cases for a in case.activities})))

    @property
    def size(self) -> int:
        """The number of tokens, special tokens included."""
        return len(self.labels)

    @property
    def labels(self) -> tuple[str, ...]:
        """The label of every token: the special tokens, then the train activities."""
        return tuple(token.label for token in SpecialToken) + self.activities

    def encode(self, activities: Iterable[str]) -> Int64[np.ndarray, ' N']:
        """Map activities to tokens, UNK for an activity the train split does not hold.

        Args:
            activities: The activities mapped.

        Returns:
            np.ndarray: The token of every activity.
        """
        return np.fromiter(
            (self._tokens.get(activity, SpecialToken.UNK) for activity in activities),
            dtype=np.int64,
        )

    def save(self, path: Path) -> None:
        """Write the activities to a JSON file.

        Args:
            path: The JSON file written.
        """
        path.write_text(json.dumps({'activities': list(self.activities)}, indent=2) + '\n')

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read the activities from a JSON file.

        Args:
            path: The JSON file read.

        Returns:
            Vocabulary: The vocabulary of the activities read.
        """
        return cls(activities=tuple(json.loads(path.read_text())['activities']))
