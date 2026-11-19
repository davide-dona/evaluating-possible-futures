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
    """The tokens the tokenizer reserves ahead of the activities."""

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
class Tokenizer:
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
            Tokenizer: The tokenizer of every activity the cases hold.
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
        """Write the token of every label, special tokens included, to a JSON file.

        Args:
            path: The JSON file written.
        """
        path.write_text(json.dumps({'tokens': self._tokens}, indent=2) + '\n')

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read the token of every label from a JSON file written by `save`.

        Args:
            path: The JSON file read.

        Returns:
            Tokenizer: The tokenizer of the activities read.

        Raises:
            ValueError: If the file maps the labels to tokens differently from this tokenizer,
                such as after a change to the special tokens.
        """
        tokens = json.loads(path.read_text())['tokens']
        specials = {token.label for token in SpecialToken}
        labels = sorted(tokens, key=tokens.get)
        tokenizer = cls(activities=tuple(label for label in labels if label not in specials))
        if tokenizer._tokens != tokens:
            raise ValueError(f'{path} does not map the labels to the tokens this tokenizer uses')
        return tokenizer
