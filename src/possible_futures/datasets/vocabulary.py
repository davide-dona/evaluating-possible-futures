import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import chain
from pathlib import Path
from typing import Self

import numpy as np


class SpecialToken(IntEnum):
    """The tokens the vocabulary reserves ahead of the activities."""

    PAD = 0
    UNK = 1
    EOS = 2

    @property
    def label(self) -> str:
        return f'<{self.name}>'


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """The tokens of the activities of the train split.

    The special tokens come first, then the train activities in sorted order.
    """

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
    def fit(cls, traces: Sequence[tuple[str, ...]]) -> Self:
        """Collect the activities of the train traces."""
        return cls(activities=tuple(sorted(set(chain.from_iterable(traces)))))

    @property
    def size(self) -> int:
        """The number of tokens, special tokens included."""
        return len(self.labels)

    @property
    def labels(self) -> tuple[str, ...]:
        """The label of every token: the special tokens, then the train activities."""
        return tuple(token.label for token in SpecialToken) + self.activities

    def encode(self, activities: Iterable[str]) -> np.ndarray:
        """Map activities to tokens, UNK for an activity the train split does not hold."""
        return np.fromiter(
            (self._tokens.get(activity, SpecialToken.UNK) for activity in activities),
            dtype=np.int64,
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({'activities': list(self.activities)}, indent=2) + '\n')

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(activities=tuple(json.loads(path.read_text())['activities']))
