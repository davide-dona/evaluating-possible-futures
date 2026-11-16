import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from possible_futures.logs.keys import ACTIVITY_KEY

PAD = 0
# The first output class: the model predicts UNK, EOS, and the activities, which are the
# vocabulary from UNK on, so output class `c` is vocabulary index `c + UNK`.
UNK = 1
EOS = 2

_SPECIAL_NAMES = ('<PAD>', '<UNK>', '<EOS>')


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """The indices of the activities of the train split.

    Index 0 is PAD, 1 UNK, 2 EOS, then the train activities in sorted order.
    """

    activities: tuple[str, ...]  # the train activities, without the special names
    _indices: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_indices', {name: i for i, name in enumerate(self.names)})

    @classmethod
    def fit(cls, train: pd.DataFrame) -> Self:
        """Read the activities off the train split."""
        return cls(activities=tuple(sorted(train[ACTIVITY_KEY].unique())))

    @property
    def size(self) -> int:
        """The number of indices, special ones included."""
        return len(self.names)

    @property
    def names(self) -> tuple[str, ...]:
        """The name of every index: the special names, then the train activities."""
        return _SPECIAL_NAMES + self.activities

    def encode(self, activities: Iterable[str]) -> np.ndarray:
        """Map activity names to indices, UNK for an activity the train split does not hold."""
        return np.fromiter((self._indices.get(name, UNK) for name in activities), dtype=np.int64)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({'activities': list(self.activities)}, indent=2) + '\n')

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(activities=tuple(json.loads(path.read_text())['activities']))
