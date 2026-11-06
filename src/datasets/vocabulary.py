import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from src.logs.keys import ACTIVITY_KEY

PAD = 0
UNK = 1
# The first output class: the model predicts EOS and the activities, which are the vocabulary
# from EOS on, so output class `c` is vocabulary index `c + EOS`.
EOS = 2

_SPECIAL_NAMES = ('<PAD>', '<UNK>', '<EOS>')


@dataclass(frozen=True)
class Vocabulary:
    """The activities of the train split, and the longest case kept by preprocessing.

    Index 0 is PAD, 1 UNK, 2 EOS, then the activities in sorted order.
    """

    activities: tuple[str, ...]
    max_case_length: int

    @classmethod
    def fit(cls, train: pd.DataFrame, *, max_case_length: int) -> Self:
        """Read the activities off the train split."""
        return cls(
            activities=tuple(sorted(train[ACTIVITY_KEY].unique())), max_case_length=max_case_length
        )

    @property
    def size(self) -> int:
        """The number of indices, special ones included."""
        return len(_SPECIAL_NAMES) + len(self.activities)

    @property
    def num_outputs(self) -> int:
        """The number of classes the model predicts: EOS and every activity."""
        return self.size - EOS

    @property
    def names(self) -> tuple[str, ...]:
        """The name of every index."""
        return _SPECIAL_NAMES + self.activities

    def encode(self, activities: pd.Series) -> np.ndarray:
        """Map activity names to indices, UNK for an activity the train split does not hold."""
        index = {name: i for i, name in enumerate(self.names)}
        return activities.map(index).fillna(UNK).to_numpy(np.int64)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {'activities': list(self.activities), 'max_case_length': self.max_case_length},
                indent=2,
            )
            + '\n'
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        fields = json.loads(path.read_text())
        return cls(
            activities=tuple(fields['activities']), max_case_length=fields['max_case_length']
        )
