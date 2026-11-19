import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Self

import numpy as np
import torch
from jaxtyping import Int64

from possible_futures.datasets.cases import Case


class SpecialToken(IntEnum):
    """The tokens the tokenizer reserves ahead of the activities."""

    # Fills the shorter sequences of a batch up to the length of the longest.
    PAD = 0
    # Stands for any activity the train split does not hold.
    UNK = 1
    # Ends a suffix: the decoder emits it when it predicts the case is over.
    EOS = 2

    @property
    def label(self) -> str:
        """The label the tokenizer gives this token, such as `<PAD>`.

        Returns:
            str: The name of the token in angle brackets.
        """
        return f'<{self.name}>'

    @property
    def is_output(self) -> bool:
        """Whether a model may predict this token. PAD only fills batches, so no model predicts
        it.

        Returns:
            bool: False for PAD, True for every other token.
        """
        return self is not SpecialToken.PAD


@dataclass(frozen=True, slots=True)
class Tokenizer:
    """Maps activities to the integer tokens a model reads and writes, and back.

    The tokens are numbered with the special tokens first, then the train activities in sorted
    order.

    Attributes:
        activities: The activities of the train split, sorted.
    """

    activities: tuple[str, ...]
    # The token of every label, special tokens included.
    _tokens: dict[str, int] = field(init=False, repr=False, compare=False)
    # The label of every token, indexed by the token.
    _labels: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, '_tokens', {label: token for token, label in enumerate(self.labels)}
        )
        object.__setattr__(self, '_labels', np.array(self.labels, dtype=object))

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
    def vocabulary_size(self) -> int:
        """The number of tokens, special tokens included.

        Returns:
            int: The number of labels.
        """
        return len(self.labels)

    @property
    def labels(self) -> tuple[str, ...]:
        """The label of every token: the special tokens, then the train activities.

        Returns:
            tuple[str, ...]: The labels, indexed by token.
        """
        return tuple(token.label for token in SpecialToken) + self.activities

    def known(self, activities: Iterable[str]) -> tuple[str, ...]:
        """Replace every activity the train split does not hold with the UNK label, so that the
        activities read as the model sees them.

        Args:
            activities: The activities.

        Returns:
            tuple[str, ...]: The activities, `<UNK>` in place of every unknown one.
        """
        unknown = SpecialToken.UNK.label
        return tuple(activity if activity in self._tokens else unknown for activity in activities)

    def encode(self, activities: Iterable[str]) -> Int64[torch.Tensor, ' L']:
        """Map activities to tokens, UNK for an activity the train split does not hold.

        Args:
            activities: The activities.

        Returns:
            torch.Tensor: The token of every activity.
        """
        return torch.tensor(
            [self._tokens.get(activity, SpecialToken.UNK) for activity in activities],
            dtype=torch.int64,
        )

    def decode(self, tokens: Int64[torch.Tensor, 'B L']) -> list[tuple[str, ...]]:
        """Map rows of tokens, each PAD past its end, to their labels.

        Args:
            tokens: The tokens of every row.

        Returns:
            list[tuple[str, ...]]: The labels of the tokens of every row, PAD left out.
        """
        tokens = tokens.cpu()
        lengths = (tokens != SpecialToken.PAD).sum(dim=1).tolist()
        labels = self._labels[tokens.numpy()]
        return [tuple(row[:length].tolist()) for row, length in zip(labels, lengths, strict=True)]

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
