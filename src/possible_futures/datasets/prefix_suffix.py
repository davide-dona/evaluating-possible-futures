from collections.abc import Sequence
from typing import NamedTuple

import torch
from jaxtyping import Int64
from torch.utils.data import Dataset

from possible_futures.cases import Case
from possible_futures.datasets.tokenizer import Tokenizer


class PrefixSuffix(NamedTuple):
    """One training example: the tokens of a prefix and of the observed suffix following it.

    Attributes:
        prefix: The tokens of the prefix.
        suffix: The tokens of the rest of the case.
    """

    prefix: Int64[torch.Tensor, ' P']
    suffix: Int64[torch.Tensor, ' L']


class PrefixSuffixDataset(Dataset[PrefixSuffix]):
    """Every prefix of some cases, paired with the suffix that follows it.

    Each case is tokenized once; its prefixes and suffixes are slices of those tokens, so the
    dataset holds every event once however many prefixes a case has.
    """

    def __init__(self, cases: Sequence[Case], tokenizer: Tokenizer) -> None:
        """
        Args:
            cases: The cases, every prefix of which is a training example.
            tokenizer: Maps the activities to tokens.
        """
        self.case_tokens = [tokenizer.encode(case.activities) for case in cases]
        # One row per prefix: the position of its case in `cases`, and its length. The view keeps
        # two columns when there are no prefixes. Prefixes follow the cases, shortest first.
        self.index = torch.tensor(
            [
                (position, length)
                for position, case in enumerate(cases)
                for length in case.prefix_lengths
            ],
            dtype=torch.int64,
        ).view(-1, 2)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> PrefixSuffix:
        position, length = self.index[i].tolist()
        tokens = self.case_tokens[position]
        return PrefixSuffix(prefix=tokens[:length], suffix=tokens[length:])
