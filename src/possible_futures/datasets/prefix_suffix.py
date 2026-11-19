from collections.abc import Sequence
from typing import NamedTuple

import torch
from jaxtyping import Int64
from torch.utils.data import Dataset

from possible_futures.cases import Case
from possible_futures.datasets.tokenizer import Tokenizer


class PrefixSuffix(NamedTuple):
    """The tokens of one prefix of a case and of the suffix following it."""

    prefix: Int64[torch.Tensor, ' P']
    suffix: Int64[torch.Tensor, ' L']


class PrefixSuffixDataset(Dataset[PrefixSuffix]):
    """Every prefix of some cases with the suffix following it, as tokens, every case encoded
    once and shared by its prefixes and suffixes."""

    def __init__(self, cases: Sequence[Case], tokenizer: Tokenizer) -> None:
        """Encode the cases and index their prefixes.

        Args:
            cases: The cases, each with the lengths of its prefixes.
            tokenizer: Maps the activities to tokens.
        """
        self.case_tokens = [torch.from_numpy(tokenizer.encode(case.activities)) for case in cases]
        # One row per prefix, its case's position in `cases` and its length, [N, 2] also when N
        # is 0. The prefixes follow the cases, shortest first within a case.
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
        case, length = self.index[i].tolist()
        tokens = self.case_tokens[case]
        return PrefixSuffix(prefix=tokens[:length], suffix=tokens[length:])
