from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch.utils.data import Dataset

from possible_futures.cases import Case
from possible_futures.datasets.vocabulary import Vocabulary


class PrefixSuffix(NamedTuple):
    """The tokens of one prefix of a case and of the suffix following it."""

    case_id: str
    prefix: torch.Tensor  # int64, [P]
    suffix: torch.Tensor  # int64, [S]


class PrefixSuffixDataset(Dataset[PrefixSuffix]):
    """Every prefix of some cases with the suffix following it, as tokens, every case encoded
    once and shared by its prefixes and suffixes."""

    def __init__(self, cases: Sequence[Case], vocabulary: Vocabulary) -> None:
        """Encode the cases and index their prefixes.

        Args:
            cases: The cases, each with the lengths of its prefixes.
            vocabulary: Maps the activities to tokens.
        """
        self.case_ids = tuple(case.case_id for case in cases)
        self.case_tokens = [torch.from_numpy(vocabulary.encode(case.activities)) for case in cases]
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
        return PrefixSuffix(
            case_id=self.case_ids[case], prefix=tokens[:length], suffix=tokens[length:]
        )

    def prefix_lengths(self) -> torch.Tensor:
        """The length of every prefix.

        Returns:
            torch.Tensor: int64 `[N]`, in item order.
        """
        return self.index[:, 1]
