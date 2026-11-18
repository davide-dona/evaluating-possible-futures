from collections.abc import Sequence
from typing import NamedTuple, Self

import torch
from torch.utils.data import Dataset

from possible_futures.datasets.prefixes import Prefix
from possible_futures.datasets.vocabulary import Vocabulary


class PrefixItem(NamedTuple):
    """The tokens of one prefix and of the suffix following it."""

    case_id: str
    prefix: torch.Tensor  # int64, [P]
    suffix: torch.Tensor  # int64, [S]


class PrefixDataset(Dataset[PrefixItem]):
    """Prefixes of cases as tokens, every case encoded once and shared by its prefixes."""

    def __init__(
        self, *, case_ids: Sequence[str], case_tokens: Sequence[torch.Tensor], index: torch.Tensor
    ) -> None:
        """Hold encoded cases and the prefixes taken from them.

        Args:
            case_ids: The ID of every case.
            case_tokens: The tokens of every case, int64 `[L]`, in the order of `case_ids`.
            index: One row per prefix, its case's position in `case_ids` and its length, int64
                `[N, 2]`.
        """
        self.case_ids = tuple(case_ids)
        self.case_tokens = list(case_tokens)
        self.index = index

    @classmethod
    def of(cls, prefixes: Sequence[Prefix], vocabulary: Vocabulary) -> Self:
        """Encode the prefixes, in their order.

        Args:
            prefixes: The prefixes to hold.
            vocabulary: Maps the activities to tokens.

        Returns:
            PrefixDataset: Item `i` is `prefixes[i]`.
        """
        positions: dict[str, int] = {}
        case_ids: list[str] = []
        case_tokens: list[torch.Tensor] = []
        rows: list[tuple[int, int]] = []
        for prefix in prefixes:
            case = prefix.case
            if case.case_id not in positions:
                positions[case.case_id] = len(case_ids)
                case_ids.append(case.case_id)
                case_tokens.append(torch.from_numpy(vocabulary.encode(case.activities)))
            rows.append((positions[case.case_id], prefix.length))
        index = torch.tensor(rows, dtype=torch.int64).view(-1, 2)  # [N, 2] also when N is 0
        return cls(case_ids=case_ids, case_tokens=case_tokens, index=index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> PrefixItem:
        case, length = self.index[i].tolist()
        tokens = self.case_tokens[case]
        return PrefixItem(
            case_id=self.case_ids[case], prefix=tokens[:length], suffix=tokens[length:]
        )

    def prefix_lengths(self) -> torch.Tensor:
        """The length of every prefix.

        Returns:
            torch.Tensor: int64 `[N]`, in item order.
        """
        return self.index[:, 1]
