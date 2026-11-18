from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import pandas as pd

from possible_futures.datasets.cases import Case
from possible_futures.datasets.prefixes import Prefix, every_prefix


@dataclass(frozen=True, slots=True)
class OutOfTimeSplit:
    """A split of the cases into training, test, and scored sets."""
    train: list[Case]
    test: list[Case]
    scored: list[Prefix]

    @classmethod
    def of(cls, cases: Sequence[Case], *, test_fraction: float, end_window_days: float) -> Self:
        """Build an out-of-time split of the cases, with a test set and scored prefixes.
        
        Args:
            cases: The list of cases to split.
            test_fraction: The fraction of cases to include in the test set.
            end_window_days: The number of days before the latest case end to consider for the split."""
        window_start = max(case.end for case in cases) - pd.Timedelta(days=end_window_days)
        kept = [case for case in cases if case.start <= window_start]
        kept.sort(key=lambda case: (case.start, case.case_id))
        separation = kept[int(len(kept) * (1 - test_fraction))].start
        test = [case for case in kept if case.end >= separation]
        return cls(
            train=[case for case in kept if case.end < separation],
            test=test,
            scored=[
                prefix for prefix in every_prefix(test) if separation <= prefix.end <= window_start
            ],
        )
