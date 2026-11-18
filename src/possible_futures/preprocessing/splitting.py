from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self

import pandas as pd

from possible_futures.preprocessing.cases import Case


@dataclass(frozen=True, slots=True)
class OutOfTimeSplit:
    """The train and test cases of a log, split out of time."""

    train: list[Case]
    test: list[Case]  # each holding only its scored prefixes

    @classmethod
    def of(cls, cases: Sequence[Case], *, test_fraction: float, end_window_days: float) -> Self:
        """Drop the cases starting in the last `end_window_days` of the log. The separation
        time is computed as the start of the case at the `1 - test_fraction` quantile of the 
        kept cases. 

        Train keeps the cases that end before the separation, with all their prefixes. Test keeps
        the cases that end at or after it, with only their scored prefixes. A prefix is scored
        when its last event falls after the separation and no later than the start of the end
        window, and at least one event of the case follows it.

        Args:
            cases: The cases of the log.
            test_fraction: The fraction of the kept cases that start the test split.
            end_window_days: The days before the last event of the log in which starting cases
                are dropped and prefixes are not scored.

        Returns:
            OutOfTimeSplit: The cases of each split, in order of their start.
        """
        # Filter out the cases starting in the last `end_window_days`
        window_start = max(case.end for case in cases) - pd.Timedelta(days=end_window_days)
        kept = sorted(
            (case for case in cases if case.start <= window_start),
            key=lambda case: (case.start, case.case_id),
        )
        # Compute the separation time as the start of the first test case
        first_test_case = int(len(kept) * (1 - test_fraction))
        separation = kept[first_test_case].start
        
        return cls(
            # Train keeps the cases that end before the separation, with all their prefixes.
            train=[case for case in kept if case.end < separation],
            test=[
                # Test keeps the cases that end at or after the separation, with only their scored prefixes.
                replace(
                    case,
                    prefix_lengths=_scored_lengths(
                        case,
                        separation=separation,
                        window_start=window_start,
                        crosses_separation=rank < first_test_case,
                    ),
                )
                for rank, case in enumerate(kept)
                if case.end >= separation
            ],
        )


def _scored_lengths(
    case: Case, *, separation: pd.Timestamp, window_start: pd.Timestamp, crosses_separation: bool
) -> range:
    """The lengths of the scored prefixes of a test case.

    Args:
        case: The test case.
        separation: The start of the first test case.
        window_start: The start of the end window.
        crosses_separation: Whether the case starts ahead of the first test case.

    Returns:
        range: The lengths, empty when no prefix is scored.
    """
    # A prefix of length k ends at the k-th event, so its length is the 1-based position of the
    # event it has to reach.
    shortest = 1
    if crosses_separation:
        shortest = next(
            (
                position
                for position, timestamp in enumerate(case.timestamps, start=1)
                if timestamp > separation
            ),
            1,
        )
    before_window = sum(timestamp <= window_start for timestamp in case.timestamps)
    return range(shortest, min(before_window, len(case) - 1) + 1)
