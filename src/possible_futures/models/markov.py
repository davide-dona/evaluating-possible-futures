from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from possible_futures.cases import Case
from possible_futures.models.process import END, Process


@dataclass(frozen=True, slots=True)
class MarkovChain(Process):
    """A process whose next activity depends only on the last one.

    Attributes:
        successors: The probability of every activity, or of `END`, after each activity.
    """

    successors: dict[str, dict[str, float]]

    @classmethod
    def fit(cls, cases: Sequence[Case]) -> Self:
        """Count the transitions between consecutive activities of every case, the last
        activity followed by `END`.

        Args:
            cases: The train cases.

        Returns:
            MarkovChain: The chain of the transition frequencies.
        """
        counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for case in cases:
            activities = case.activities
            for activity, following in zip(activities, (*activities[1:], END), strict=True):
                counts[activity][following] += 1
        return cls(
            successors={
                activity: {
                    name: count / counts[activity].total()
                    for name, count in sorted(counts[activity].items())
                }
                for activity in sorted(counts)
            }
        )

    def next_activity_probabilities(self, prefix: tuple[str, ...]) -> dict[str, float]:
        return self.successors[prefix[-1]]
