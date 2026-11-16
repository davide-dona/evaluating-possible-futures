from dataclasses import dataclass
from typing import Self

import pandas as pd

from possible_futures.logs.keys import ACTIVITY_KEY, CASE_KEY
from possible_futures.processes.decoding import END, Process


@dataclass(frozen=True, slots=True)
class MarkovChain(Process):
    """A process whose next activity depends only on the last one.

    Attributes:
        successors: The probability of every activity, or of `END`, after each activity.
    """

    successors: dict[str, dict[str, float]]

    @classmethod
    def fit(cls, split: pd.DataFrame) -> Self:
        """Count the transitions between consecutive events of a split read by `read_split`,
        the last event of a case followed by `END`."""
        following = split.groupby(CASE_KEY, sort=False)[ACTIVITY_KEY].shift(-1).fillna(END)
        shares = pd.crosstab(split[ACTIVITY_KEY], following, normalize='index')
        return cls(
            successors={
                activity: {name: share for name, share in row.items() if share > 0.0}
                for activity, row in shares.iterrows()
            }
        )

    def next_activity_probabilities(self, prefix: tuple[str, ...]) -> dict[str, float]:
        return self.successors[prefix[-1]]
