from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.logs.keys import ACTIVITY_KEY, CASE_KEY, TIMESTAMP_KEY

# The pseudo-activity that ends a case.
END = '<END>'


class NextActivities(Protocol):
    """A source of the distribution of the activity that follows a prefix."""

    def next_activities(self, prefix: tuple[str, ...]) -> dict[str, float]:
        """The probability of every activity, or of `END`, right after `prefix`."""
        ...


def sample_suffix(
    source: NextActivities, prefix: tuple[str, ...], rng: np.random.Generator, *, max_steps: int
) -> tuple[tuple[str, ...], bool]:
    """Draw the activities following `prefix` one at a time until `END`.

    Returns:
        The drawn activities, and whether `max_steps` of them were drawn without reaching `END`.
    """
    return _decode(source, prefix, lambda options: _choose(options, rng), max_steps=max_steps)


def greedy_suffix(
    source: NextActivities, prefix: tuple[str, ...], *, max_steps: int
) -> tuple[tuple[str, ...], bool]:
    """Take the most probable activity following the prefix at every step, until `END`.

    Returns:
        As `sample_suffix`.
    """
    return _decode(
        source, prefix, lambda options: max(options, key=options.get), max_steps=max_steps
    )


def _decode(
    source: NextActivities,
    prefix: tuple[str, ...],
    pick: Callable[[dict[str, float]], str],
    *,
    max_steps: int,
) -> tuple[tuple[str, ...], bool]:
    """Extend the prefix with the activity `pick` chooses among the next ones, until `END`."""
    suffix = []
    while len(suffix) < max_steps:
        activity = pick(source.next_activities((*prefix, *suffix)))
        if activity == END:
            return tuple(suffix), False
        suffix.append(activity)
    return tuple(suffix), True


def _choose(options: dict[str, float], rng: np.random.Generator) -> str:
    """Draw one of the options with its probability."""
    threshold = rng.random()
    for option, probability in options.items():
        threshold -= probability
        if threshold < 0.0:
            return option
    # Rounding can leave the threshold just above zero after the last option.
    return option


# Every case of `SyntheticProcess` holds A, one of B and C, D, one of E and F, and G.
CASE_LENGTH = 5


@dataclass(frozen=True, kw_only=True)
class SyntheticProcess:
    """A, then B or C, then D, then E or F, then G.

    The choice between E and F depends on whether the case went through B or C, which D hides
    from a first-order Markov chain.

    Attributes:
        p_b: Probability of B after A.
        q_b: Probability of E after D on a case through B.
        q_c: Probability of E after D on a case through C.
    """

    p_b: float
    q_b: float
    q_c: float

    def __post_init__(self) -> None:
        for name in ('p_b', 'q_b', 'q_c'):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f'{name} must be a probability, got {getattr(self, name)}')

    def next_activities(self, prefix: tuple[str, ...]) -> dict[str, float]:
        match prefix:
            case ():
                return {'A': 1.0}
            case (*_, 'A'):
                return {'B': self.p_b, 'C': 1.0 - self.p_b}
            case (*_, 'B' | 'C'):
                return {'D': 1.0}
            case (_, branch, 'D'):
                q = self.q_b if branch == 'B' else self.q_c
                return {'E': q, 'F': 1.0 - q}
            case (*_, 'E' | 'F'):
                return {'G': 1.0}
            case _:
                return {END: 1.0}


# When the first simulated case starts.
_ORIGIN = pd.Timestamp('2026-01-01')


def simulate_log(
    process: SyntheticProcess, *, num_cases: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Draw a log of `num_cases` cases from the process.

    Cases arrive as a Poisson process of one case per hour on average, and the events of a case
    are one hour apart.

    Returns:
        One row per event with the case, the activity, and the timestamp, in case order.
    """
    starts = _ORIGIN + pd.to_timedelta(np.cumsum(rng.exponential(1.0, size=num_cases)), unit='h')
    rows = []
    for number, start in enumerate(starts):
        trace, _ = sample_suffix(process, (), rng, max_steps=CASE_LENGTH)
        for position, activity in enumerate(trace):
            rows.append((f'case-{number:05d}', activity, start + pd.Timedelta(hours=position)))
    return pd.DataFrame(rows, columns=[CASE_KEY, ACTIVITY_KEY, TIMESTAMP_KEY])


@dataclass(frozen=True)
class MarkovChain:
    """A process whose next activity depends only on the last one.

    Attributes:
        successors: The probability of every activity, or of `END`, after each activity.
    """

    successors: dict[str, dict[str, float]]

    @classmethod
    def fit(cls, split: pd.DataFrame) -> 'MarkovChain':
        """Count the transitions between consecutive events of a split read by `read_split`,
        the last event of a case followed by `END`."""
        following = split.groupby(CASE_KEY, sort=False)[ACTIVITY_KEY].shift(-1).fillna(END)
        shares = pd.crosstab(split[ACTIVITY_KEY], following, normalize='index')
        return cls(
            {
                activity: {name: share for name, share in row.items() if share > 0.0}
                for activity, row in shares.iterrows()
            }
        )

    def next_activities(self, prefix: tuple[str, ...]) -> dict[str, float]:
        return self.successors[prefix[-1]]
