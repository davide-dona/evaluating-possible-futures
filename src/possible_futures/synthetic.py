from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import numpy as np
import pandas as pd

from possible_futures.models.process import END, Process

# When the first simulated case starts.
_ORIGIN = pd.Timestamp('2026-01-01')


class SimulatedEvents(NamedTuple):
    """The events of a simulated log, in case order."""

    case_ids: list[str]
    activities: list[str]
    timestamps: list[pd.Timestamp]


@dataclass(frozen=True, slots=True, kw_only=True)
class SyntheticProcess(Process):
    """A, then B or C, then D, then E or F, then G.

    The choice between E and F depends on whether the case went through B or C.

    Attributes:
        p_b: Probability of B after A.
        q_b: Probability of E after D on a case through B.
        q_c: Probability of E after D on a case through C.
    """

    # Every case holds A, one of B and C, D, one of E and F, and G.
    CASE_LENGTH: ClassVar[int] = 5

    p_b: float
    q_b: float
    q_c: float

    def __post_init__(self) -> None:
        for name in ('p_b', 'q_b', 'q_c'):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f'{name} must be a probability, got {getattr(self, name)}')

    def __str__(self) -> str:
        return f'P(B | A) = {self.p_b}, P(E | B, D) = {self.q_b}, P(E | C, D) = {self.q_c}'

    def next_activity_probabilities(self, prefix: tuple[str, ...]) -> dict[str, float]:
        """Return the true probabilities of the next activity after `prefix`."""
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

    def simulate(self, *, num_cases: int, rng: np.random.Generator) -> SimulatedEvents:
        """Draw the events of `num_cases` cases from the process."""
        # Simulate trace arrival times as a Poisson process
        arrivals = np.cumsum(rng.exponential(1.0, size=num_cases))
        case_starts = _ORIGIN + pd.to_timedelta(arrivals, unit='h')

        # Simulate the events of every case
        events = SimulatedEvents(case_ids=[], activities=[], timestamps=[])
        for number, start in enumerate(case_starts):
            # Simulate the trace of the case
            trace = self.sample_suffix((), rng=rng, max_steps=self.CASE_LENGTH)
            # Assign case id and timestamps to every event of the trace
            for position, activity in enumerate(trace):
                events.case_ids.append(f'case-{number:05d}')
                events.activities.append(activity)
                events.timestamps.append(start + pd.Timedelta(hours=position))
        return events
