from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from possible_futures.cases import Case
from possible_futures.processes.process import END, Process

# When the first simulated case starts.
_ORIGIN = pd.Timestamp('2026-01-01')


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

    def simulate(self, *, num_cases: int, rng: np.random.Generator) -> list[Case]:
        """Draw cases from the process, starting as a Poisson process of one case per hour and
        one event per hour within a case.

        Args:
            num_cases: The number of cases drawn.
            rng: Draws the start and the activities of every case.

        Returns:
            list[Case]: The cases in order of their start, each with every prefix that leaves a
                non-empty suffix.
        """
        arrivals = np.cumsum(rng.exponential(1.0, size=num_cases))
        starts = _ORIGIN + pd.to_timedelta(arrivals, unit='h')
        cases = []
        for number, start in enumerate(starts):
            activities = self.sample_suffix((), rng=rng, max_steps=self.CASE_LENGTH)
            cases.append(
                Case(
                    case_id=f'case-{number:05d}',
                    activities=activities,
                    timestamps=tuple(
                        start + pd.Timedelta(hours=position) for position in range(len(activities))
                    ),
                    prefix_lengths=range(1, len(activities)),
                )
            )
        return cases
