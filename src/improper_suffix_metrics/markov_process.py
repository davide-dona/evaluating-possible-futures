import heapq
import math
from dataclasses import dataclass

import numpy as np

END = 'END'


@dataclass(frozen=True)
class Trace:
    """A sequence of activities, each with the time in days elapsed since the previous event."""

    activities: tuple[str, ...]
    times: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.activities)

    def __getitem__(self, index: slice) -> 'Trace':
        return Trace(self.activities[index], self.times[index])


@dataclass(frozen=True)
class Activity:
    """An activity of a Markov process.

    Attributes:
        median_days: Median of the lognormal time elapsed before the activity.
        spread: Log-scale standard deviation of that time.
        successors: Probability of each activity, or `END`, that follows this one.
    """

    median_days: float
    spread: float
    successors: dict[str, float]


@dataclass(frozen=True)
class MarkovProcess:
    """A process whose next activity depends only on the last one.

    Attributes:
        start: Probability of each activity that opens a case.
        activities: Each activity by name.
    """

    start: dict[str, float]
    activities: dict[str, Activity]

    def sample(self, rng: np.random.Generator) -> Trace:
        """Draw a whole case."""
        return self._walk(self.start, rng)

    def sample_suffix(self, last: str, rng: np.random.Generator) -> Trace:
        """Draw the activities and times that follow the activity `last` until the case ends."""
        return self._walk(self.activities[last].successors, rng)

    def mode_suffix(self, last: str) -> Trace:
        """Return the most probable path from the activity `last` to the end, each activity at its
        median time.

        The most probable path is the shortest one under the cost `-log p` of each transition.
        """
        costs = {last: 0.0}
        previous = {}
        queue = [(0.0, last)]
        while (current := heapq.heappop(queue)[1]) != END:
            for following, probability in self.activities[current].successors.items():
                cost = costs[current] - math.log(probability)
                if cost < costs.get(following, math.inf):
                    costs[following] = cost
                    previous[following] = current
                    heapq.heappush(queue, (cost, following))
        names = []
        while (current := previous[current]) != last:
            names.append(current)
        names.reverse()
        return Trace(tuple(names), tuple(self.activities[name].median_days for name in names))

    def _walk(self, options: dict[str, float], rng: np.random.Generator) -> Trace:
        names = []
        times = []
        while (name := _choose(options, rng)) != END:
            activity = self.activities[name]
            names.append(name)
            times.append(activity.median_days * math.exp(activity.spread * rng.standard_normal()))
            options = activity.successors
        return Trace(tuple(names), tuple(times))


def _choose(options: dict[str, float], rng: np.random.Generator) -> str:
    threshold = rng.random()
    for option, probability in options.items():
        threshold -= probability
        if threshold < 0.0:
            return option
    # Rounding can leave the threshold just above zero after the last option.
    return option
