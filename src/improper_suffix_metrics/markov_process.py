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

    def __str__(self) -> str:
        return ' -> '.join(self.activities)

    @property
    def duration(self) -> float:
        """Total time in days of the activities."""
        return sum(self.times)

    def __getitem__(self, index: slice) -> 'Trace':
        return Trace(self.activities[index], self.times[index])


@dataclass(frozen=True, kw_only=True)
class LogNormal:
    """A lognormal distribution.

    Attributes:
        median: Median of the distribution.
        spread: Standard deviation of its logarithm.
    """

    median: float
    spread: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a value from the distribution."""
        return rng.lognormal(math.log(self.median), self.spread)


@dataclass(frozen=True, kw_only=True)
class Activity:
    """An activity of a Markov process.

    Attributes:
        time: Distribution of the time in days elapsed before the activity.
        successors: Probability of each activity, or `END`, that follows this one.
    """

    time: LogNormal
    successors: dict[str, float]

    def __post_init__(self) -> None:
        _check_probabilities(self.successors)


@dataclass(frozen=True)
class MarkovProcess:
    """A process whose next activity depends only on the last one.

    Attributes:
        start: Probability of each activity that opens a case.
        activities: Each activity by name.
    """

    start: dict[str, float]
    activities: dict[str, Activity]

    def __post_init__(self) -> None:
        _check_probabilities(self.start)

    def sample(self, rng: np.random.Generator) -> Trace:
        """Draw a whole case from its first activity to its last.

        Args:
            rng: Source of the random choices of activities and times.

        Returns:
            The case, each activity after the first with its time drawn from the activity's
            distribution.
        """
        # A walk from the start to the end allows to draw full cases
        case = self._walk(self.start, rng)
        return Trace(case.activities, (0.0, *case.times[1:]))

    def sample_suffix(self, prefix: Trace, rng: np.random.Generator) -> Trace:
        """Draw a suffix from the true distribution of the suffixes that follow the prefix.

        Repeated calls with the same prefix give independent draws, so their spread reflects the
        uncertainty of the process about how the case continues.

        Args:
            prefix: The non-empty beginning of a case.
            rng: Source of the random choices of activities and times.

        Returns:
            The activities that follow the prefix until the case ends, each with its time drawn
            from the activity's distribution. It is empty when the case ends right after the
            prefix.
        """
        # Walk from the successor of the prefix last activity to the end
        last = prefix.activities[-1]
        return self._walk(self.activities[last].successors, rng)

    def mode_suffix(self, prefix: Trace) -> Trace:
        """Return the most probable sequence of activities that follows the prefix, each activity
        at its median time.

        The probability of a sequence is the product of the probabilities of its transitions, so
        the most probable sequence is the shortest path to `END` when each transition costs
        `-log p`. Dijkstra's algorithm finds it.

        Args:
            prefix: The non-empty beginning of a case.

        Returns:
            The activities on the most probable path from the last activity of the prefix to the
            end of the case, each with its median time. It is empty when ending the case right
            after the prefix is the most probable continuation.
        """
        # Initialize Dijkstra's algorithm variables
        last = prefix.activities[-1]
        costs = {last: 0.0}
        previous = {}
        queue = [(0.0, last)]
        # While there are still activities to explore
        while (current := heapq.heappop(queue)[1]) != END:
            # For each of the successors of the current activity
            for following, probability in self.activities[current].successors.items():
                cost = costs[current] - math.log(probability)
                # If the cost of reaching the successor is lower than any previously found
                if cost < costs.get(following, math.inf):
                    # Update the cost and previous activity, and add it to the queue
                    costs[following] = cost
                    previous[following] = current
                    heapq.heappush(queue, (cost, following))

        # Reconstruct the most probable path
        names = []
        while (current := previous[current]) != last:
            names.append(current)
        names.reverse()
        return Trace(tuple(names), tuple(self.activities[name].time.median for name in names))

    def _walk(self, options: dict[str, float], rng: np.random.Generator) -> Trace:
        """Draw activities and their times, starting from `options`, until `END` is drawn."""
        names = []
        times = []
        while (name := _choose(options, rng)) != END:
            activity = self.activities[name]
            names.append(name)
            times.append(activity.time.sample(rng))
            options = activity.successors
        return Trace(tuple(names), tuple(times))


def _check_probabilities(options: dict[str, float]) -> None:
    """Raise a `ValueError` unless the probabilities of the options are non-negative and sum to
    one."""
    if any(probability < 0.0 for probability in options.values()):
        raise ValueError(f'Probabilities must be non-negative, got {options}')
    if not math.isclose(math.fsum(options.values()), 1.0):
        raise ValueError(f'Probabilities must sum to one, got {options}')


def _choose(options: dict[str, float], rng: np.random.Generator) -> str:
    """Draw one of the options with its probability."""
    threshold = rng.random()
    for option, probability in options.items():
        threshold -= probability
        if threshold < 0.0:
            return option
    # Rounding can leave the threshold just above zero after the last option.
    return option


MIN_MEDIAN_DAYS = 1.0 / 24.0
MAX_MEDIAN_DAYS = 10.0
MIN_SPREAD = 0.1
MAX_SPREAD = 1.0


def randomize(process: MarkovProcess, rng: np.random.Generator) -> MarkovProcess:
    """Draw new probabilities and times on the activities and transitions of the process.

    Only the structure of the process is kept: its activities and which transitions exist between
    them. The probabilities are drawn by `_random_probabilities` and the times by `_random_time`.

    Args:
        process: The process whose structure is kept.
        rng: Source of the new probabilities and times.

    Returns:
        A process with the same structure and the new probabilities and times.
    """
    return MarkovProcess(
        start=_random_probabilities(process.start, rng),
        activities={
            name: Activity(
                time=_random_time(rng),
                successors=_random_probabilities(activity.successors, rng),
            )
            for name, activity in process.activities.items()
        },
    )


def _random_probabilities(options: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    """Draw new probabilities on the options from a flat Dirichlet, so that every way of splitting
    the probability among them is equally likely."""
    probabilities = rng.dirichlet(np.ones(len(options)))
    return dict(zip(options, probabilities.tolist(), strict=True))


def _random_time(rng: np.random.Generator) -> LogNormal:
    """Draw a time distribution whose median is log-uniform between `MIN_MEDIAN_DAYS` and
    `MAX_MEDIAN_DAYS`, so that every order of magnitude is equally likely, and whose spread is
    uniform between `MIN_SPREAD` and `MAX_SPREAD`."""
    log_median = rng.uniform(math.log(MIN_MEDIAN_DAYS), math.log(MAX_MEDIAN_DAYS))
    return LogNormal(median=math.exp(log_median), spread=rng.uniform(MIN_SPREAD, MAX_SPREAD))
