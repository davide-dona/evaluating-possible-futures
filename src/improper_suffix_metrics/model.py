import math
from dataclasses import dataclass

import numpy as np

from improper_suffix_metrics.markov_process import END, Activity, MarkovProcess


@dataclass(frozen=True)
class Graph:
    """The structure of a process, without probabilities or times.

    Attributes:
        start: The activities that can open a case.
        successors: For each activity, the activities or `END` that can follow it.
    """

    start: tuple[str, ...]
    successors: dict[str, tuple[str, ...]]


LOAN_APPLICATION = Graph(
    start=('Submit application',),
    successors={
        'Submit application': ('Automatic check', 'Request documents'),
        'Automatic check': ('Approve',),
        'Approve': (END,),
        'Request documents': ('Manual assessment',),
        'Manual assessment': ('Final decision',),
        'Final decision': (END,),
    },
)

MEDIAN_DAYS_RANGE = (1.0 / 24.0, 10.0)
SPREAD_RANGE = (0.1, 1.0)


def random_markov_process(graph: Graph, rng: np.random.Generator) -> MarkovProcess:
    """Draw a Markov process on the graph.

    The probabilities of the edges leaving each state are drawn from a flat Dirichlet. Each
    activity's median time is log-uniform and its log-scale spread uniform.
    """
    low, high = (math.log(days) for days in MEDIAN_DAYS_RANGE)

    def probabilities(options: tuple[str, ...]) -> dict[str, float]:
        return dict(zip(options, rng.dirichlet(np.ones(len(options))).tolist(), strict=True))

    return MarkovProcess(
        start=probabilities(graph.start),
        activities={
            name: Activity(
                median_days=math.exp(rng.uniform(low, high)),
                spread=rng.uniform(*SPREAD_RANGE),
                successors=probabilities(successors),
            )
            for name, successors in graph.successors.items()
        },
    )
