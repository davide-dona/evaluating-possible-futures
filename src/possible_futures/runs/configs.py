from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ProcessConfig:
    """The parameters of the synthetic process.

    Attributes:
        p_b: Probability of B after A.
        q_b: Probability of E after D on a case through B.
        q_c: Probability of E after D on a case through C.
    """

    p_b: float = MISSING
    q_b: float = MISSING
    q_c: float = MISSING


@dataclass
class SplitConfig:
    """How a log is filtered and split into train and test.

    Attributes:
        test_fraction: The share of the cases, by start time, that start the test split.
        max_case_length_percentile: The percentile of the case lengths above which a case is
            dropped.
        max_case_duration_percentile: The percentile of the case durations above which a case is
            dropped. That duration is also the length of the end window, where cases still
            running would be cut short, so its cases are dropped and its prefixes not scored.
    """

    test_fraction: float = MISSING
    max_case_length_percentile: float = MISSING
    max_case_duration_percentile: float = MISSING


@dataclass
class DataConfig:
    """The settings of one dataset.

    Attributes:
        name: The name of the dataset, also its directory under `data` and `outputs`.
        case_key: The column of the case IDs in the original log.
        activity_key: The column of the activities.
        timestamp_key: The column of the timestamps.
        separator: The separator of the original CSV log.
        split: How the log is filtered and split.
        num_cases: The number of cases simulated, on a simulated dataset only.
        simulation_seed: The seed of the simulation, on a simulated dataset only.
        process: The process simulated, on a simulated dataset only.
    """

    name: str = MISSING
    case_key: str = MISSING
    activity_key: str = MISSING
    timestamp_key: str = MISSING
    separator: str = MISSING
    split: SplitConfig = MISSING
    num_cases: int | None = None
    simulation_seed: int | None = None
    process: ProcessConfig | None = None
