from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ProcessConfig:
    """The parameters of the synthetic process."""

    p_b: float = MISSING
    q_b: float = MISSING
    q_c: float = MISSING


@dataclass
class DataConfig:
    """The settings of one dataset."""

    name: str = MISSING
    case_key: str = MISSING
    activity_key: str = MISSING
    timestamp_key: str = MISSING
    separator: str = MISSING
    test_fraction: float = MISSING
    max_case_length_percentile: float = MISSING
    max_case_duration_percentile: float = MISSING
    # Set only on a dataset simulated from a known process.
    num_cases: int | None = None
    simulation_seed: int | None = None
    process: ProcessConfig | None = None
