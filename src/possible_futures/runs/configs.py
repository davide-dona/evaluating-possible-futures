from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class ProcessConfig:
    """The parameters of the synthetic process."""

    p_b: float = MISSING
    q_b: float = MISSING
    q_c: float = MISSING


@dataclass
class DataConfig:
    """A dataset, as one file of `config/dataset/` sets it."""

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


@dataclass
class SimulateConfig:
    data: DataConfig = MISSING


@dataclass
class PreprocessConfig:
    data: DataConfig = MISSING


@dataclass
class ModelConfig:
    num_layers: int = MISSING
    hidden_size: int = MISSING
    dropout: float = MISSING
    logit_samples: int = MISSING


@dataclass
class TeacherForcingConfig:
    start: float = MISSING
    decay: float = MISSING
    every: int = MISSING


@dataclass
class TrainingConfig:
    epochs: int = MISSING
    lr: float = MISSING
    batch_size: int = MISSING
    window: int = MISSING
    weight_penalty: float = MISSING
    grad_clip_norm: float = MISSING
    teacher_forcing: TeacherForcingConfig = MISSING


@dataclass
class TrainConfig:
    data: DataConfig = MISSING
    seed: int = MISSING
    device: str = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING


@dataclass
class GenerateConfig:
    data: DataConfig = MISSING
    seed: int = MISSING
    device: str = MISSING
    num_draws: int = MISSING
    max_rows: int = MISSING


@dataclass
class CollapseConfig:
    data: DataConfig = MISSING
    # The `Distance` the medoid is taken under, built with `hydra.utils.instantiate`.
    distance: Any = MISSING


@dataclass
class BaselinesConfig:
    data: DataConfig = MISSING
    seed: int = MISSING
    num_draws: int = MISSING


@dataclass
class EvaluateConfig:
    data: DataConfig = MISSING
    workers: int = MISSING
    # The `Metric` of every column of the scores, built with `hydra.utils.instantiate`.
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoresTableConfig:
    datasets: list[str] = MISSING
    decodings: list[str] = MISSING


@dataclass
class MetricColumnConfig:
    header: str = MISSING
    higher_is_better: bool = MISSING


@dataclass
class TablesConfig:
    datasets: dict[str, str] = MISSING
    scores: dict[str, ScoresTableConfig] = MISSING
    properties: list[str] = MISSING
    decodings: dict[str, str] = MISSING
    metrics: dict[str, MetricColumnConfig] = MISSING
