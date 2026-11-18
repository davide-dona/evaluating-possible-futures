from enum import StrEnum
from pathlib import Path

# Relative to the repository root, which every stage runs from.
DATA = Path('data')
OUTPUTS = Path('outputs')


class Split(StrEnum):
    """The train and test splits, named as their directories."""

    TRAIN = 'train'
    TEST = 'test'


class Stage(StrEnum):
    """The stages writing under `OUTPUTS`, named as their scripts."""

    TRAIN = 'train'
    DECODE = 'decode'
    COLLAPSE = 'collapse'
    EVALUATE = 'evaluate'


def original_log_file(dataset: str) -> Path:
    return DATA / dataset / 'original.csv'


def split_file(dataset: str, split: Split) -> Path:
    return DATA / dataset / 'processed' / f'{split}.parquet'


def vocabulary_file(dataset: str) -> Path:
    return DATA / dataset / 'processed' / 'vocabulary.json'


def stage_dir(dataset: str, stage: Stage) -> Path:
    """The output directory of a stage, the same as its Hydra run directory."""
    return OUTPUTS / dataset / stage


def model_dir(dataset: str) -> Path:
    return stage_dir(dataset, Stage.TRAIN)


def decoding_file(dataset: str, decoder: str) -> Path:
    return stage_dir(dataset, Stage.DECODE) / f'{decoder}.parquet'


def sampled_file(dataset: str) -> Path:
    """The decoding the trained model samples."""
    return decoding_file(dataset, 'sampled')


def scores_file(dataset: str) -> Path:
    return stage_dir(dataset, Stage.EVALUATE) / 'scores.csv'
