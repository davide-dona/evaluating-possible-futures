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


def decoding_file(dataset: str, decoding: str) -> Path:
    return stage_dir(dataset, Stage.DECODE) / f'{decoding}.parquet'


def scores_file(dataset: str, decoding: str) -> Path:
    return stage_dir(dataset, Stage.EVALUATE) / f'{decoding}.json'
