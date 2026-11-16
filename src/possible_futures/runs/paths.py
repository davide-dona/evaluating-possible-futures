from enum import StrEnum
from pathlib import Path

# Relative to the repository root, which every stage runs from.
DATA = Path('data')
OUTPUTS = Path('outputs')


class Split(StrEnum):
    """The two splits preprocessing cuts a log into, named as their files on disk are."""

    TRAIN = 'train'
    TEST = 'test'


def original_log(dataset: str) -> Path:
    return DATA / dataset / 'original.csv'


def split_file(dataset: str, split: Split) -> Path:
    return DATA / dataset / 'processed' / f'{split}.csv'


def vocabulary_file(dataset: str) -> Path:
    return DATA / dataset / 'processed' / 'vocabulary.json'


def max_case_length_file(dataset: str) -> Path:
    return DATA / dataset / 'processed' / 'max_case_length.txt'


def stage_dir(dataset: str, stage: str) -> Path:
    """The output directory of a stage, as `config/output/default.yaml` sets it."""
    return OUTPUTS / dataset / stage


def model_file(dataset: str) -> Path:
    return stage_dir(dataset, 'train') / 'model.pt'


def sampled_file(dataset: str) -> Path:
    return stage_dir(dataset, 'generate') / 'sampled.parquet'


def scores_file(dataset: str) -> Path:
    return stage_dir(dataset, 'evaluate') / 'scores.parquet'
