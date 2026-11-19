from enum import StrEnum
from pathlib import Path

# Relative to the repository root, which every stage runs from.
DATA = Path('data')
OUTPUTS = Path('outputs')


class Split(StrEnum):
    """The train and test splits, named as their files."""

    TRAIN = 'train'
    TEST = 'test'


class Stage(StrEnum):
    """The stages writing under `OUTPUTS`, named as their scripts."""

    TRAIN = 'train'
    FORECAST = 'forecast'
    EVALUATE = 'evaluate'


def original_log_file(dataset: str) -> Path:
    """The CSV log a dataset starts from.

    Args:
        dataset: The name of the dataset.

    Returns:
        Path: The original log file.
    """
    return DATA / dataset / 'original.csv'


def split_file(dataset: str, split: Split) -> Path:
    """The Parquet file of the cases of one split.

    Args:
        dataset: The name of the dataset.
        split: The split.

    Returns:
        Path: The cases file of the split.
    """
    return DATA / dataset / 'processed' / f'{split}.parquet'


def tokenizer_file(dataset: str) -> Path:
    """The tokenizer fitted on the train split.

    Args:
        dataset: The name of the dataset.

    Returns:
        Path: The JSON file of the tokenizer.
    """
    return DATA / dataset / 'processed' / 'tokenizer.json'


def stage_dir(dataset: str, stage: Stage) -> Path:
    """The output directory of a stage, the same as its Hydra run directory.

    Args:
        dataset: The name of the dataset.
        stage: The stage.

    Returns:
        Path: The directory the stage writes into.
    """
    return OUTPUTS / dataset / stage


def model_dir(dataset: str) -> Path:
    """The directory of the trained model and its tokenizer.

    Args:
        dataset: The name of the dataset.

    Returns:
        Path: The output directory of the train stage.
    """
    return stage_dir(dataset, Stage.TRAIN)


def forecasts_file(dataset: str, forecasts_name: str) -> Path:
    """The Parquet file of one set of forecasts.

    Args:
        dataset: The name of the dataset.
        forecasts_name: The name of the forecasts, such as `sampled`.

    Returns:
        Path: The forecasts file.
    """
    return stage_dir(dataset, Stage.FORECAST) / f'{forecasts_name}.parquet'


def scores_file(dataset: str, forecasts_name: str) -> Path:
    """The JSON file of the mean scores of one set of forecasts.

    Args:
        dataset: The name of the dataset.
        forecasts_name: The name of the forecasts scored.

    Returns:
        Path: The scores file.
    """
    return stage_dir(dataset, Stage.EVALUATE) / f'{forecasts_name}.json'
