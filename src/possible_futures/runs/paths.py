from enum import StrEnum
from pathlib import Path

# Relative to the repository root, which every stage runs from.
DATA = Path('data')
# Every stage writes into `OUTPUTS / dataset / <script name>`, its Hydra run directory.
OUTPUTS = Path('outputs')


class Split(StrEnum):
    """The train and test splits, named as their files."""

    TRAIN = 'train'
    TEST = 'test'


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


def scored_file(dataset: str) -> Path:
    """The Parquet file of the lengths of the scored test prefixes.

    Args:
        dataset: The name of the dataset.

    Returns:
        Path: The scored prefixes file.
    """
    return DATA / dataset / 'processed' / 'scored.parquet'


def tokenizer_file(dataset: str) -> Path:
    """The tokenizer fitted on the train split.

    Args:
        dataset: The name of the dataset.

    Returns:
        Path: The JSON file of the tokenizer.
    """
    return DATA / dataset / 'processed' / 'tokenizer.json'


def model_dir(dataset: str) -> Path:
    """The directory of the trained model and its tokenizer.

    Args:
        dataset: The name of the dataset.

    Returns:
        Path: The output directory of the train stage.
    """
    return OUTPUTS / dataset / 'train'


def forecasts_file(dataset: str, forecasts_name: str) -> Path:
    """The Parquet file of one set of forecasts.

    Args:
        dataset: The name of the dataset.
        forecasts_name: The name of the forecasts, such as `sampled`.

    Returns:
        Path: The forecasts file.
    """
    return OUTPUTS / dataset / 'forecast' / f'{forecasts_name}.parquet'


def scores_file(dataset: str, forecasts_name: str) -> Path:
    """The JSON file of the mean scores of one set of forecasts.

    Args:
        dataset: The name of the dataset.
        forecasts_name: The name of the forecasts scored.

    Returns:
        Path: The scores file.
    """
    return OUTPUTS / dataset / 'evaluate' / f'{forecasts_name}.json'
