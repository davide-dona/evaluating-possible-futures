from pathlib import Path

from possible_futures.logs.keys import Split

# Relative to the repository root, which every stage runs from.
DATA = Path('data')
OUTPUTS = Path('outputs')


def original_log(dataset: str) -> Path:
    return DATA / dataset / 'original.csv'


def split_file(dataset: str, split: Split) -> Path:
    return DATA / dataset / 'processed' / f'{split}.csv'


def vocabulary_file(dataset: str) -> Path:
    return DATA / dataset / 'processed' / 'vocabulary.json'


def stage_dir(dataset: str, stage: str) -> Path:
    """The output directory of a stage, as `config/output/default.yaml` sets it."""
    return OUTPUTS / dataset / stage
