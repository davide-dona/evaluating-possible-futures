import hydra
import pandas as pd
from omegaconf import DictConfig

from pipelines.helpers.console import banner, step
from pipelines.helpers.paths import split_file, stage_dir
from src.evaluation.distances import DISTANCES
from src.evaluation.scores import score_generations
from src.logs.io import read_split
from src.logs.keys import Split

DECODINGS = ('sampled', 'greedy')


def run(config: DictConfig) -> None:
    """Score the sampled and greedy suffixes of every test prefix and print the means."""
    name = config.data.name
    generations = stage_dir(name, 'generate')
    output = stage_dir(name, 'evaluate') / 'scores.parquet'
    banner(f'Scoring the suffixes of "{name}"', {'generations': generations, 'scores': output})
    test = read_split(split_file(name, Split.TEST))
    frames = []
    for decoding in DECODINGS:
        with step(f'Scoring the {decoding} suffixes'):
            scores = score_generations(generations / f'{decoding}.parquet', test)
            frames.append(scores.assign(decoding=decoding))
    scores = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output, index=False)
    print(f'Wrote {len(scores):,} rows to {output}\n\n{summary(scores).to_string()}')


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """The mean DLS and fair energy scores of every decoding, one row each, every prefix weighed
    equally."""
    means = scores.groupby('decoding', sort=False).mean(numeric_only=True)
    columns = {'DLS': 1.0 - means['dl_accuracy']}
    for distance in DISTANCES:
        columns[f'ES {distance}'] = means[f'{distance}_accuracy'] - means[f'{distance}_spread']
    return pd.DataFrame(columns).round(4)


@hydra.main(version_base='1.3', config_path='../config', config_name='evaluate')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
