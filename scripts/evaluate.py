import hydra
import pandas as pd
from omegaconf import DictConfig

from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.evaluation.scores import score_generations, summary
from possible_futures.logs.io import read_split
from possible_futures.logs.keys import ACTIVITY_KEY, Split
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import split_file, stage_dir, vocabulary_file


def run(config: DictConfig) -> None:
    """Score the suffixes of every test prefix in each file of the generate, collapse, and
    baselines stages, and print the means."""
    name = config.data.name
    generations = [
        path
        for stage in ('generate', 'collapse', 'baselines')
        for path in sorted(stage_dir(name, stage).glob('*.parquet'))
    ]
    output = stage_dir(name, 'evaluate') / 'scores.parquet'
    banner(
        f'Scoring the suffixes of "{name}"',
        {
            'decodings': ', '.join(path.stem for path in generations),
            'workers': config.workers,
            'scores': output,
        },
    )
    vocabulary = Vocabulary.load(vocabulary_file(name))
    test = read_split(split_file(name, Split.TEST))
    test[ACTIVITY_KEY] = vocabulary.known(test[ACTIVITY_KEY])
    frames = []
    for path in generations:
        with step(f'Scoring the {path.stem} suffixes'):
            scores = score_generations(path, test, workers=config.workers)
            frames.append(scores.assign(decoding=path.stem))
    scores = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output, index=False)
    print(f'Wrote {len(scores):,} rows to {output}\n\n{summary(scores).round(4).to_string()}')


@hydra.main(version_base='1.3', config_path='../config', config_name='evaluate')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
