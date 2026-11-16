import hydra
import pandas as pd
from omegaconf import DictConfig

from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.evaluation.scores import score_generations
from possible_futures.logs.io import read_split
from possible_futures.logs.keys import ACTIVITY_KEY
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import Split, scores_file, split_file, stage_dir, vocabulary_file


@hydra.main(version_base='1.3', config_path='../config', config_name='evaluate')
def main(config: DictConfig) -> None:
    """Score the suffixes of every test prefix in each file of the generate, collapse, and
    baselines stages, and write the means of every decoding."""
    name = config.data.name
    generations = [
        path
        for stage in ('generate', 'collapse', 'baselines')
        for path in sorted(stage_dir(name, stage).glob('*.parquet'))
    ]
    output = scores_file(name)
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
    means = {}
    for path in generations:
        with step(f'Scoring the {path.stem} suffixes'):
            means[path.stem] = score_generations(path, test, workers=config.workers)
    scores = pd.DataFrame.from_dict(means, orient='index').rename_axis('decoding')
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output)
    print(f'Wrote the means to {output}\n\n{scores.round(4).to_string()}')


if __name__ == '__main__':
    main()
