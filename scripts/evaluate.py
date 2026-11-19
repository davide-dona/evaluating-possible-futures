from dataclasses import dataclass, field
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from possible_futures.cases import load_cases
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.evaluation.scoring import mean_scores, save_scores
from possible_futures.inference.draws import Decoding
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    decoding_file,
    scores_file,
    split_file,
    vocabulary_file,
)


@dataclass
class EvaluateConfig:
    data: DataConfig = MISSING
    # The name of the decoding scored, the stem of its file.
    decoding: str = MISSING
    workers: int = MISSING
    # The `Metric`s the decoding is scored with, built with `hydra.utils.instantiate`.
    metrics: list[Any] = field(default_factory=list)


ConfigStore.instance().store(name='evaluate_schema', node=EvaluateConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='evaluate')
def main(config: EvaluateConfig) -> None:
    """Score one decoding against the test suffixes and write its mean scores."""
    name, decoding = config.data.name, config.decoding
    path = decoding_file(name, decoding)
    metrics = [instantiate(metric) for metric in config.metrics]
    names = [metric.name for metric in metrics]
    if len(set(names)) < len(names):
        raise ValueError(f'The metrics share a name: {names}')
    output = scores_file(name, decoding)
    banner(
        f'Scoring the {decoding} suffixes of "{name}"',
        {
            'decoding': path,
            'metrics': ', '.join(names),
            'workers': config.workers,
            'scores': output,
        },
    )
    with step('Scoring every forecast against its observed suffix'):
        vocabulary = Vocabulary.load(vocabulary_file(name))
        test = load_cases(split_file(name, Split.TEST))
        means = mean_scores(
            decoding=Decoding.load(path, vocabulary),
            test=test,
            vocabulary=vocabulary,
            metrics=metrics,
            workers=config.workers,
        )
    save_scores(means=means, path=output)
    print(f'Wrote the mean scores to {output}\n')
    for metric, mean in means.items():
        print(f'  {metric}: {mean:.4f}')


if __name__ == '__main__':
    main()
