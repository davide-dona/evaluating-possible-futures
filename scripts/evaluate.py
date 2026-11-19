from dataclasses import dataclass, field
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from possible_futures.datasets.cases import load_cases
from possible_futures.datasets.tokenizer import Tokenizer
from possible_futures.evaluation.scoring import mean_scores, save_scores
from possible_futures.forecasters.forecasts import load_forecasts
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    forecasts_file,
    scores_file,
    split_file,
    tokenizer_file,
)


@dataclass
class EvaluateConfig:
    """The settings of the evaluate stage.

    Attributes:
        data: The dataset.
        forecasts: The name of the forecasts scored, the stem of their file.
        workers: Processes scoring the forecasts in parallel.
        metrics: The `Metric`s the forecasts are scored with, built with
            `hydra.utils.instantiate`.
    """

    data: DataConfig = MISSING
    forecasts: str = MISSING
    workers: int = MISSING
    metrics: list[Any] = field(default_factory=list)


ConfigStore.instance().store(name='evaluate_schema', node=EvaluateConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='evaluate')
def main(config: EvaluateConfig) -> None:
    """Score one set of forecasts against the test suffixes and write its mean scores."""
    name, forecasts_name = config.data.name, config.forecasts
    path = forecasts_file(name, forecasts_name)
    metrics = [instantiate(metric) for metric in config.metrics]
    names = [metric.name for metric in metrics]
    if len(set(names)) < len(names):
        raise ValueError(f'The metrics share a name: {names}')
    output = scores_file(name, forecasts_name)
    banner(
        f'Scoring the {forecasts_name} suffixes of "{name}"',
        {
            'forecasts': path,
            'metrics': ', '.join(names),
            'workers': config.workers,
            'scores': output,
        },
    )
    with step('Scoring every forecast against its observed suffix'):
        means = mean_scores(
            load_forecasts(path),
            test=load_cases(split_file(name, Split.TEST)),
            tokenizer=Tokenizer.load(tokenizer_file(name)),
            metrics=metrics,
            workers=config.workers,
        )
    save_scores(means=means, path=output)
    print(f'Wrote the mean scores to {output}\n')
    for metric, mean in means.items():
        print(f'  {metric}: {mean:.4f}')


if __name__ == '__main__':
    main()
