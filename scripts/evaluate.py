from dataclasses import dataclass, field
from typing import Any

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from possible_futures.datasets.cases import Cases
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.evaluation.scoring import MeanScores, observed_suffixes, score_prefixes
from possible_futures.inference.draws import Decoding
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    Stage,
    scores_file,
    split_file,
    stage_dir,
    vocabulary_file,
)


@dataclass
class EvaluateConfig:
    data: DataConfig = MISSING
    workers: int = MISSING
    # The `Metric` of every column of the scores, built with `hydra.utils.instantiate`.
    metrics: dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(name='evaluate_schema', node=EvaluateConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='evaluate')
def main(config: EvaluateConfig) -> None:
    """Score every decoding against the test suffixes and write their mean scores."""
    name = config.data.name
    decoding_files = [
        path
        for stage in (Stage.GENERATE, Stage.COLLAPSE, Stage.BASELINES)
        for path in sorted(stage_dir(name, stage).glob('*.parquet'))
    ]
    metrics = {column: instantiate(metric) for column, metric in config.metrics.items()}
    output = scores_file(name)
    banner(
        f'Scoring the suffixes of "{name}"',
        {
            'decodings': ', '.join(path.stem for path in decoding_files),
            'metrics': ', '.join(metrics),
            'workers': config.workers,
            'scores': output,
        },
    )
    vocabulary = Vocabulary.load(vocabulary_file(name))
    test = Cases.load(split_file(name, Split.TEST))
    means = {}
    for path in decoding_files:
        with step(f'Scoring the {path.stem} suffixes'):
            prefixes = Decoding.load(path, vocabulary).prefixes
            observed = observed_suffixes(prefixes, test=test, vocabulary=vocabulary)
            per_prefix = score_prefixes(
                prefixes, observed=observed, metrics=metrics, workers=config.workers
            )
            means[path.stem] = per_prefix.mean()
    scores = MeanScores(table=pd.DataFrame.from_dict(means, orient='index'))
    scores.save(output)
    print(f'Wrote the mean scores to {output}\n\n{scores.table.round(4).to_string()}')


if __name__ == '__main__':
    main()
