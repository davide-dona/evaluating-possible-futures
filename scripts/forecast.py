from dataclasses import dataclass
from enum import StrEnum

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.cases import Case, load_cases, load_scored
from possible_futures.forecasters.distribution import GreedyForecaster, SamplingForecaster
from possible_futures.forecasters.forecaster import Forecaster
from possible_futures.forecasters.forecasts import write_forecasts
from possible_futures.forecasters.u_ed_lstm import UEDLSTMForecaster
from possible_futures.forecasters.unconditional import UnconditionalForecaster
from possible_futures.models.markov import MarkovChain
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    forecasts_file,
    model_dir,
    scored_file,
    split_file,
)
from possible_futures.simulation.synthetic import SyntheticProcess


class ForecasterModel(StrEnum):
    """What the draws come from."""

    U_ED_LSTM = 'u_ed_lstm'  # the trained model
    ORACLE = 'oracle'  # the true process of a simulated dataset
    MARKOV = 'markov'  # a first-order chain fitted on train
    UNCONDITIONAL = 'unconditional'  # the suffixes of train, whatever the prefix


class Strategy(StrEnum):
    """How a forecaster chooses the next activity at every step."""

    SAMPLED = 'sampled'  # at random, with its probability
    GREEDY = 'greedy'  # the most likely activity at every step


@dataclass
class ForecasterConfig:
    """Which forecaster draws the suffixes.

    Attributes:
        name: The name of the forecasts, the stem of their file.
        model: What the draws come from.
        strategy: How the next activity is chosen at every step.
    """

    name: str = MISSING
    model: ForecasterModel = MISSING
    strategy: Strategy = MISSING


@dataclass
class ForecastConfig:
    """The settings of the forecast stage.

    Attributes:
        data: The dataset.
        forecaster: Which forecaster draws the suffixes.
        seed: Seeds every random choice.
        device: The device of the model, such as `cuda:0` or `cpu`.
        num_draws: The number of suffixes drawn for each prefix, `S`.
        max_rows: The most suffixes drawn at once.
    """

    data: DataConfig = MISSING
    forecaster: ForecasterConfig = MISSING
    seed: int = MISSING
    device: str = MISSING
    num_draws: int = MISSING
    max_rows: int = MISSING


ConfigStore.instance().store(name='forecast_schema', node=ForecastConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='forecast')
def main(config: ForecastConfig) -> None:
    """Draw suffixes for every test prefix from one forecaster and write them as forecasts."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    name, forecaster_config = config.data.name, config.forecaster
    path = forecasts_file(name, forecaster_config.name)
    banner(
        f'Forecasting "{name}" as {forecaster_config.name}',
        {
            'model': forecaster_config.model,
            'strategy': forecaster_config.strategy,
            'device': device,
            'draws': f'{config.num_draws} per prefix',
            'output': path,
        },
    )
    with step('Building the forecaster and loading the test prefixes'):
        train = load_cases(split_file(name, Split.TRAIN))
        test = load_cases(split_file(name, Split.TEST))
        scored = load_scored(scored_file(name))
        forecaster = build_forecaster(
            forecaster_config, data=config.data, train=train, rng=rng, device=device
        )

    with step(f'Writing the draws to {path}'):
        num_prefixes = write_forecasts(
            forecaster,
            test=test,
            scored=scored,
            path=path,
            num_draws=config.num_draws,
            max_steps=max(len(case) for case in train),
            max_rows=config.max_rows,
        )
        print(f'  {num_prefixes:,} prefixes', flush=True)


def build_forecaster(
    config: ForecasterConfig,
    *,
    data: DataConfig,
    train: list[Case],
    rng: np.random.Generator,
    device: torch.device,
) -> Forecaster:
    """Build the forecaster `config` asks for, fitting it on the train split where it needs one.

    Args:
        config: Which forecaster to build.
        data: The dataset.
        train: The train cases.
        rng: Makes the random choices of the forecaster.
        device: The device the model is loaded onto.

    Returns:
        Forecaster: The forecaster.

    Raises:
        ValueError: If the dataset has no oracle, or the forecaster has no such strategy.
    """
    match config.model, config.strategy:
        case ForecasterModel.U_ED_LSTM, Strategy.SAMPLED:
            return UEDLSTMForecaster.load(model_dir(data.name), device=device)
        case ForecasterModel.UNCONDITIONAL, Strategy.SAMPLED:
            return UnconditionalForecaster.fit(train, rng=rng)
        case ForecasterModel.ORACLE | ForecasterModel.MARKOV, strategy:
            if config.model is ForecasterModel.MARKOV:
                distribution = MarkovChain.fit(train)
            elif data.process is None:
                raise ValueError(f'"{data.name}" is not simulated, so it has no oracle')
            else:
                distribution = SyntheticProcess(**data.process)
            if strategy is Strategy.SAMPLED:
                return SamplingForecaster(distribution=distribution, rng=rng)
            return GreedyForecaster(distribution=distribution)
    raise ValueError(f'The {config.model} forecaster has no {config.strategy} strategy')


if __name__ == '__main__':
    main()
