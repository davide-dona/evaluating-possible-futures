from dataclasses import dataclass
from enum import StrEnum

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.cases import Case, load_cases
from possible_futures.forecasters.forecaster import Forecaster, write_forecasts
from possible_futures.forecasters.process import GreedyForecaster, SamplingForecaster
from possible_futures.forecasters.u_ed_lstm import UEDLSTMForecaster
from possible_futures.forecasters.unconditional import UnconditionalForecaster
from possible_futures.processes.markov import MarkovChain
from possible_futures.processes.synthetic import SyntheticProcess
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import (
    Split,
    forecasts_file,
    model_dir,
    split_file,
)


class ForecasterModel(StrEnum):
    """What the draws come from."""

    U_ED_LSTM = 'u_ed_lstm'  # the trained model
    ORACLE = 'oracle'  # the true process of a simulated dataset
    MARKOV = 'markov'  # a first-order chain fitted on train
    UNCONDITIONAL = 'unconditional'  # the suffixes of train, whatever the prefix


class Strategy(StrEnum):
    SAMPLED = 'sampled'
    GREEDY = 'greedy'  # the most probable activity at every step


@dataclass
class ForecasterConfig:
    # The name of the forecasts, the stem of their file.
    name: str = MISSING
    model: ForecasterModel = MISSING
    strategy: Strategy = MISSING


@dataclass
class ForecastConfig:
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
        forecaster = build_forecaster(
            forecaster_config, data=config.data, train=train, rng=rng, device=device
        )

    with step(f'Writing the draws to {path}'):
        num_prefixes = write_forecasts(
            forecaster,
            test=test,
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
    """The forecaster `config` asks for, built from the train split where it needs one."""
    match config.model, config.strategy:
        case ForecasterModel.U_ED_LSTM, Strategy.SAMPLED:
            return UEDLSTMForecaster.load(model_dir(data.name), device=device)
        case ForecasterModel.UNCONDITIONAL, Strategy.SAMPLED:
            return UnconditionalForecaster.fit(train, rng=rng)
        case ForecasterModel.ORACLE | ForecasterModel.MARKOV, strategy:
            if config.model is ForecasterModel.MARKOV:
                process = MarkovChain.fit(train)
            elif data.process is None:
                raise ValueError(f'"{data.name}" is not simulated, so it has no oracle')
            else:
                process = SyntheticProcess(**data.process)
            if strategy is Strategy.SAMPLED:
                return SamplingForecaster(process=process, rng=rng)
            return GreedyForecaster(process=process)
    raise ValueError(f'The {config.model} forecaster has no {config.strategy} strategy')


if __name__ == '__main__':
    main()
