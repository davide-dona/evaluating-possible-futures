from dataclasses import dataclass

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.cases import num_events
from possible_futures.preprocessing.log import write_log
from possible_futures.processes.synthetic import SyntheticProcess
from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import original_log_file


@dataclass
class SimulateConfig:
    data: DataConfig = MISSING


ConfigStore.instance().store(name='simulate_schema', node=SimulateConfig)


@hydra.main(version_base='1.3', config_path='../config', config_name='simulate')
def main(config: SimulateConfig) -> None:
    """Draw the synthetic log from its process and write it as an original log."""
    data = config.data
    process = SyntheticProcess(**data.process)
    path = original_log_file(data.name)
    banner(
        f'Simulating "{data.name}"',
        {
            'process': process,
            'cases': f'{data.num_cases:,}',
            'seed': data.simulation_seed,
            'original log': path,
        },
    )
    with step('Drawing the cases'):
        rng = np.random.default_rng(data.simulation_seed)
        cases = process.simulate(num_cases=data.num_cases, rng=rng)
        print(f'  {num_events(cases):,} events over {len(cases):,} cases', flush=True)
    with step(f'Writing the log to {path}'):
        write_log(
            cases,
            path=path,
            case_key=data.case_key,
            activity_key=data.activity_key,
            timestamp_key=data.timestamp_key,
            separator=data.separator,
        )


if __name__ == '__main__':
    main()
