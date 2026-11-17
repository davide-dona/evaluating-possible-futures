from dataclasses import dataclass

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from possible_futures.runs.configs import DataConfig
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import original_log_file
from possible_futures.synthetic import SyntheticProcess


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
        events = process.simulate(num_cases=data.num_cases, rng=rng)
        log = pd.DataFrame(
            {
                data.case_key: events.case_ids,
                data.activity_key: events.activities,
                data.timestamp_key: events.timestamps,
            }
        )
        print(f'  {len(log):,} events over {data.num_cases:,} cases', flush=True)
    with step(f'Writing the log to {path}'):
        path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(path, sep=data.separator, index=False)


if __name__ == '__main__':
    main()
