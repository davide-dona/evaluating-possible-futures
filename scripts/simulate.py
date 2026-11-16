import hydra
import numpy as np
from omegaconf import DictConfig

from possible_futures.logs.keys import CASE_KEY
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import original_log
from possible_futures.simulation.processes import SyntheticProcess


@hydra.main(version_base='1.3', config_path='../config', config_name='simulate')
def main(config: DictConfig) -> None:
    """Draw the synthetic log from its process and write it as an original log."""
    data = config.data
    process = SyntheticProcess(**data.process)
    path = original_log(data.name)
    banner(
        f'Simulating "{data.name}"',
        {
            'process': f'P(B | A) = {process.p_b}, P(E | B, D) = {process.q_b}, '
            f'P(E | C, D) = {process.q_c}',
            'cases': f'{data.num_cases:,}',
            'seed': data.simulation_seed,
            'original log': path,
        },
    )
    with step('Drawing the cases'):
        rng = np.random.default_rng(data.simulation_seed)
        log = process.simulate_log(num_cases=data.num_cases, rng=rng)
        print(f'  {len(log):,} events over {log[CASE_KEY].nunique():,} cases', flush=True)
    with step(f'Writing the log to {path}'):
        path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(path, sep=data.separator, index=False)


if __name__ == '__main__':
    main()
