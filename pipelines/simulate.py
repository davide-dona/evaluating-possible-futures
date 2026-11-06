import hydra
import numpy as np
from omegaconf import DictConfig

from pipelines.helpers.console import banner, step
from pipelines.helpers.paths import original_log
from src.logs.keys import CASE_KEY
from src.simulation.processes import SyntheticProcess, simulate_log


def run(data: DictConfig) -> None:
    """Draw the synthetic log and write it where preprocessing reads an original log.

    Args:
        data: The `data` section of the synthetic dataset.
    """
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
        log = simulate_log(process, num_cases=data.num_cases, rng=rng)
        print(f'  {len(log):,} events over {log[CASE_KEY].nunique():,} cases', flush=True)
    with step(f'Writing the log to {path}'):
        path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(path, sep=data.separator, index=False)


@hydra.main(version_base='1.3', config_path='../config', config_name='simulate')
def main(config: DictConfig) -> None:
    run(config.data)


if __name__ == '__main__':
    main()
