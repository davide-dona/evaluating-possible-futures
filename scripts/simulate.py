import hydra
import numpy as np
from omegaconf import DictConfig

from possible_futures.logs.keys import CASE_KEY
from possible_futures.runs.console import banner, step
from possible_futures.runs.paths import original_log
from possible_futures.simulation.processes import SyntheticProcess


def run(data: DictConfig) -> None:
    """Generate the synthetic log according to the specified process and
    write it to a CSV file.

    Args:
        data: The configuration for the synthetic log generation.
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
    # Generate the synthetic log
    with step('Drawing the cases'):
        rng = np.random.default_rng(data.simulation_seed)
        log = process.simulate_log(num_cases=data.num_cases, rng=rng)
        print(f'  {len(log):,} events over {log[CASE_KEY].nunique():,} cases', flush=True)
    # Write the synthetic log to a CSV file
    with step(f'Writing the log to {path}'):
        path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(path, sep=data.separator, index=False)


@hydra.main(version_base='1.3', config_path='../config', config_name='simulate')
def main(config: DictConfig) -> None:
    run(config.data)


if __name__ == '__main__':
    main()
