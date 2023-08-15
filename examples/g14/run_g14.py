import json
import logging
import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Sequence, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from tqdm import tqdm

from nonlocalgames import methods
from nonlocalgames.hamiltonians import G14
from nonlocalgames import util

gym.logger.set_level(logging.CRITICAL)

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cpus', type=int, default=1)
    parser.add_argument('--seeds', default='../../data/seeds.txt')
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace):
    cpus = args.num_cpus
    seeds = util.load_seeds(args.seeds)

    print('Starting processing')
    with mp.Pool(processes=cpus) as p:
        results: Sequence[TaskResult] = list(tqdm(p.imap(task, seeds), total=len(seeds)))

    print('Postprocessing')
    dataframes = []
    best_energy = np.inf
    for result in results:
        energy = result.energy
        dataframes.append(result.df)

        if energy < best_energy:
            best_energy = energy

            with open('data/g14_state.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'state': result.state,
                    'phi': result.phi.reshape(2, 14, 2).tolist(),
                    'metrics': result.metrics
                }, f)

    print('Aggregating results')
    df = pd.concat(dataframes, axis=0, ignore_index=True)
    df.to_csv('data/g14_trials.csv', index=False)


@dataclass
class TaskResult:
    df: pd.DataFrame
    state: Sequence[Any]
    phi: np.ndarray
    metrics: dict

    @property
    def energy(self):
        return self.df.energy.min()


def task(seed: int) -> TaskResult:
    records = []
    ham = G14()
    state, phi, metrics = methods.dual_phase_optim(
        ham, verbose=0, seed=seed, tol=1e-6, save_mutual_information=True)

    for iter_ in range(len(metrics['energy'])):
        record = {
            'seed': seed,
            'iter': iter_
        }

        for k, series in metrics.items():
            record[k] = series[iter_]

        records.append(record)

    df = pd.DataFrame.from_records(records)
    return TaskResult(df, state, phi, metrics)

if __name__ == '__main__':
    args = get_cli_args()
    main(args)
