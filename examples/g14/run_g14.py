import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import pickle as pkl
import sys
from dataclasses import dataclass
from typing import Any, Dict, Sequence
import shutil

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm

from nonlocalgames import methods, util
from nonlocalgames.hamiltonians import G14

gym.logger.set_level(logging.CRITICAL)

TMPDIR = 'tmpdata'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                return obj.item()

            return obj.tolist()

        return super().default(obj)

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cpus', type=int, default=1)
    parser.add_argument('--seeds', default='../../data/seeds.txt')
    parser.add_argument('--weighting', default=None)
    parser.add_argument('-n', '--trials', type=int, default=None)
    parser.add_argument('--dpo-tol', type=float, default=1e-6)
    parser.add_argument('--adapt-tol', type=float, default=1e-3)
    parser.add_argument('--constrained', action='store_true')
    args = parser.parse_args()
    return args

def create_trials(args: argparse.Namespace):
    seeds = util.load_seeds(args.seeds)
    trials = args.trials or len(seeds)

    # Construct temporary directory to hold intermediate
    # results in case stuff goes sideways
    os.makedirs(TMPDIR, exist_ok=True)

    assert trials <= len(seeds), \
        f'Not enough seeds saved to perform {trials} trials (max {len(seeds)})'

    seeds = set(seeds[:trials])

    # Now we find all trials in the intermediate directory and don't repeat them
    # to resume from where we started
    already_done = set()
    for file in glob.glob(os.path.join(TMPDIR, '*.pkl')):
        seed: str = os.path.splitext(os.path.basename(file))[0]
        seed = int(seed)
        already_done.add(seed)
    
    remaining = seeds.difference(already_done)

    task_args = list(map(
        lambda s: TaskArgs(s,
                           weighting=args.weighting,
                           dpo_tol=args.dpo_tol,
                           adapt_tol=args.adapt_tol,
                           constrain_phi=args.constrained),
        remaining
    ))

    return task_args

def main(args: argparse.Namespace):
    cpus = args.num_cpus
    task_args = create_trials(args)
    phi_shape = task_args[0].ham().desired_shape

    os.makedirs('data', exist_ok=True)

    print('Starting processing')
    # Call tqdm(pool.imap) to construct a progress bar. We then wrap that
    # in list() to pull results from the pool as they come in.
    with mp.Pool(processes=cpus) as p:
        results: Sequence[TaskResult] = list(tqdm(
            p.imap(task, task_args),
            total=len(task_args),
            file=sys.stdout
        ))

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
                    'phi': result.phi.reshape(phi_shape).tolist(),
                    'metrics': result.metrics
                }, f, cls=NumpyEncoder)

    print('Aggregating results')
    df = pd.concat(dataframes, axis=0, ignore_index=True)
    df.to_csv('data/g14_trials.csv', index=False)

    # Cleanup intermediate results directory
    shutil.rmtree(TMPDIR)

@dataclass
class TaskArgs:
    seed: int
    weighting: str | None = None
    dpo_tol: float = 1e-6
    adapt_tol: float = 1e-3
    constrain_phi: bool = True

    def ham(self):
        return G14(
            weighting=self.weighting,
            measurement_layer='ry',
            constrain_phi=self.constrain_phi
        )

@dataclass
class TaskResult:
    df: pd.DataFrame
    state: Sequence[Any]
    phi: np.ndarray
    metrics: dict
    metadata: TaskArgs

    @property
    def energy(self):
        return self.df.energy.min()


def task(args: TaskArgs) -> TaskResult:
    seed = args.seed
    records = []
    ham = args.ham()
    state, phi, metrics = methods.dual_phase_optim(
        ham,
        verbose=0,
        seed=seed,
        tol=args.dpo_tol,
        save_mutual_information=True,
        adapt_thresh=args.adapt_tol)

    for iter_ in range(len(metrics['energy'])):
        record = {
            'seed': seed,
            'iter': iter_
        }

        for k, series in metrics.items():
            if k not in ('theta_grad', 'phi_grad'):
                record[k] = series[iter_]

        records.append(record)

    df = pd.DataFrame.from_records(records)
    result = TaskResult(df, state, phi, metrics, args)

    # Dump to temporary directory
    with open(os.path.join(TMPDIR, f'{seed}.pkl'), 'wb') as f:
        pkl.dump(result, f)

    return result

if __name__ == '__main__':
    args = get_cli_args()
    main(args)
