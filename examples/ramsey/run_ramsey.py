import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import pickle as pkl
import sys
from dataclasses import dataclass, asdict
from typing import Any, Sequence
import shutil

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm

from nonlocalgames import methods, util
from nonlocalgames.hamiltonians import Ramsey, MiniRamsey

gym.logger.set_level(logging.CRITICAL)

TMPDIR = 'tmpdata'
DATADIR = 'data'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                return obj.item()

            return obj.tolist()

        return super().default(obj)

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cpus', type=int, default=1,
                        help='Number of processes to use')
    parser.add_argument('--seeds', default='../../data/seeds.txt',
                        help='File containing random seeds to read from')
    parser.add_argument('-n', '--trials', type=int, default=None,
                        help='How many trials to run with different seeds')

    parser.add_argument('--dpo-tol', type=float, default=1e-6,
                        help='Maximum change in energy for convergence')
    parser.add_argument('--adapt-tol', type=float, default=1e-3,
                        help='Maximum allowed gradient of ADAPT for convergence')
    parser.add_argument('--phi-tol', type=float, default=1e-5,
                        help='Maximum gradient of measurement parameters for convergence')
    parser.add_argument('--theta-tol', type=float, default=1e-9,
                        help='Maximum gradient for theta optimization during ADAPT')

    parser.add_argument('--layer', default='ry',
                        help='Measurement layer type (see measurement.py)')
    
    parser.add_argument('--mini', action='store_true')

    args = parser.parse_args()
    return args

def create_trials(args: argparse.Namespace):
    global TMPDIR, DATADIR
    TMPDIR = f'{TMPDIR}_{args.layer}'
    DATADIR = f'{DATADIR}_{args.layer}'

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
                           dpo_tol=args.dpo_tol,
                           adapt_tol=args.adapt_tol,
                           phi_tol=args.phi_tol,
                           theta_tol=args.theta_tol,
                           layer=args.layer,
                           mini=args.mini),
        remaining
    ))

    return task_args

def main(args: argparse.Namespace):
    cpus = args.num_cpus
    task_args = create_trials(args)

    os.makedirs(DATADIR, exist_ok=True)

    print('Starting processing')
    # Call tqdm(pool.imap) to construct a progress bar. We then wrap that
    # in list() to pull results from the pool as they come in.
    with mp.Pool(processes=cpus) as p:
        results: Sequence[TaskResult] = list(tqdm(
            p.imap(task, task_args),
            total=len(task_args),
            file=sys.stdout
        ))

    print('Loading results from directory')
    other_results = load_results_from_dir()
    print(f'Loaded {len(other_results)} results')
    results: set = set(results)
    results.update(other_results)
    results = list(results)

    phi_shape = results[0].metadata.ham().desired_shape

    print('Postprocessing')
    dataframes = []
    best_energy = np.inf
    for result in results:
        energy = result.energy
        dataframes.append(result.df)

        if energy < best_energy:
            best_energy = energy

            with open(f'{DATADIR}/g14_state.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'state': result.state,
                    'phi': result.phi.reshape(phi_shape).tolist(),
                    'metrics': result.metrics,
                    'metadata': result.metadata.to_dict()
                }, f, cls=NumpyEncoder)

    print('Aggregating results')
    df = pd.concat(dataframes, axis=0, ignore_index=True)
    df.to_csv(f'{DATADIR}/ramsey_trials.csv', index=False)

    # Cleanup intermediate results directory
    shutil.rmtree(TMPDIR)

def load_results_from_dir():
    results = []
    for file in glob.glob(os.path.join(TMPDIR, '*.pkl')):
        with open(file, 'rb') as f:
            results.append(pkl.load(f))

    return results

@dataclass
class TaskArgs:
    seed: int
    dpo_tol: float = 1e-6
    adapt_tol: float = 1e-3
    phi_tol: float = 1e-5
    theta_tol: float = 1e-9
    layer: str = 'ry'
    mini: bool = False

    def ham(self):
        cls = MiniRamsey if self.mini else Ramsey
        return cls(
            measurement_layer=self.layer
        )
    
    def to_dict(self):
        return asdict(self)

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

    def __hash__(self):
        return self.metadata.seed

    def __eq__(self, other):
        return self.metadata.seed == other.metadata.seed


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
        adapt_thresh=args.adapt_tol,
        theta_thresh=args.theta_tol,
        phi_tol=args.phi_tol)

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
