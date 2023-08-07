import argparse
import multiprocessing as mp
import os
import json

import numpy as np

import methods
from hamiltonians import NPartiteSymmetricNLG

def get_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--trials', type=int)
    parser.add_argument('-p', '--players', type=int)
    parser.add_argument('--cpus', type=int)

    return parser.parse_args()

def task(i: int, N: int):
    print(f'Starting trial {i}')
    game = NPartiteSymmetricNLG(N)

    converged = None
    energies = None

    obj = {}

    try:
        energies, ansatz, phi = methods.dual_phase_optim(game, seed=i)
        obj['energies'] = energies
        obj['phi'] = (phi % (2 * np.pi)).tolist()
        obj['ansatz'] = ansatz
        converged = True

    except RuntimeWarning as w:
        obj['error'] = w.args[0]
        converged = False
    
    obj['converged'] = converged

    with open(f'{i}.json', 'w', encoding='utf-8') as f:
        json.dump(obj, f)

    return i

def merge(trials: list):
    results = []
    filenames = list(map(lambda i: f'{i}.json', trials))

    # Load the individual trials here
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)

    # Merge into a single file
    with open('dpo.json', 'w', encoding='utf-8') as f:
        json.dump(results, f)
    
    # Delete original files only after merging in case something
    # goes wrong above
    for filename in filenames:
        os.remove(filename)

def main():
    args = get_cli_args()
    cpus = args.cpus

    with mp.Pool(cpus) as pool:
        N = args.players
        trials = args.trials

        print(f'Starting {trials} trials for N={N} with {cpus} processes')
        completed_trials = pool.starmap(task, [(i, N) for i in range(trials)])

    print('Done, merging')
    merge(completed_trials)

if __name__ == '__main__':
    main()
