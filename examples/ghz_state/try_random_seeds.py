import argparse
import csv
import json
import multiprocessing as mp
from collections import OrderedDict
from dataclasses import dataclass

import jax
import jaxopt
import networkx as nx
import numpy as np
import pennylane as qml
from jax import random

jax.config.update("jax_enable_x64", True)

n_wires = 4
dev = qml.device("default.qubit", wires=n_wires)

def u3ry(params: jax.Array, wires):
    """U3Ry measurement layer from nonlocalgames paper"""
    # Params should have shape (wires, 4)
    qml.U3(*params[0, 0:3], wires=wires[0])
    qml.U3(*params[1, 0:3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[0, 3], wires=wires[0])
    qml.RY(params[1, 3], wires=wires[1])

@jax.jit
def conj_params(params: jax.Array):
    """Conjugates measurement parameters"""
    # Params has shape (wires, 4)
    copy = params.copy()
    copy = copy.at[:, 1:3].set(-copy[:, 1:3])
    return copy

def ghz(wires):
    """GHZ state of 4 qubits"""
    qml.Hadamard(wires=wires[0])

    for ctrl, tgt in zip(wires[:-1], wires[1:]):
        qml.CNOT(wires=[ctrl, tgt])

def pcc(wires, chi_q = 4):
    projectors = []

    for c in range(chi_q):
        b1, b0 = c // 2, c % 2
        proj = qml.Projector(state=[b1, b0, b1, b0], wires=wires)
        projectors.append(proj)

    return sum(projectors)

Pcc = pcc(wires=(0, 1, 2, 3))

@qml.qnode(dev, interface='jax')
def circuit(params: jax.Array, va: int, vb: int):
    # State preparation
    ghz(wires=(0, 1, 2, 3))
    qml.Barrier(wires=(0, 1, 2, 3))

    # Apply measurement layer
    u3ry(params[va], wires=(0, 1))
    u3ry(conj_params(params[vb]), wires=(2, 3))

    return qml.expval(Pcc)

def value(G: nx.Graph, params: jax.Array):
    Q = len(G) + 2 * len(G.edges)
    val = 0
    for v in G:
        val += circuit(params, v, v)

    for va, vb in G.edges:
        val += 1 - circuit(params, va, vb)
        val += 1 - circuit(params, vb, va)

    return val / Q


@dataclass
class GHZStrategyTask:
    seed: int

    starting_value = 0
    final_value = 0
    params: jax.Array = None

    def run(self):
        G14 = nx.from_graph6_bytes(b'>>graph6<<M{dAH?`AoXAgCg~~_')
        key = random.PRNGKey(self.seed)
        params = random.normal(key, (14, 2, 4))

        self.starting_value = value(G14, params).item()

        def loss(params):
            return 1 - value(G14, params)

        solver = jaxopt.GradientDescent(loss)
        result = solver.run(params)

        self.params = result.params
        self.final_value = value(G14, result.params).item()

        # Fault-tolerance; save our results to a unique file in case
        # the cluster goes down before we have a chance to aggregate everything
        self.save_to_disk()
        return self.as_record()

    def save_to_disk(self):
        with open(f'{self.seed}.json', 'w', encoding='utf-8') as f:
            data = {
                "seed": self.seed,
                "starting_value": self.starting_value,
                "final_value": self.final_value,
                # For some reason jax doesn't have an equivalent
                "params": np.asarray(self.params).tolist()
            }
            json.dump(data, f)

    def as_record(self):
        result = OrderedDict()
        result["seed"] = self.seed
        result["starting_value"] = self.starting_value
        result["final_value"] = self.final_value
        return result

def main(args: argparse.Namespace):
    cpus = args.num_cpus
    n_trials = args.trials

    key = random.PRNGKey(0)
    seeds = random.randint(key, (n_trials,), 0, 2**32)
    tasks = [GHZStrategyTask(seed.item()) for seed in seeds]

    # Spawn method required for JAX. See https://github.com/google/jax/issues/743
    with mp.get_context("spawn").Pool(cpus) as pool:
        print("Starting trials...", flush=True)
        results = pool.imap_unordered(GHZStrategyTask.run, tasks)

        csv_file = 'results.csv'
        with open(csv_file, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['seed', 'starting_value', 'final_value'])
            writer.writeheader()
            f.flush()
            for result in results:
                print(f"Seed {result['seed']}\t{result['starting_value']:.6f} → {result['final_value']:.6f}", flush=True)
                writer.writerow(result)
                f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--trials', type=int)
    parser.add_argument('-p', '--num-cpus', type=int, default=1)

    args = parser.parse_args()
    main(args)
