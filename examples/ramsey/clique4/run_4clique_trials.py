import argparse
import glob
import itertools
import json
import logging
import os
import pickle as pkl
from importlib import resources
from typing import Dict, Sequence

import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from nonlocalgames import methods, util
from args import TaskArgs, TaskResult

gym.logger.set_level(logging.CRITICAL)

TMPDIR = "clique_tmpdata"
DATADIR = "clique_data"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                return obj.item()

            return obj.tolist()

        return super().default(obj)


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-cpus", type=int, default=1, help="Number of processes to use"
    )
    parser.add_argument(
        "--seeds",
        default="../../data/seeds.txt",
        help="File containing random seeds to read from",
    )
    parser.add_argument(
        "-n",
        "--trials",
        type=int,
        default=None,
        help="How many trials per graph to run with different seeds",
    )

    parser.add_argument(
        "--dpo-tol",
        type=float,
        default=1e-6,
        help="Maximum change in energy for convergence",
    )
    parser.add_argument(
        "--adapt-tol",
        type=float,
        default=1e-3,
        help="Maximum allowed gradient of ADAPT for convergence",
    )
    parser.add_argument(
        "--phi-tol",
        type=float,
        default=1e-5,
        help="Maximum gradient of measurement parameters for convergence",
    )
    parser.add_argument(
        "--theta-tol",
        type=float,
        default=1e-9,
        help="Maximum gradient for theta optimization during ADAPT",
    )

    parser.add_argument(
        "--layer", default="ry", help="Measurement layer type (see measurement.py)"
    )

    args = parser.parse_args()
    return args


def load_graphs() -> Dict[str, nx.Graph]:
    graphs = list(resources.files("nonlocalgames.data.clique4vertex8").iterdir())
    graph_name = lambda path: os.path.splitext(os.path.basename(path))[0]
    return {graph_name(path): nx.read_graph6(path) for path in graphs}


def create_trials(args: argparse.Namespace):
    global TMPDIR, DATADIR
    TMPDIR = f"{TMPDIR}_{args.layer}"
    DATADIR = f"{DATADIR}_{args.layer}"

    # Load the graphs and seeds
    graphs = load_graphs()
    seeds = util.load_seeds(args.seeds)
    trials = args.trials or len(seeds)

    # Construct temporary directory to hold intermediate
    # results in case stuff goes sideways
    os.makedirs(TMPDIR, exist_ok=True)

    assert trials <= len(
        seeds
    ), f"Not enough seeds saved to perform {trials} trials (max {len(seeds)})"

    # Save (seed, graph) tuples
    seeds = set(itertools.product(graphs.keys(), seeds[:trials]))

    # Now we find all trials in the intermediate directory and don't repeat them
    # to resume from where we started
    already_done = set()
    for file in glob.glob(os.path.join(TMPDIR, "*.pkl")):
        pathname = os.path.splitext(os.path.basename(file))[0]
        graph_name, seed = pathname.split("-")
        seed = int(seed)
        already_done.add((graph_name, seed))

    remaining = seeds.difference(already_done)

    task_args = list(
        map(
            lambda gs: TaskArgs(
                seed=gs[1],
                graph_name=gs[0],
                graph=graphs[gs[0]],
                dpo_tol=args.dpo_tol,
                adapt_tol=args.adapt_tol,
                phi_tol=args.phi_tol,
                theta_tol=args.theta_tol,
                layer=args.layer,
            ),
            remaining,
        )
    )

    return task_args


def main(args: argparse.Namespace):
    cpus = args.num_cpus
    task_args = create_trials(args)

    os.makedirs(DATADIR, exist_ok=True)

    print("Starting processing")
    results: Sequence[TaskResult] = Parallel(n_jobs=cpus, verbose=10)(
        delayed(task)(task_arg) for task_arg in task_args
    )

    print("Loading results from directory")
    other_results = load_results_from_dir()
    print(f"Loaded {len(other_results)} results")
    results: set = set(results)
    results.update(other_results)
    results = list(results)

    print("Postprocessing")
    dataframes = list(map(lambda r: r.df, results))

    print("Aggregating results")
    df = pd.concat(dataframes, axis=0, ignore_index=True)
    df.to_parquet(f"{DATADIR}/4clique_trials.parquet", index=False)

    # Cleanup intermediate results directory
    # shutil.rmtree(TMPDIR)


def load_results_from_dir():
    results = []
    for file in glob.glob(os.path.join(TMPDIR, "*.pkl")):
        with open(file, "rb") as f:
            results.append(pkl.load(f))

    return results


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
        phi_tol=args.phi_tol,
    )

    for iter_ in range(len(metrics["energy"])):
        record = {"seed": seed, "iter": iter_, "graph": args.graph_name}

        for k, series in metrics.items():
            if k not in ("theta_grad", "phi_grad"):
                record[k] = series[iter_]

        records.append(record)

    df = pd.DataFrame.from_records(records)
    result = TaskResult(df, state, phi, metrics, args)

    # Dump to temporary directory
    with open(os.path.join(TMPDIR, f"{args.graph_name}-{seed}.pkl"), "wb") as f:
        pkl.dump(result, f)

    return result


if __name__ == "__main__":
    args = get_cli_args()
    main(args)
