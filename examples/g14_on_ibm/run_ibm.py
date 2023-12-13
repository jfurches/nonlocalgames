"""Script to evaluate a g14 circuit on several ibm backends"""
import argparse
from multiprocessing.pool import ThreadPool as Pool
from dataclasses import dataclass
from typing import List, Sequence
import json
from copy import deepcopy

import tqdm
import numpy as np
import pandas as pd
from qiskit_ibm_provider import IBMProvider, IBMBackend
from qiskit.providers import JobV1 as Job, JobStatus
from qiskit import QuantumCircuit

from nonlocalgames.hamiltonians import G14
from nonlocalgames.circuit import NLGCircuit, load_adapt_ansatz, Counts
from nonlocalgames.measurement import MeasurementLayer


def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", help="IBMQ Account Name")
    parser.add_argument("--shots", default=1024, type=int)
    parser.add_argument("--circuit", default=None, help="Path to g14_state.json file")
    parser.add_argument(
        "--backends", default=None, help="Comma-separated ibm computers"
    )
    parser.add_argument(
        "--jobs", default=None, help="Path to file containing old job ids to retrieve"
    )

    return parser.parse_args()


def load_g14_circuit(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qubits = 2
    players = 2
    qc = load_adapt_ansatz(
        data["state"], "++++", [qubits] * players, adapt_order=False, min_theta=1e-4
    )

    # Don't need to transform phi since qinfo's Ry gate follows the qiskit convention
    phi = np.array(data["phi"])

    constrained = False
    if "constrained" in path:
        constrained = True

    if "metadata" in data:
        layer = data["metadata"].get("layer", "ry")
        constrained = data["metadata"].get("constrain_phi", constrained)
    else:
        layer = "ry"

    ml = MeasurementLayer.get(layer, players=2, questions=14, qubits=2)

    if constrained:
        if phi.ndim == 1:
            phi = np.concatenate([phi, phi]).reshape(ml.shape)
        else:
            phi = np.concatenate([phi, phi])
            assert phi.shape == ml.shape

    ml.params = phi

    if constrained:
        ml.conj(1)

    return qc, ml


def run(account: str, circuit: str, shots=1024, backends: str = None, jobs: str = None):
    provider = IBMProvider(name=account)
    print("Loaded account", account)

    if jobs is not None:
        with open(jobs, "r", encoding="utf-8") as f:
            job_ids = f.readlines()

        job_ids = [l.strip() for l in job_ids]
        job_ids = list(filter(lambda x: len(x) > 0 and not x.startswith("#"), job_ids))
        tasks = [
            IBMTask(None, None, None, account, shots=shots, job_id=id) for id in job_ids
        ]
    else:
        if backends is not None:
            # Use the user-provided list of backends
            backends = backends.split(",")
            backends: List[IBMBackend] = list(map(provider.get_backend, backends))
        else:
            # Get all real backends that have at least 4 qubits
            backends: List[IBMBackend] = provider.backends(
                simulator=False, operational=True, min_num_qubits=4
            )

        shared_state, ml = load_g14_circuit(circuit)
        tasks = [
            IBMTask(backend, deepcopy(shared_state), deepcopy(ml), account, shots=shots)
            for backend in backends
        ]

    with Pool(processes=len(tasks)) as p:
        results = tqdm.tqdm(p.imap(IBMTask.run, tasks), total=len(tasks))
        results = list(filter(lambda x: x is not None, results))
        print("Retrieved results", flush=True)

    print("Aggregating", flush=True)
    df = pd.concat(results, axis=0, ignore_index=True)
    df.to_csv("ibm_results.csv", index=False)


@dataclass
class IBMTask:
    backend: IBMBackend
    shared_state: QuantumCircuit
    ml: MeasurementLayer
    account: str
    shots: int = 1024
    job_id: str = None

    def run(self):
        provider = IBMProvider(name=self.account)

        # Prepare questions
        questions = G14.get_questions()

        # If no provided job id, prepare circuit and send to IBM
        if self.job_id is None:
            # Prepare nonlocal game circuit
            nlg = NLGCircuit(
                self.shared_state.copy(),
                None,
                sim=self.backend,
                measurement_layer=self.ml,
            )

            # Use timeout of 1 second so this returns the job object
            job: Job = nlg.ask(questions, timeout=1, shots=self.shots)
            self.job_id = job.job_id()
            print(f"Submitted job {self.job_id} to {self.backend.name}", flush=True)

        # Retrieve job from the cloud given its id
        else:
            job = provider.retrieve_job(self.job_id)
            self.backend = job.backend()
            print(
                f"Retrieved job {self.job_id} for backend {self.backend.name}",
                flush=True,
            )

        try:
            # Get result and process the counts into a dataframe
            result = job.result()
            print(f"Processing results for {self.backend.name}", flush=True)
            count_results = NLGCircuit.transform_results(result)
            assert len(count_results) == len(questions)

            print(f"Making df for {self.backend.name}", flush=True)
            df = self._make_df(questions, count_results)
            df["job"] = self.job_id
            df["backend"] = self.backend.name
            df["shots"] = self.shots

            print(f"Finished {self.backend.name}", flush=True)
            return df

        except Exception as e:
            # If it errors, for example with an errored job,
            # return an empty dataframe that does not affect the
            # concat() call
            print(f"{self.backend.name} had error:", e)

        print(f"Returning None for {self.backend.name}")
        return None

    def _make_df(
        self, questions: Sequence[tuple], count_results: Sequence[Counts]
    ) -> pd.DataFrame:
        assert len(questions) == len(count_results)

        records = []
        for (va, vb), counts in zip(questions, count_results):
            for (ca, cb), n in counts.items():
                win = 0
                if ca == cb and va == vb:
                    win = 1
                elif ca != cb and va != vb:
                    win = 1

                records.append(
                    {"va": va, "vb": vb, "ca": ca, "cb": cb, "win": win, "n": n}
                )

        return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    args = get_cli_args()
    run(
        args.account,
        args.circuit,
        shots=args.shots,
        backends=args.backends,
        jobs=args.jobs,
    )
