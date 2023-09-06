'''Script to evaluate a g14 circuit on several ibm backends'''
import argparse
from multiprocessing.pool import ThreadPool
from dataclasses import dataclass
from typing import List, Sequence
import json

import numpy as np
import pandas as pd
import tqdm
from qiskit_ibm_provider import IBMProvider, IBMBackend
from qiskit.providers import JobV1 as Job
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from nonlocalgames.hamiltonians import G14
from nonlocalgames.circuit import NLGCircuit, load_adapt_ansatz, Counts
from nonlocalgames.measurement import MeasurementLayer

def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', help='IBMQ Account Name')
    parser.add_argument('--shots', default=1024, type=int)
    parser.add_argument('--circuit', help='Path to g14_state.json file')
    parser.add_argument('--backends', default=None, help='Comma-separated ibm computers')

    return parser.parse_args()

def load_g14_circuit(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qubits = 2
    players = 2
    qc = load_adapt_ansatz(
        data['state'],
        '++++',
        [qubits] * players,
        adapt_order=False,
        min_theta = 1e-4)

    # Don't need to transform phi since qinfo's Ry gate follows the qiskit convention
    phi = np.array(data['phi'])

    constrained = False
    if 'constrained' in path:
        constrained = True

    if 'metadata' in data:
        layer = data['metadata'].get('layer', 'ry')
        constrained = data['metadata'].get('constrain_phi', constrained)
    else:
        layer = 'ry'
    
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

def run(account: str, circuit: str, shots = 1024, backends: str = None):
    provider = IBMProvider(name=account)
    print('Loaded account', account)

    if backends is not None:
        # Use the user-provided list of backends
        backends = backends.split(',')
        backends: List[IBMBackend] = list(map(provider.get_backend, backends))
    else:
        # Get all real backends that have at least 4 qubits
        backends: List[IBMBackend] = provider.backends(
            simulator=False,
            operational=True,
            min_num_qubits=4
        )
    
    print(backends)
    
    shared_state, ml = load_g14_circuit(circuit)
    tasks = [IBMTask(backend, shared_state, ml, shots) for backend in backends]
    with ThreadPool(processes=len(tasks)) as p:
        results = list(tqdm.tqdm(p.imap(task, tasks), total=len(tasks)))

    print('Aggregating')
    df = pd.concat(results, axis=0, ignore_index=True)
    df.to_csv('ibm_results.csv', index=False)

@dataclass
class IBMTask:
    backend: IBMBackend
    shared_state: QuantumCircuit
    ml: MeasurementLayer
    shots: int = 1024

    def run(self):
        # Prepare nonlocal game circuit
        nlg = NLGCircuit(
            self.shared_state.copy(),
            None,
            sim=self.backend,
            measurement_layer=self.ml
        )
        
        # Prepare questions
        vertices = [(v, v) for v in range(14)]
        edges = G14._get_graph().edge_links.tolist()
        questions = vertices + edges

        # Use timeout of 1 second so this returns the job object
        job: Job = nlg.ask(questions, timeout=1, shots=self.shots)
        print(f'Submitted job {job.job_id()} to {self.backend.name}')

        # This blocks until the job is completed. This is made of time.sleep
        # calls, so other threads can do stuff while this one waits
        job.wait_for_final_state()

        # Get result and process the counts into a dataframe
        result = job.result()
        count_results = nlg._transform_results(result)

        df = self._make_df(questions, count_results)
        df['job'] = job.job_id()
        df['backend'] = self.backend.name
        df['shots'] = self.shots

        return df
    
    def _make_df(self, questions, count_results: Sequence[Counts]) -> pd.DataFrame:
        records = []
        for (va, vb), counts in zip(questions, count_results):
            for (ca, cb), n in counts.items():
                win = 0
                if ca == cb and va == vb:
                    win = 1
                elif ca != cb and va != vb:
                    win = 1

                records.append({
                    'va': va,
                    'vb': vb,
                    'ca': ca,
                    'cb': cb,
                    'win': win,
                    'n': n
                })

        return pd.DataFrame.from_records(records)

def task(task: IBMTask):
    task.run()

if __name__ == '__main__':
    args = get_cli_args()
    run(args.account, args.circuit, shots=args.shots, backends=args.backends)
