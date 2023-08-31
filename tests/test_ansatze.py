'''Test cases for qubit ordering in qiskit vs our code'''
import json
from pathlib import Path
from typing import Tuple, Dict, TypeVar

import pytest

# Qiskit has deprecated code inside of it
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    from qiskit import Aer
    from qiskit.quantum_info import Statevector

import openfermion as of
import numpy as np
from adaptgym import AdaptiveAnsatz

from nonlocalgames.circuit import (
    NLGCircuit,
    qubitop_from_str,
    load_adapt_ansatz
)
from nonlocalgames.hamiltonians import G14
from nonlocalgames.qinfo import Ry, tensor

@pytest.fixture(scope='session')
def sim():
    '''Fixture to create qiskit backend'''
    return Aer.get_backend('aer_simulator_statevector')

@pytest.fixture(scope='session')
def saved_state():
    '''Fixture that loads G14 ansatz data'''
    # tests/ directory
    tests_dir = Path(__file__).parent.resolve()
    data_path = tests_dir.parent.resolve() / \
        'data' / 'g14_imbalanced' / 'g14_state.json'
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data: dict = json.load(f)
    
    state = data['state']
    phi = np.array(data['phi'])
    return state, phi

@pytest.fixture(scope='session')
def adapt_state(saved_state):
    '''Fixture that constructs AdaptiveAnsatz from saved state data'''
    state, phi = saved_state
    ham = G14()
    ham.init()
    qubits = 2 * ham._qubits

    ansatz = AdaptiveAnsatz(ham.mat, ham.ref_ket)
    # Load in reversed order so we can do append() calls instead of
    # insert(0). The first operator added to the ansatz should be at
    # index -1 for the AdaptiveAnsatz code.
    for theta, gate_str in reversed(state):
        op = qubitop_from_str(gate_str)
        sp_op = of.get_sparse_operator(op, n_qubits=qubits)

        ansatz._curr_params.append(theta)
        ansatz.G.append(sp_op)
        ansatz.is_qubit_op.append(True)
    
    return ansatz, phi

@pytest.fixture()
def nlg(saved_state, sim) -> NLGCircuit:
    state, phi = saved_state
    qc = load_adapt_ansatz(
        state,
        '++++',
        qreg_sizes=(2,2),
        adapt_order=False)
    
    nlg = NLGCircuit(qc, phi, sim=sim, save_statevector=True)
    return nlg

class TestState:
    def test_states(self,
                    adapt_state: Tuple[AdaptiveAnsatz, np.ndarray],
                    nlg: NLGCircuit):
        # This test demonstrates how to get ADAPT and qiskit to output
        # the same statevector

        # Get ADAPT state. NLGCircuit provided through fixture
        adapt_ansatz, phi = adapt_state

        graph = G14._get_graph()
        questions = [(v, v) for v in range(13)] + graph.edge_links.tolist()

        for q in questions:
            adapt_ket = adapt_ansatz.prepare_state()
            U = tensor(
                [
                    Ry(phi[0, q[0], 0]), Ry(phi[0, q[0], 1]),
                    Ry(phi[1, q[1], 0]), Ry(phi[1, q[1], 1])
                ],
                indices=list(range(4)),
                N=4
            )

            adapt_ket = U @ adapt_ket
            adapt_ket = np.asarray(adapt_ket).ravel()
            adapt_statevector = Statevector(adapt_ket).reverse_qargs()
            # adapt_prob_vec = np.abs(adapt_ket) ** 2
            # adapt_probs = {}
            # for i, p in enumerate(adapt_prob_vec):
            #     bits = f'{i:04b}'
            #     bitstring = bits[2:4][::-1] + ' ' + bits[0:2][::-1]
            #     adapt_probs[bitstring] = p
            
            # adapt_probs = from_ket_form(adapt_probs)

            counts, qiskit_statevector = nlg.ask(q, return_statevector=True, shots=1024)
            # qiskit_statevector.probabilities_dict()
            # qiskit_probs = from_ket_form(qiskit_statevector.probabilities_dict())

            assert np.allclose(adapt_statevector, qiskit_statevector)

            # assert dict_allclose(adapt_probs, qiskit_probs)

            # We reverse the qubit order and the Pauli ordering in circuit preparation
            # to convert between qiskit and ADAPT
    
    def test_results(self, nlg):
        # Test that it gets vertex questions right. This checks that we constructed
        # the circuit properly.
        shots = 1024
        for q in range(14):
            counts = nlg.ask((q, q), shots=shots)
            s = 0
            for k, c in counts.items():
                if k[0] == k[1]:
                    s += c
            
            p_win = s / shots
            assert np.isclose(p_win, 1)

T = TypeVar('T')
def dict_allclose(d1: Dict[T, float], d2: Dict[T, float]) -> bool:
    assert d1.keys() == d2.keys()

    for k in d1.keys():
        v1, v2 = d1[k], d2[k]
        assert np.isclose(v1, v2)
