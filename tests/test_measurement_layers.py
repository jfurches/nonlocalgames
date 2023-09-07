import itertools
from pathlib import Path
import json

import pytest
import numpy as np

from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit_aer import Aer

from nonlocalgames.measurement import MeasurementLayer
from nonlocalgames.qinfo import is_unitary
from nonlocalgames.hamiltonians import G14

class TestMeasurement:
    @pytest.mark.parametrize('layer', ('ry', 'cnotry', 'u3', 'u10', 'u3ry'))
    def test_layers(self, layer):
        ml = MeasurementLayer.get(layer, 2, 14, 2)
        ml.phi[:] = np.random.normal(size=ml.phi.shape)

        for va, vb in itertools.product(range(14), repeat=2):
            Uq = ml.uq((va, vb))
            assert is_unitary(Uq)

    def test_u10_g14(self):
        path = (
            Path(__file__).parent.parent.resolve() 
            / 'data' / 'g14_constrained_u10' / 'g14_state.json'
        )

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        phi = np.array(data['phi'])
        assert phi.shape == (1, 14, 2, 5)
        phi = np.concatenate([phi, phi], axis=0)
        ml = MeasurementLayer.get('u10', phi=phi)
        ml.conj(1)

        # Make dummy circuit
        qc = QuantumCircuit(QuantumRegister(2), QuantumRegister(2), name='q')
        params = ParameterVector('p', 20)
        for i, qreg in enumerate(qc.qregs):
            ml.add(i, qc, qreg, params)
        
        # Reverse bits to be compatible with our ordering
        # qc = qc.reverse_bits()
        backend = Aer.get_backend('unitary_simulator')
        qc = transpile(qc, backend)

        vertices = [(v, v) for v in range(14)]
        edges = G14._get_graph().edge_links.tolist()
        questions = vertices + edges

        for q in questions:
            phi = ml.map(q)
            qc_test = qc.bind_parameters({params: phi})
            job = backend.run(qc_test, shots=8192)
            result = job.result()
            qiskit_unitary = result.get_unitary()

            ua = ml.to_unitary(0, q[0])
            ub_conj = ml.to_unitary(0, q[1]).conj()
            ub = ml.to_unitary(1, q[1])
            assert np.allclose(ub, ub_conj)
            our_unitary = np.kron(ua, ub)

            assert np.allclose(qiskit_unitary, our_unitary, rtol=1e-3)
