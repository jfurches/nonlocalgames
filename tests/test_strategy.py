from pathlib import Path

import numpy as np
import pytest
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector, SaveUnitary

from nonlocalgames.circuit import NLGCircuit
from nonlocalgames.games import G14


@pytest.fixture(scope="module")
def nlg():
    path = (
        Path(__file__).parent.parent.resolve()
        / "data"
        / "g14_constrained_u3ry"
        / "g14_state.json"
    )

    nlg = NLGCircuit.from_file(path, sim=AerSimulator())
    return nlg


class TestG14:
    def test_graph(self):
        """Tests that the graph itself matches the definition (arXiv 1801.03542)

        This test can check for out-of-distribution questions, such as 2 vertices
        that don't form an edge.
        """
        G = G14.get_graph()
        questions = G14.get_questions()

        assert len(G) == 14

        # Check the questions function gives us all the questions
        assert len(questions) == 88

        # Check the apex vertex is correct
        for i in range(13):
            assert G.has_edge(i, 13)

        # Check some edges (not all)
        assert G.has_edge(0, 1)
        assert G.has_edge(1, 5)
        assert G.has_edge(6, 10)

    def test_full_strategy(self, nlg: NLGCircuit):
        """This evaluates the game on each question, ensuring a perfect win rate.

        We check the probability p(ca = cb). Vertex questions get compared to 1,
        edge questions get compared to 0.
        """
        for q in G14.get_questions():
            is_edge = q[0] != q[1]
            counts = nlg.ask(q, shots=8192)

            shots = sum(counts.values())
            win_rate = sum(counts.get((c, c), 0) for c in range(4)) / shots

            if is_edge:
                assert np.isclose(win_rate, 0)
            else:
                assert np.isclose(win_rate, 1)

    def test_g14_state_prep(self, nlg: NLGCircuit):
        """This ensures we obtain the maximally entangled state
        from the G14 state prep circuit.

        We'll compare it to the bell state prep version, and checking
        that they're equal up to some tolerance.
        """

        test_qc = nlg.shared_state
        test_qc.append(SaveStatevector(4), [0, 1, 2, 3])

        # Bell pair circuit that should produce 1/2 H^4(|00> + |11> + |22> + |33>)
        qc = QuantumCircuit(QuantumRegister(4, "q"))
        qc.h(2)
        qc.h(3)
        qc.cx(2, 0)
        qc.cx(3, 1)
        qc.append(SaveStatevector(4), [0, 1, 2, 3])

        backend = AerSimulator(method="statevector")

        test_qc = qk.transpile(test_qc, backend, optimization_level=0)
        test_job = backend.run(test_qc, shots=8192)
        test_result = test_job.result().get_statevector(test_qc)

        qc = qk.transpile(qc, backend, optimization_level=0)
        job = backend.run(qc, shots=8192)
        result = job.result().get_statevector(qc)

        # Check up to a 1e-5 error. Remember that we truncated some gates
        # at theta = 1e-4, so some error is natural.
        assert np.allclose(np.abs(test_result - result), 0, atol=1e-5)

    def test_g14_measurement_layer(self, nlg: NLGCircuit):
        """This test should validate the G14 strategy measurements.

        In particular, we test that we get the same unitary matrix as Qiskit
        for each game question.
        """
        ml = nlg.measurement_layer

        # Make parametrized circuit without state prep. This is the measurement
        # layer applied to |0>
        qc = QuantumCircuit(QuantumRegister(2), QuantumRegister(2), name="q")
        params = ParameterVector("p", 16)
        for i, qreg in enumerate(qc.qregs):
            ml.add(i, qc, qreg, params)

        # Reverse bits to be compatible with our ordering.
        # Our qc procedure uses big endian while qiskit uses
        # little endian
        qc = qc.reverse_bits()

        # Transpile the circuit for the unitary simulator, which will
        # allow us to check that the unitary is correct
        backend = AerSimulator(method="unitary")
        qc.append(SaveUnitary(4), [0, 1, 2, 3])
        qc = qk.transpile(qc, backend)

        for q in G14.get_questions():
            # Get the qiskit unitary
            phi = ml.map(q)
            qc_test = qc.assign_parameters({params: phi})
            job = backend.run(qc_test, shots=8192)
            result = job.result()
            qiskit_unitary = result.get_unitary()

            # Get our unitary, after checking that the conjugation
            # procedure works as intended
            ua = ml.to_unitary(0, q[0])
            ub_conj = ml.to_unitary(0, q[1]).conj()  # manually conjugate alice with vb
            ub = ml.to_unitary(1, q[1])
            # Check the measurement layer producing the unitary for bob is equivalent
            # to getting Alice's unitary for v_b, then conjugating it
            assert np.allclose(ub, ub_conj)
            our_unitary = np.kron(ua, ub)  # construct full unitary with tensor product

            # Check that our version matches qiskit
            assert np.allclose(qiskit_unitary, our_unitary, rtol=1e-4)


if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_strategy.py"])
