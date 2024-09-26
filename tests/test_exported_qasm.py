from pathlib import Path

import re
import numpy as np
import pytest
import qiskit as qk
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler, SamplerOptions


class TestQiskit:
    def test_qiskit_1_0(self):
        """Ensures we're using the modern version of qiskit"""
        version = tuple(map(int, qk.__version__.split(".")))
        assert version >= (1, 0, 0)


class TestCircuits:
    @pytest.mark.parametrize("strategy", ["4q", "bell_pair"])
    def test_exported_circuits(self, strategy: str):
        """Tests that the exported circuits reproduce perfect win rates"""

        backend = AerSimulator(method="statevector")
        pattern = re.compile(r".*_(\d+)_(\d+).qasm")
        sampler = Sampler(backend, options=SamplerOptions(default_shots=4096))

        folder = Path(__file__).parent.parent / "circuits" / strategy
        for file in folder.glob("*.qasm"):
            match = pattern.match(file.name)
            if not match:
                continue

            va = int(match.group(1))
            vb = int(match.group(2))
            is_vertex = va == vb

            qc = qk.QuantumCircuit.from_qasm_file(file)
            qc = qk.transpile(qc, backend, optimization_level=0)

            job = sampler.run([qc])
            result = job.result()[0].data

            counts = result.c.get_counts()
            shots = sum(counts.values())
            prob_ca_eq_cb = (
                sum(counts.get(f"{bs:04b}", 0) for bs in (0, 5, 10, 15)) / shots
            )

            win_rate = prob_ca_eq_cb if is_vertex else 1 - prob_ca_eq_cb
            assert np.isclose(win_rate, 1)


if __name__ == "__main__":
    pytest.main([__file__])
