import argparse
from pathlib import Path
import os

from nonlocalgames.circuit import NLGCircuit
from nonlocalgames.games import G14

import qiskit as qk
from qiskit import qasm2


def export_circuits(folder: str, type: str = "4q"):
    os.makedirs(folder, exist_ok=True)

    path = (
        Path(__file__).parent.parent.resolve()
        / "data"
        / "g14_constrained_u3ry"
        / "g14_state.json"
    )

    nlg = NLGCircuit.from_file(path)

    # Replace the state prep circuit with the smaller version
    if type == "bell_pair":
        areg = qk.QuantumRegister(2, "a")
        breg = qk.QuantumRegister(2, "b")
        qc = qk.QuantumCircuit(areg, breg)

        qc.h(breg[0])
        qc.h(breg[1])
        qc.cx(breg[0], areg[0])
        qc.cx(breg[1], areg[1])
        nlg = NLGCircuit(qc, None, measurement_layer=nlg.measurement_layer)

    for question in G14.get_questions():
        output_path = Path(folder) / f"g14_{type}_{question[0]}_{question[1]}.qasm"
        qc = nlg._prepare_question(question)
        qc = merge_registers(qc)
        qasm2.dump(qc, output_path)


def merge_registers(qc: qk.QuantumCircuit) -> qk.QuantumCircuit:
    qreg = qk.QuantumRegister(qc.num_qubits, "q")
    clreg = qk.ClassicalRegister(qc.num_clbits, "c")
    new_qc = qk.QuantumCircuit(qreg, clreg)

    qubit_offset = {}
    _offset = 0
    for _qreg in qc.qregs:
        qubit_offset[_qreg] = _offset
        _offset += len(_qreg)

    clbit_offset = {}
    _offset = 0
    for _clreg in qc.cregs:
        clbit_offset[_clreg] = _offset
        _offset += len(_clreg)

    for instruction in qc.data:
        operation = instruction.operation.to_mutable()
        qubits = [
            qreg[qubit_offset[qubit._register] + qubit._index]
            for qubit in instruction.qubits
        ]
        clbits = [
            clreg[clbit_offset[bit._register] + bit._index]
            for bit in instruction.clbits
        ]

        new_qc.append(operation, qubits, clbits)

    return new_qc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--folder", default="")
    parser.add_argument("-s", "--strategy", default="4q", choices=["4q", "bell_pair"])
    args = parser.parse_args()

    export_circuits(args.folder, args.strategy)
