from dataclasses import dataclass, field
import json
from typing import List, Sequence, Tuple, Dict
from functools import cached_property, reduce
from string import ascii_lowercase
from abc import ABC, abstractmethod, abstractproperty

import numpy as np

# Qiskit has deprecated code inside of it
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.extensions import Initialize
    from qiskit.quantum_info import Pauli
    from qiskit.circuit import ParameterVector, Parameter
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.providers import Backend, JobV1
    from qiskit.providers.exceptions import JobTimeoutError
    from qiskit.result import Result
    from qiskit_aer import AerSimulator, AerJob
    from qiskit_aer.library import SaveStatevector

from openfermion import QubitOperator
from . import util
from .measurement import MeasurementLayer

Question = Sequence[int]
Counts = Dict[str, int]


@dataclass
class NLGCircuit:
    """Utility class to play non-local games from a quantum circuit

    First construct the circuit from a shared quantum state and set of measurements,

    >>> NLGCircuit(shared_state, phi)

    where `phi` is a numpy array of shape (players, questions, [qubits=1]). If qubits is not
    specified, it defaults to 1. It infers the number of players from `shared_state.qregs` and
    maps each qubit register to a classical register with the same number of bits. These registers
    can be accessed through `NLGCircuit.qregs` and `NLGCircuit.cregs`. `shared_state` should already
    have its parameters (theta) bound since those don't depend on the question.

    This class then adds a measurement layer consisting of Ry rotations per qubit and binds the
    appropriate parameters (phi) depending on the question when asked.

    Example (from G14):
        >>> shared_state, phi = load_adapt_ansatz(...)
        >>> nlg = NLGCircuit(shared_state, phi, sim=AerSimulator())
        >>> nlg.ask((1, 1), shots=1024)
        ... {(1, 1): 28,
            (0, 0): 43,
            (1, 0): 88,
            (1, 3): 15,
            (0, 2): 5,
            (0, 3): 42,
            (1, 2): 170,
            (0, 1): 633}
    """

    shared_state: QuantumCircuit

    # Array of shape (players, questions, [qubits_per_player = 1])
    phi: np.ndarray

    save_statevector: bool = field(default=False)
    sim: Backend = field(default=None, kw_only=True)
    measurement_layer_type: str = field(default="ry", kw_only=True)

    # Layer of rotation gates at the end that we'll construct
    measurement_params: ParameterVector = field(default=None, init=False)
    qc: QuantumCircuit = field(default=None, init=False)
    measurement_layer: MeasurementLayer = field(default=None, kw_only=True)
    conjugated: List[int] = field(default_factory=list, kw_only=True)

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        with open(filename, "r") as f:
            data = json.load(f)

        phi = np.array(data["phi"])
        players, questions, qubits, params_per_qubit = phi.shape
        conjugated = []
        if data["metadata"]["constrain_phi"]:
            players *= 2
            phi = np.concatenate([phi, phi], axis=0)
            conjugated = [1]

        # Todo: Maybe include reference state in metadata, but for
        # now just hardcode it to |+>
        qreg_sizes = tuple(qubits for _ in range(players))
        shared_state = load_adapt_ansatz(
            data["state"],
            "+" * (qubits * players),
            qreg_sizes,
            min_theta=1e-4,
        )

        return cls(
            shared_state,
            phi,
            measurement_layer_type=data["metadata"]["layer"],
            conjugated=conjugated,
            **kwargs,
        )

    def __post_init__(self):
        self.qc = self.shared_state.copy()

        if self.measurement_layer is None:
            self.measurement_layer = MeasurementLayer.get(
                self.measurement_layer_type, phi=self.phi
            )

            for i in self.conjugated:
                self.measurement_layer.conj(i)

        self.measurement_params = ParameterVector(
            "φ", length=self.measurement_layer.n_layer_params
        )

        # Add classical bit registers for each quantum register
        for qreg in self.qregs:
            creg = ClassicalRegister(qreg.size, "c" + qreg.name)
            self.qc.add_register(creg)

        # Construct measurement layer onto circuit
        # Iterate over players, each has their own qreg and creg
        for i, (qreg, creg) in enumerate(zip(self.qregs, self.cregs)):
            self.measurement_layer.add(i, self.qc, qreg, self.measurement_params)

        if self.save_statevector:
            full_qreg = reduce(lambda x, y: x + y[:], self.qregs, [])
            self.qc.append(SaveStatevector(self.qc.num_qubits), full_qreg)

        # Add measurement
        for qreg, creg in zip(self.qregs, self.cregs):
            self.qc.measure(qreg, creg)

    def ask(
        self,
        q: Question | Sequence[Question],
        return_statevector=False,
        timeout: float = None,
        **run_options,
    ):
        """Query the players with a vector of n questions.

        Args:
            q: Sequence of questions for each player
            run_options: Keyword arguments passed to `Backend.run`

        Returns:
            A dict of counts with the keys being a tuple of each player's response, e.g.
                `{(a1, a2, ..., an): count}`.
        """
        try:
            eval_qc = list(map(self._prepare_question, q))
        except TypeError:  # int object is not iterable
            eval_qc = self._prepare_question(q)

        job = self._submit_job(eval_qc, **run_options)
        try:
            if not isinstance(job, AerJob):
                job.wait_for_final_state(timeout=timeout)
            result: Result = job.result()
            return self._transform_results(
                result, return_statevector=return_statevector
            )

        except JobTimeoutError:
            return job

    def _prepare_question(self, q: Question) -> QuantumCircuit:
        phi = self.measurement_layer.map(q)
        eval_qc = self.transpiled.assign_parameters({self.measurement_params: phi})
        eval_qc.metadata["question"] = q
        return eval_qc

    def _transform_results(self, result: Result, return_statevector=False):
        return NLGCircuit.transform_results(result, return_statevector)

    @staticmethod
    def transform_results(result: Result, return_statevector=False):
        counts: Counts | Sequence[Counts] = result.get_counts()
        if isinstance(counts, dict):
            counts = util.from_ket_form(counts)

            if return_statevector:
                return counts, result.get_statevector()
        else:
            counts = list(map(util.from_ket_form, counts))

        return counts

    def _submit_job(
        self, qc: QuantumCircuit | Sequence[QuantumCircuit], **run_options
    ) -> JobV1:

        job = self.sim.run(qc, **run_options)
        return job

    @cached_property
    def transpiled(self):
        """Returns the quantum circuit transpiled for the simulator"""

        if self.sim is None:
            return self.qc

        return transpile(self.qc, backend=self.sim, optimization_level=0)

    @property
    def qregs(self):
        """The qubit registers for each player"""
        return self.qc.qregs

    @property
    def cregs(self):
        """The classical registers for each player"""
        return self.qc.cregs


def load_adapt_ansatz(
    state: Sequence[Tuple[float, str]],
    ref_ket,
    qreg_sizes: Sequence[int],
    adapt_order=False,
    min_theta=0,
):

    qregs: Sequence[QuantumRegister] = []
    qubits = 0
    for i, size in enumerate(qreg_sizes):
        qregs.append(QuantumRegister(size, name=ascii_lowercase[i]))
        qubits += size

    qc = QuantumCircuit(*qregs)
    parameters = {}
    full_qreg = reduce(lambda x, y: x + y[:], qregs, [])

    # Initialize the circuit with reference ket
    qc.append(
        Initialize(ref_ket, num_qubits=qubits if isinstance(ref_ket, int) else None),
        full_qreg,
    )

    # Make reference kets prettier if they're in standard basis, like |+>
    # goes from Initialize(+) to H|0>
    if isinstance(ref_ket, str):
        qc = qc.decompose(reps=2)

    # adapt_order means the parameters are stored in [tN, tN-1, ..., t1]
    iter_ = reversed(state) if adapt_order else state
    idx = 0
    for theta, qubit_op_str in iter_:
        if np.abs(theta) < min_theta:
            continue

        op = qubitop_from_str(qubit_op_str)

        # Construct pauli gate.
        qiskit_str = qubitop_to_str(op, qubits)
        pauli = Pauli(qiskit_str)
        param = Parameter(f"θ{idx}")

        # Use -θ since qiskit does e^{-iθP} while ADAPT expects e^{θA} for antihermitian A
        parameters[param] = -theta
        gate = PauliEvolutionGate(pauli, time=param)

        qc.append(gate, full_qreg)
        idx += 1

    # Set all the theta parameters to their values
    qc = qc.assign_parameters(parameters)

    # Delete all reset preparations
    for idx, instruction in reversed(list(enumerate(qc.data))):
        if instruction.operation.name == "reset":
            qc.data.pop(idx)

    return qc


def qubitop_from_str(s: str):
    # Convert term looking like '1j [X0 Y7]' into complex coeff 1.0j
    # and operator term into 'X0 Y7'.
    coeff, op_str = s.split(" ", maxsplit=1)
    coeff, op_str = complex(coeff), op_str[1:-1]
    op = QubitOperator(op_str, coeff)

    return op


def qubitop_to_str(op: QubitOperator, n_qubits: int, qiskit_format=True):
    """Converts an OpenFermion QubitOperator (assumed to be a Pauli string) to a string representation

    Args:
        op: The QubitOperator to convert
        n_qubits: The number of qubits this operator acts on
        qiskit_format: If True, returns in a format that Qiskit accepts. Otherwise,
            it uses the OpenFermion format. Default True.
    """

    assert len(op.terms) == 1
    term = list(op.terms.keys())[0]  # tuple of tuples (i, Q)

    if qiskit_format:
        ops = {idx: Q for (idx, Q) in term}

        # Construct string of operators, filling missing values with I,
        # e.g. XIIY
        term_str = "".join([ops.get(i, "I") for i in range(n_qubits)])
        term_str = term_str[::-1]
    else:
        # Join with spaces, e.g. X0 Y3. This format is used by openfermion
        term_str = " ".join(map(lambda x: f"{x[1]}{x[0]}", term))

    return term_str
