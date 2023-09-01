from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from typing import Tuple, Callable

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    from qiskit import (
        QuantumCircuit,
        QuantumRegister
    )
    from qiskit.circuit import ParameterVector

from .qinfo import *

@dataclass
class MeasurementLayer(ABC):
    '''Utility class for constructing measurement layers and mapping
    the phi tensors to circuit parameters'''
    players: int
    questions: int
    qubits: int
    phi: np.ndarray = None

    @property
    def n_layer_params(self):
        # phi has shape (players, question, qubits, params per qubit)
        assert self.phi.ndim == 4
        players, qubits, params_per_qubit = (
            self.phi.shape[0], self.phi.shape[-2], self.phi.shape[-1]
        )

        return players * qubits * params_per_qubit
    
    @abstractmethod
    def add(self, i: int, qc: QuantumCircuit, qreg: QuantumRegister, phi: ParameterVector):
        '''Adds the appropriate gates and parameters to the circuit
        
        Args:
            i: Index of the player's qreg
            qc: The quantum circuit
            qreg: The qubit register of player i
            phi: The full parameter vector object
        '''
        ...

    def map(self, q: tuple) -> np.ndarray:
        '''Returns the parameter vector based on question q = (q1, ..., qn) that
        maps to the ParameterVector phi'''

        phi = np.concatenate([self.phi[i, qi].ravel() for i, qi in enumerate(q)])
        return phi
    
    def uq(self, q: Tuple[int]) -> Tuple[np.ndarray]:
        ops = tuple(map(lambda a: self.to_unitary(*a), enumerate(q)))
        return tensor(ops, tuple(range(self.players)), self.players)
    
    @abstractmethod
    def to_unitary(self, i: int, qi: int) -> np.ndarray:
        ...

    @staticmethod
    def get(type_: str,
            players: int = None,
            questions = None,
            qubits = None,
            phi: np.ndarray = None) -> "MeasurementLayer":

        if phi is not None:
            players, questions, qubits = phi.shape[:3]

        if type_ in ('ry', 'u3'):
            return SingleQubitLayer(players, questions, qubits, phi=phi, type=type_)
        elif type_ == 'u10':
            return U10Layer(players, questions, qubits, phi=phi)
        elif type_ == 'cnotry':
            return CnotRyLayer(players, questions, qubits, phi=phi)
    
    @property
    def params(self):
        return self.phi
    
    @params.setter
    def params(self, v: np.ndarray):
        self.phi[:] = v.reshape(self.phi.shape)

@dataclass
class SingleQubitLayer(MeasurementLayer):
    type: str = 'ry'
    ufunc: Callable = field(default=None, init=False)

    def __post_init__(self):
        if self.type == 'ry':
            self.ufunc = Ry
            params_per_qubit = 1
        elif self.type == 'u3':
            self.ufunc = U3
            params_per_qubit = 3
        
        # Initialize the parameters if not given
        if self.phi is None:
            shape = (self.players, self.questions, self.qubits, params_per_qubit)
            self.phi = np.zeros(shape, dtype=float)

    def add(self, i, qc, qreg, phi):
        # Stride = params_per_qubit * qubits
        stride = self.phi.shape[-1] * self.phi.shape[-2]
        n = self.phi.shape[-1]
        # Iterate over qubits in each register
        for j, qubit in enumerate(qreg):
            idx = i*stride + j*n
            if self.type == 'ry':
                # Add y rotations that the players choose based on
                # the question they receive
                qc.ry(phi[idx], qubit)
            elif self.type == 'u3':
                qc.u(*phi[idx:idx+n], qubit)
    
    def to_unitary(self, i: int, qi: int):
        ops = [self.ufunc(*self.phi[i, qi, qubit])
               for qubit in range(self.qubits)]
        return tensor(ops, list(range(self.qubits)), self.qubits)
    
@dataclass
class CnotRyLayer(MeasurementLayer):
    def __post_init__(self):
        # Initialize the parameters if not given
        if self.phi is None:
            shape = (self.players, self.questions, self.qubits, 2)
            self.phi = np.zeros(shape, dtype=float)

    # Phi has standard shape (players, question, qubits, params_per_qubit)
    def add(self, i, qc, qreg, phi):
        assert len(qreg) == 2

        # Stride = qubits * params_per_qubit
        stride = self.qubits * 2
        idx = i * stride

        qc.ry(phi[idx + 0], qreg[0])
        qc.ry(phi[idx + 2], qreg[1])
        qc.cx(qreg[0], qreg[1])
        qc.ry(phi[idx + 1], qreg[0])
        qc.ry(phi[idx + 3], qreg[1])
    
    def to_unitary(self, i: int, qi: int):
        # This specifically has 2 qubits
        U1 = np.kron(
            Ry(self.phi[i, qi, 0, 0]),
            Ry(self.phi[i, qi, 1, 0]))
        U2 = np.kron(
            Ry(self.phi[i, qi, 0, 1]),
            Ry(self.phi[i, qi, 1, 1]))
        U = U2 @ cnot01 @ U1
        return U
    
@dataclass
class U10Layer(MeasurementLayer):
    def __post_init__(self):
        # Initialize the parameters if not given
        if self.phi is None:
            shape = (self.players, self.questions, self.qubits, 5)
            self.phi = np.zeros(shape, dtype=float)

    # Phi has shape (players, question, qubits=2, params_per_qubit=5)
    def add(self, i, qc, qreg, phi):
        assert len(qreg) == 2

        stride = self.phi.shape[-2] * self.phi.shape[-1]
        idx = i * stride

        qc.u(*phi[idx:idx+3], qreg[0])
        qc.u(*phi[idx+5:idx+8], qreg[1])

        qc.cx(qreg[0], qreg[1])

        qc.rx(phi[idx+3], qreg[0])
        qc.rz(phi[idx+4], qreg[0])

        qc.rz(phi[idx+8], qreg[1])
        qc.rx(phi[idx+9], qreg[1])
    
    def to_unitary(self, i: int, qi: int):
        # This also has 2 qubits
        U1 = np.kron(
            U3(*self.phi[i, qi, 0, 0:3]),
            U3(*self.phi[i, qi, 1, 0:3])
        )
        U2 = np.kron(
            Rx(self.phi[i, qi, 0, 3]) @ Rz(self.phi[i, qi, 0, 4]),
            Rz(self.phi[i, qi, 1, 3]) @ Rx(self.phi[i, qi, 1, 4]),
        )
        U = U2 @ cnot01 @ U1
        return U
