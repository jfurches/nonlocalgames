import functools
from typing import Union, Sequence

import numpy as np
from scipy.sparse import csc_matrix

ArrayType = Union[np.ndarray, csc_matrix]

def is_hermitian(op: ArrayType) -> bool:
    if isinstance(op, np.ndarray):
        return np.all(op.T.conj() == op)
    else:
        return (op.conj().transpose() != op).nnz == 0

def is_antihermitian(op: ArrayType) -> bool:
    if isinstance(op, np.ndarray):
        return np.all(op.T.conj() == -op)
    else:
        return (op.conj().transpose() != -op).nnz == 0

def is_diagonal(op: ArrayType) -> bool:
    if isinstance(op, np.ndarray):
        return np.allclose(op, np.diag(op.diagonal()))
    else:
        return (op - op.todia()).nnz == 0

def is_unitary(op: ArrayType) -> bool:
    return np.allclose(op.T.conj() @ op, np.eye(op.shape[0]))

def tensor_i(A: ArrayType, i: int, N: int) -> ArrayType:
    '''Returns the full operator ⊗_{j != i} Ij ⊗ Ai
    
    Assumes each operator acts on 1 qubit
    '''

    return tensor([A], (i,), N)

def tensor(A: Sequence[ArrayType], indices: Sequence[int], N: int) -> ArrayType:
    '''Computes general tensor product for multiple operators
    
    Args:
        A: List of operators to tensor together
        indices: Index of operator A[i] in the system subspace. For any operator
            not listed, it is assumed to be an identity. We assume these indices are
            sorted in ascending order
        N: Size of the system

    Ex:
        `tensor((Ry, Ry), (1, 2), 3)` is equivalent to I ⊗ Ry ⊗ Ry since operator 0
        is not specified.
    '''

    ops = []
    for i in range(N):
        if i in indices:
            ops.append(A[indices.index(i)])
        else:
            ops.append(I)
    
    return functools.reduce(
        lambda x, y: np.kron(x, y),
        ops
    )

def commutator(A: ArrayType, B: ArrayType) -> ArrayType:
    return A@B - B@A

def anticommutator(A: ArrayType, B: ArrayType) -> ArrayType:
    return A@B + B@A

I = np.eye(2, dtype=complex)
X = np.array([
    [0, 1 + 0j],
    [1 + 0j, 0]
])
Y = np.array([
    [0, -1j],
    [1j, 0]
])
Z = np.array([
    [1.0 + 0j, 0],
    [0, -1.0]
])

Rx = lambda t: np.cos(t/2) * I - 1j * np.sin(t/2) * X
'''X-axis rotation consistent with qiskit'''
Ry = lambda t: np.cos(t/2) * I - 1j * np.sin(t/2) * Y
'''Y-axis rotation consistent with qiskit'''
Rz = lambda t: np.cos(t/2) * I - 1j * np.sin(t/2) * Z
'''Z-axis rotation consistent with qiskit'''

U3 = lambda t, p, l: np.array([
    [np.cos(t/2),               -np.exp(1j*l)*np.sin(t/2)],
    [np.exp(1j*p)*np.sin(t/2),  np.exp(1j*(p + l)) * np.cos(t/2)]
])
'''General single-qubit unitary gate'''

P0 = np.array([
    [1, 0],
    [0, 0]],
    dtype=complex
)

P1 = np.array([
    [0, 0],
    [0, 1]],
    dtype=complex
)

Cnot = lambda c, t, n: tensor((P0, I), (c, t), n) + tensor((P1, X), (c, t), n)

cnot01 = Cnot(0, 1, 2)
def U10(phi):
    # Phi should have 10 parameters

    # Start with 2 general U3 on each qubit
    U = np.kron(U3(*phi[0:3]), U3(*phi[3:6]))
    U = cnot01 @ U
    # Apply (Rz7Rx6 x Rx9Rz8)
    U = np.kron(
        Rz(phi[7]) @ Rx(phi[6]),
        Rx(phi[9]) @ Rz(phi[8])
    ) @ U

    return U
