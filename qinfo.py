import functools
from typing import Union

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

def tensor_i(A: ArrayType, i: int, N: int) -> ArrayType:
    '''Returns the full operator ⊗_{j != i} Ij ⊗ Ai
    
    Assumes each operator acts on 1 qubit
    '''

    return functools.reduce(
        lambda x, y: np.kron(x,y),
        (A if j == i else I for j in range(N))
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
Ry = lambda t: np.cos(t/2) * I - 1j * np.sin(t/2) * Y
Rz = lambda t: np.cos(t/2) * I - 1j * np.sin(t/2) * Z
