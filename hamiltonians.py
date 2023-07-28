import itertools
import functools

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from adaptgym.hamiltonians import Hamiltonian
from adaptgym.pools import PauliPool, AllPauliPool

from qinfo import *

class CHSHHamiltonian(Hamiltonian):
    optimal_params: np.ndarray = np.array([0, -np.pi/4, -np.pi/8, np.pi/8])

    def __init__(self, *, initialize_mode: str = 'optimal', **kwargs):
        super().__init__(**kwargs)

        self.mode = initialize_mode
        assert self.mode in ('optimal', 'normal')

        self.__pool = PauliPool(lengths=(2,))
        self.sp_ham = None
        self._params = None
    
    def init(self, seed: int | None = None):
        np_random = np.random.default_rng(seed)

        # Initialize pauli pool with 2 qubits
        self.__pool.init(qubits=2)
        self.__pool.get_operators()
        self.__pool.generate_sparse_ops()

        self._params = self.optimal_params if self.mode == 'optimal' else \
                        np_random.normal(loc=0, scale=np.pi/4, size=4)  # 95% of values are within [-pi/2, pi/2]
        
        # Construct the hamiltonian
        self.sp_ham = self._generate_hamiltonian()
    
    @property
    def ref_ket(self):
        # complex |00> state
        ket = csc_matrix(([[1 + 0j, 0, 0, 0]])).transpose()
        return ket

    def get_n_ops(self):
        # Initialize pauli pool with 2 qubits
        self.__pool.init(qubits=2)
        self.__pool.get_operators()
        self.__pool.generate_sparse_ops()

        return len(self.__pool)

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params):
        old_params = self._params
        self._params = params

        # Regenerate the hamiltonian if we changed the parameters
        if not all(old_params == params):
            self.sp_ham = self._generate_hamiltonian()

    def _generate_hamiltonian(self):
        Y = np.array([
            [0, -1j],
            [1j, 0]
        ])
        Z = np.array([
            [1 + 0j, 0],
            [0, -1 + 0j]
        ])
        I = np.eye(2, dtype=complex)

        sp_ham = csc_matrix((4, 4), dtype=complex)
        for qa, qb in itertools.product((0, 1), repeat=2):
            c = -1 if qa + qb == 2 else 1
            phia, phib = self._params[qa], self._params[2 + qb]

            # Ry gates for each measurement, decompose using the exponential formula
            # exp(-itY) = cos(t)I - isin(t)Y
            Ma = np.cos(phia) * I - 1j * np.sin(phia) * Y
            Mb = np.cos(phib) * I - 1j * np.sin(phib) * Y

            M = np.kron(Ma.conj().T @ Z @ Ma, Mb.conj().T @ Z @ Mb)
            # Use -= to invert the sign in order to make the smallest eigenvalue have the
            # largest inequality violation
            sp_ham -= c * M

        return sp_ham
    
    @property
    def pool(self):
        return self.__pool
    
    @property
    def mat(self):
        return self.sp_ham


class NPartiteSymmetricNLG(Hamiltonian):
    def __init__(self, N: int):
        super().__init__()

        self.N = N
        self._params = np.zeros((N, 2))
        self._sp_ham: csc_matrix = None

        self._pool = AllPauliPool(N)

    def _generate_hamiltonian(self, N: int) -> csc_matrix:
        d = 2 ** N
        phi = self._params

        def M(i, q):
            A = Ry(-phi[i, q]) @ Z @ Ry(phi[i, q])
            return tensor_i(A, i, N)

        # Initialize each of these terms to 0
        s0 = csc_matrix((d,d), dtype=complex)
        s01 = csc_matrix((d,d), dtype=complex)
        # These two terms will technically be 1/2 their value in the paper
        s00 = csc_matrix((d,d), dtype=complex)
        s11 = csc_matrix((d,d), dtype=complex)

        # one-body terms
        for i in range(N):
            s0 += M(i, 0)
        
        # two-body terms. using combinations ensures i != j
        for i, j in itertools.combinations(range(N), 2):
            # These are guaranteed to be hermitian since i != j
            s00 += M(i, 0) @ M(j, 0)
            s11 += M(i, 1) @ M(j, 1)
            s01 += M(i, 0) @ M(j, 1) + M(j, 0) @ M(i, 1)

        # We skip the 1/2 factors on s00, s11 because we summed
        # skipping repeated operators.
        # Sum of all hermitian operators is also hermitian
        H = -2 * s0 + s00 + s11 - s01 + 2 * N * np.eye(d)

        # Disable this check here since it's expensive, and instead use unit tests
        # if is_diagonal(H):
        #     raise RuntimeWarning('Diagonal hamiltonian')

        return H

    def init(self, seed: int | None = None):
        super().init(seed)

        self._pool.init(qubits=self.N)
        self._pool.get_operators()
        self._pool.generate_sparse_ops()

        self._sp_ham = self._generate_hamiltonian(self.N)

    def get_n_ops(self):
        # print('Initializing pool')
        self._pool.init(qubits=self.N)
        ops = self._pool.get_operators()
        # print(f'Generated {len(ops)} operators')
        self._pool.generate_sparse_ops()

        return len(ops)

    @property
    def mat(self):
        return self._sp_ham

    @property
    def params(self):
        # Return the parameters as a vector for scipy
        return self._params.reshape(2 * self.N)

    @params.setter
    def params(self, phi: np.ndarray):
        phi = phi.reshape(self.N, 2)
        changed = np.any(phi != self._params)
        self._params = phi

        # If the parameters changed, recompute the hamiltonian
        if changed:
            self._sp_ham = self._generate_hamiltonian(self.N)
    
    @property
    def ref_ket(self):
        d = 2 ** self.N
        # Equal superposition state
        ket = csc_matrix(np.full((d, 1), 1 / np.sqrt(d)), dtype=complex)
        # ket[0, 0] = 1 # |00...> state

        return ket
    
    @property
    def pool(self):
        return self._pool
