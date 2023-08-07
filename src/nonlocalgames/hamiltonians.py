'''Contains hamiltonians representing non-local games'''

import itertools
from abc import abstractmethod

import numpy as np
from scipy.sparse import csc_matrix
from adaptgym.hamiltonians import Hamiltonian
from adaptgym.pools import OperatorPool, PauliPool, AllPauliPool

from .qinfo import *


class NLGHamiltonian(Hamiltonian):
    def __init__(self):
        self._params: np.ndarray = None
        self._sp_ham: csc_matrix = None
        self._pool: OperatorPool = None

    @abstractmethod
    def _generate_hamiltonian(self) -> csc_matrix:
        ...

    @property
    def params(self) -> np.ndarray:
        return self._params

    @params.setter
    def params(self, params: np.ndarray):
        '''Setter for hamiltonian measurement parameters that regenerates the
        matrix if the parameters change'''

        old_params = self._params
        self._params = params

        # Regenerate the hamiltonian if we changed the parameters
        if not np.allclose(old_params, params):
            self._sp_ham = self._generate_hamiltonian()

    @property
    def pool(self) -> OperatorPool:
        return self._pool

    @property
    def mat(self) -> csc_matrix:
        return self._sp_ham

class CHSHHamiltonian(NLGHamiltonian):
    optimal_params: np.ndarray = np.array([0, -np.pi/2, -np.pi/4, np.pi/4])

    def __init__(self, *, initialize_mode: str = 'optimal', **kwargs):
        super().__init__(**kwargs)

        self.mode = initialize_mode
        assert self.mode in ('optimal', 'normal')

        self._pool = PauliPool(lengths=(2,))
        self._sp_ham = None
        self._params = None

    def init(self, seed: int | None = None):
        np_random = np.random.default_rng(seed)

        # Initialize pauli pool with 2 qubits
        self._pool.init(qubits=2)
        self._pool.get_operators()
        self._pool.generate_sparse_ops()

        # 95% of values are within [-pi/2, pi/2]
        self._params = self.optimal_params if self.mode == 'optimal' else \
                        np_random.normal(loc=0, scale=np.pi/4, size=4)

        # Construct the hamiltonian
        self._sp_ham = self._generate_hamiltonian()

    @property
    def ref_ket(self):
        # complex |00> state
        ket = csc_matrix(([[1 + 0j, 0, 0, 0]])).transpose()
        return ket

    def get_n_ops(self):
        # Initialize pauli pool with 2 qubits
        self._pool.init(qubits=2)
        self._pool.get_operators()
        self._pool.generate_sparse_ops()

        return len(self._pool)

    def _generate_hamiltonian(self):
        sp_ham = csc_matrix((4, 4), dtype=complex)
        for qa, qb in itertools.product((0, 1), repeat=2):
            c = -1 if qa + qb == 2 else 1
            phia, phib = self._params[qa], self._params[2 + qb]

            # Ry gates for each measurement, decompose using the exponential formula
            # exp(-itY) = cos(t)I - isin(t)Y
            Ma = Ry(phia)
            Mb = Ry(phib)

            M = np.kron(Ma.conj().T @ Z @ Ma, Mb.conj().T @ Z @ Mb)
            # Use -= to invert the sign in order to make the smallest eigenvalue have the
            # largest inequality violation
            sp_ham -= c * M

        return sp_ham


class NPartiteSymmetricNLG(NLGHamiltonian):
    def __init__(self, N: int):
        super().__init__()

        self.N = N
        self._params = np.zeros((N, 2))
        self._sp_ham: csc_matrix = None

        self._pool = AllPauliPool(N)

    def _generate_hamiltonian(self) -> csc_matrix:
        N = self.N
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

        self._sp_ham = self._generate_hamiltonian()

    def get_n_ops(self):
        self._pool.init(qubits=self.N)
        ops = self._pool.get_operators()
        self._pool.generate_sparse_ops()

        return len(ops)

    @property
    def params(self):
        # Return the parameters as a vector for scipy
        return self._params.ravel()

    @params.setter
    def params(self, params: np.ndarray):
        old_params = self._params
        self._params = params.reshape(self.N, 2)

        # Regenerate the hamiltonian if we changed the parameters
        if not np.allclose(old_params, self._params, rtol=0, atol=1e-10):
            self._sp_ham = self._generate_hamiltonian()

    @property
    def ref_ket(self):
        d = 2 ** self.N
        # Equal superposition state
        ket = csc_matrix(np.full((d, 1), 1 / np.sqrt(d)), dtype=complex)
        # ket[0, 0] = 1 # |00...> state

        return ket
    
    def gradient(self, state: np.ndarray, phi: np.ndarray | None = None) -> np.ndarray:
        phi = self._params if phi is None else phi
        output_shape = phi.shape
        phi = phi.reshape(self.N, 2)

        def M(i, q):
            # Difference between this one and the the version
            # above is this returns a measurement operator Miq that's
            # only in the subspace of i and not the full tensor
            # product
            A = Ry(-phi[i, q]) @ Z @ Ry(phi[i, q])
            return A
        
        # Pre-compute all the measurement operators, which we can index with
        # M_[i][q]
        M_ = [
            [M(j, q) for q in (0, 1)]
            for j in range(self.N)
        ]

        # Compute the full 2-body operator Delta0
        A = [M_[j][0] - M_[j][1] for j in range(self.N)]
        Delta0 = tensor(A, tuple(range(self.N)), self.N)

        grad = np.zeros_like(phi)
        for i in range(self.N):
            # Two-body term can be computed here since it does not
            # depend on q except for the (-1)^q factor
            term = M_[i][0] - M_[i][1]
            Deltai0 = tensor_i(term, i, self.N)
            two_body_op = (Delta0 - Deltai0)

            for q in (0, 1):
                # Compute partial operator, we use -i since our convention for
                # Ry in qinfo.py is exp(-itY/2)
                term = -1j/2 * (M_[i][q] @ Y)
                partial_iq = tensor_i(term, i, self.N)

                # Expectation of two body operator
                term = (state.T.conj() @ partial_iq) @ (two_body_op @ state)
                assert term.size == 1
                grad[i,q] += (-1) ** q * (term.item().real)

                # Add in the single-body term
                if q == 0:
                    term = state.T.conj() @ partial_iq @ state
                    assert term.size == 1
                    grad[i,q] += -4 * term.item().real
            
        return grad.reshape(output_shape)
