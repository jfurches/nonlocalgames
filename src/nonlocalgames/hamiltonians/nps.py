import itertools

import numpy as np
from scipy.sparse import csc_matrix

from adaptgym.pools import AllPauliPool

from .nlg_hamiltonian import NLGHamiltonian
from ..qinfo import *

class NPartiteSymmetricNLG(NLGHamiltonian):
    questions = 2
    qubits = 1

    def __init__(self, N: int, **kwargs):
        self.players = N

        super().__init__(**kwargs)

        self._pool = AllPauliPool(N)

    def _generate_hamiltonian(self) -> csc_matrix:
        N = self.players
        d = 2 ** N

        def M(i, q):
            Uqi = self._ml.to_unitary(i, q)
            A = Uqi.conj().T @ Z @ Uqi
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

        return H

    def _init_pool(self):
        self._pool.init(qubits=self.players * self.qubits)
        self._pool.get_operators()
        self._pool.generate_sparse_ops()

    @property
    def ref_ket(self):
        d = 2 ** self.players
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
