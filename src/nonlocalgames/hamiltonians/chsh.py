import itertools

import numpy as np
from scipy.sparse import csc_matrix

from adaptgym.pools import PauliPool

from .nlg_hamiltonian import NLGHamiltonian
from ..qinfo import *

class CHSHHamiltonian(NLGHamiltonian):
    optimal_params: np.ndarray = np.array([0, -np.pi/2, -np.pi/4, np.pi/4])
    
    players = 2
    questions = 2
    qubits = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self._param_init_mode in ('optimal', 'normal', None)

        self._pool = PauliPool(lengths=(2,))

    def _init_pool(self):
        self._pool.init(qubits=2)
        self._pool.get_operators()
        self._pool.generate_sparse_ops()

    def _generate_hamiltonian(self):
        if self._param_init_mode == 'optimal':
            self._params[:] = self.optimal_params.reshape(self.desired_shape)

        ZZ = np.kron(Z, Z)
        sp_ham = csc_matrix((4, 4), dtype=complex)
        for qa, qb in itertools.product((0, 1), repeat=2):
            c = -1 if qa + qb == 2 else 1
            Uq = self._ml.uq((qa, qb))

            M = Uq.conj().T @ ZZ @ Uq
            # Use -= to invert the sign in order to make the smallest eigenvalue have the
            # largest inequality violation
            sp_ham -= c * M

        return sp_ham

    @property
    def ref_ket(self):
        # complex |00> state
        ket = csc_matrix(([[1 + 0j, 0, 0, 0]])).transpose()
        return ket
