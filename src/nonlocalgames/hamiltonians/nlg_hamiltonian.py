'''Base class for hamiltonians representing non-local games'''

from abc import abstractmethod

import numpy as np
from scipy.sparse import csc_matrix
from adaptgym.hamiltonians import Hamiltonian
from adaptgym.pools import OperatorPool


class NLGHamiltonian(Hamiltonian):
    desired_shape = -1

    def __init__(self, init_mode = None):
        self._params: np.ndarray = np.zeros(self.desired_shape)
        self._param_init_mode = init_mode

        self._sp_ham: csc_matrix = None
        self._pool: OperatorPool = None

        self._np_random = None

    @abstractmethod
    def _generate_hamiltonian(self, seed: int | None = None) -> csc_matrix:
        ...

    @abstractmethod
    def _init_pool(self):
        ...

    def init(self, seed: int | None = None):
        # Seed first then generate normal parameters if necessary
        self._np_random = np.random.default_rng(seed)

        if self._param_init_mode == 'normal':
            self._params = self._np_random.normal(loc=0, scale=np.pi/2, size=self.desired_shape)

        self._sp_ham = self._generate_hamiltonian()
        self._init_pool()

    @property
    def params(self) -> np.ndarray:
        return self._params.ravel()

    @params.setter
    def params(self, params: np.ndarray):
        '''Setter for hamiltonian measurement parameters that regenerates the
        matrix if the parameters change'''

        old_params = self._params
        self._params = params.reshape(self.desired_shape)

        # Regenerate the hamiltonian if we changed the parameters
        if not np.allclose(old_params, self._params, rtol=0, atol=1e-10):
            self._sp_ham = self._generate_hamiltonian()

    @property
    def pool(self) -> OperatorPool:
        return self._pool

    @property
    def mat(self) -> csc_matrix:
        return self._sp_ham
    
    def get_n_ops(self):
        self._init_pool()
        return len(self._pool)
