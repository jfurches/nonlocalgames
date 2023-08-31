'''Base class for hamiltonians representing non-local games'''

from abc import abstractmethod

import numpy as np
from scipy.sparse import csc_matrix
from adaptgym.hamiltonians import Hamiltonian
from adaptgym.pools import OperatorPool


class NLGHamiltonian(Hamiltonian):
    '''Base class for hamiltonians of non-local games, which
    have a few extra features in addition to regular ADAPT
    hamiltonians.
    
    In particular, changing measurement settings requires regenerating
    the hamiltonian, but in VQE problems the hamiltonian is constant
    throughout an optimization procedure.
    '''

    desired_shape = -1
    '''The shape the parameter vector should take internally'''

    def __init__(self, init_mode = None):
        '''Sets up the hamiltonian without computing the sparse matrix
        and operator pool.
        
        Args:
            init_mode: An optional `str` to determine how the parameters
                should be initialized. e.g. `normal` might mean to initialize
                measurement parameters as a normal distribution. This is
                determined by the subclass implementing `NLGHamiltonian`.
        '''
        self._params: np.ndarray = np.zeros(self.desired_shape)
        self._param_init_mode = init_mode

        self._sp_ham: csc_matrix = None
        self._pool: OperatorPool = None

        self._np_random = None
        '''Random number generator to be used'''

    @abstractmethod
    def _generate_hamiltonian(self) -> csc_matrix:
        '''Constructs the sparse matrix using the current measurement
        parameters.
        
        Called when `params` is updated or during `init()`.
        '''
        ...

    @abstractmethod
    def _init_pool(self):
        '''Initializes the operator pool'''
        ...

    def init(self, seed: int | None = None):
        '''Initializes the hamiltonian and pool, optionally with a seed for replication.
        
        Args:
            seed: Number used to seed numpy `default_rng`
        '''

        # Seed first then generate normal parameters if necessary
        self._np_random = np.random.default_rng(seed)

        if self._param_init_mode == 'normal':
            self._params = self._np_random.normal(loc=0, scale=np.pi/2, size=self.desired_shape)

        self._sp_ham = self._generate_hamiltonian()
        self._init_pool()

    @property
    def params(self) -> np.ndarray:
        '''Returns the parameters of this hamiltonian as a 1D vector to
        be compatible with scipy minimize.'''

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
        '''Returns the operator pool'''
        return self._pool

    @property
    def mat(self) -> csc_matrix:
        '''Returns the sparse matrix representation of this hamiltonian'''
        return self._sp_ham
    
    def get_n_ops(self):
        '''Returns the number of operators in the pool'''
        self._init_pool()
        return len(self._pool)
