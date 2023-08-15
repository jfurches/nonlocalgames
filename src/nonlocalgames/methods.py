'''Contains algorithms for non-local games'''

from typing import List, Tuple
import functools

from scipy.optimize import minimize, OptimizeResult
import numpy as np
from openfermion import SymbolicOperator

from adaptgym import AdaptGame
from adaptgym.hamiltonians import Hamiltonian

def dual_phase_optim(ham: Hamiltonian, verbose = 0, tol = 1e-5, seed = None):
    '''Performs dual-phase optimization on a hamiltonian whose ground state represents
    the optimal settings for a non local game'''

    # Starting hamiltonian, random measurement parameters
    np_random = np.random.default_rng(seed=seed)
    phi_random = np_random.normal(scale=np.pi/2, size=ham.params.shape)
    phi = None
    shared_state = None

    ineq_value = -np.inf
    new_ineq_value = 0
    iter_ = 1

    ineq_values = []

    # Calculate initial energy
    ham.init(seed=seed)
    ham.params = phi_random
    bra = ham.ref_ket.conj().T
    E = (bra @ ham.mat @ ham.ref_ket).item().real
    ineq_values.append(E)

    while np.abs(new_ineq_value - ineq_value) > tol:
        ineq_value = new_ineq_value

        if verbose:
            print(f'Iter {iter_}\n-----------')
        # Phase 1: Create optimal shared state for measurement params
        # using ADAPT

        # Make sure to move hamiltonian parameters before reset() call, otherwise
        # ADAPT will return wrong pool gradients
        phi = phi_random if phi is None else phi
        ham.params = phi

        env = AdaptGame(ham, criteria='max')
        done = False
        _, info = env.reset()

        shared_state = env.ansatz
        shared_state.H = ham.mat
        if verbose >= 2:
            print('Starting phi:', phi)
            print('Variance:', info['var'])
            print('Generating state with ADAPT')

        while not done:
            if verbose >= 3:
                print('Energy:', info['energy'])
                print('Grad norm:', info['grad_norm'])

            _, _, done, _, info = env.step(0)

            if verbose >= 3:
                if not info['optim_success']:
                    print(info['optim_message'])
                # print('Optim:', info['optim_success'])
                print('Added gate',
                      ham.pool.get_operators()[shared_state.pool_idx[0]],
                      shared_state.params[0])
                print()

        theta = shared_state.params

        if verbose >= 2:
            print('Theta:', theta)
            print('Gates:', [ham.pool.get_operators()[i] for i in shared_state.pool_idx])
            print('Optimizing phi')

        # Phase 2: Optimize phi using ADAPT ansatz
        ket = shared_state.prepare_state()
        bra = ket.T.conj()
        def get_energy(phi):
            ham.params = phi
            E = (bra @ ham.mat @ ket).item().real
            return E
        
        # Use analytic gradient if provided
        gradient = None
        if hasattr(ham, 'gradient') and callable(ham.gradient):
            # Transform gradient function gradient(self, state, params=None)
            # into gradient(self, params=None)
            gradient = functools.partial(ham.gradient, ket)
        
        last_phi = phi
        def callback(res: OptimizeResult):
            print(res)
            diff = res.x - last_phi
            grad = res.jac

            dot = np.dot(diff.ravel(), grad.ravel())
            last_phi = res.x

        kwargs = {
            # Minimize phi while fixing phi_a0 to 0
            # 'constraints': ({'type': 'eq', 'fun': lambda x: x[0]},),
            'method': 'BFGS',
            # 'jac': gradient,
            # 'callback': callback
        }
        res = minimize(get_energy, x0=phi, **kwargs)

        # Get our optimization results
        if not res.success:
            raise RuntimeWarning('Phi optimization did not converge:', res.message)

        phi: np.ndarray = res.x
        new_ineq_value = res.fun
        if verbose:
            if verbose >= 2:
                print(res)
                print('New phi:', phi)
            print('Energy:', new_ineq_value)
            print()
        iter_ += 1

        ineq_values.append(new_ineq_value)
    
    # Fixme: this saves in reversed order, we should have a reverse call
    # so that we can deserialize and just append to a qiskit circuit easier
    ansatz_obj: List[Tuple[float, SymbolicOperator]] = []
    for pool_idx, theta in zip(shared_state.pool_idx, shared_state.params):
        gate = ham.pool.get_operators()[pool_idx]
        ansatz_obj.append((theta, str(gate)))

    return ineq_values, ansatz_obj[::-1], phi

