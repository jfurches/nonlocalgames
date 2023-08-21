'''Contains algorithms for non-local games'''

from typing import List, Tuple
import functools

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.optimize._optimize import _prepare_scalar_function
from scipy.sparse import csc_matrix
from openfermion import SymbolicOperator

from qiskit.quantum_info import (
    Statevector,
    mutual_information as qiskit_mi
)

from adaptgym import AdaptGame

from nonlocalgames.hamiltonians import NLGHamiltonian

def dual_phase_optim(
        ham: NLGHamiltonian,
        save_mutual_information = False,
        verbose = 0,
        tol = 1e-5,
        seed = None,
        phi_tol = 1e-5,
        **adapt_opt):
    '''Performs dual-phase optimization on a hamiltonian whose ground state represents
    the optimal settings for a non local game
    
    Args:
        ham: The hamiltonian representing the game. It should be a subclass of
            `NLGHamiltonian` to enable optimizing over its parameters.

        save_mutual_information: Whether or not to calculate the mutual information of
            the shared quantum state at each iteration. Default false

        verbose: Controls the amount of output printed to the console. Options
            are

                0. Nothing
                1. Iteration number and energy
                2. Parameters
                3. Everything, including during ADAPT and phi optimization
        
        tol: The convergence tolerance for the bell inequality. Default 1e-5

        seed: A seed for the RNG to produce replicable results.
    '''

    # Starting hamiltonian, random measurement parameters
    np_random = np.random.default_rng(seed=seed)
    phi_random = np_random.normal(scale=np.pi/2, size=ham.params.shape)
    phi = None
    shared_state = None

    ineq_value = np.inf
    new_ineq_value = 0
    iter_ = 1

    metrics = {}

    # Calculate initial energy
    ham.init(seed=seed)
    ham.params = phi_random
    bra = ham.ref_ket.conj().T
    new_ineq_value = (bra @ ham.mat @ ham.ref_ket).item().real
    metrics.setdefault('energy', []).append(new_ineq_value)

    if save_mutual_information:
        mi = mutual_information(ham.ref_ket)
        metrics.setdefault('mutual_information', []).append(mi)

    shared_state = None

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

        env = AdaptGame(ham, criteria='max', **adapt_opt)
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
                print('Added gate',
                      ham.pool.get_operators()[shared_state.pool_idx[0]],
                      shared_state.params[0])
                print()

        theta = shared_state.params
        metrics.setdefault('adapt_pool_gradmax', [0]).append(info['grad_max'])

        if verbose >= 2:
            print('Energy:', shared_state.curr_energy)
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

        def get_variance(phi):
            ham.params = phi
            E = get_energy(phi)
            E2 = (bra @ ham.mat @ ham.mat @ ket).item().real
            return E2 - E ** 2

        # We can save this before the phi optimization since
        # that won't change the MI of our shared state.
        if save_mutual_information:
            mi = mutual_information(ket)
            metrics.setdefault('mutual_information', []).append(mi)

        # Use analytic gradient if provided
        # gradient = None
        # if hasattr(ham, 'gradient') and callable(ham.gradient):
        #     # Transform gradient function gradient(self, state, params=None)
        #     # into gradient(self, params=None)
        #     gradient = functools.partial(ham.gradient, ket)

        # last_phi = phi
        # def callback(res: OptimizeResult):
        #     print(res)
        #     diff = res.x - last_phi
        #     grad = res.jac

        #     dot = np.dot(diff.ravel(), grad.ravel())
        #     last_phi = res.x

        kwargs = {
            'method': 'BFGS',
            'options': {
                'gtol': phi_tol,
                'norm': np.inf,
                'maxiter': 1000,
                # 'learning_rate': 0.1
            }
        }
        res = minimize(get_energy,
                       x0=phi,
                       **kwargs)

        # Get our optimization results
        if not res.success:
            raise RuntimeWarning('Phi optimization did not converge:', res.message, res)

        phi: np.ndarray = res.x
        new_ineq_value = res.fun

        if verbose:
            if verbose >= 2:
                print(res)
                print('New phi:', phi)
                print('Variance:', get_variance(phi))
            print('Energy:', new_ineq_value)
            print()
        iter_ += 1

        metrics.setdefault('energy', []).append(new_ineq_value)

        # Save parameter gradients
        metrics['phi_grad'] = res.jac
        shared_state.H = ham.mat
        metrics['theta_grad'] = shared_state.gradient(theta)

    # Optimization finished, save ansatz in a serialized manner
    ansatz_obj: List[Tuple[float, SymbolicOperator]] = []
    for pool_idx, theta in zip(shared_state.pool_idx, shared_state.params):
        gate = ham.pool.get_operators()[pool_idx]
        ansatz_obj.append((theta, str(gate)))

    # Reverse the gates from Adapt-order to regular
    ansatz_obj = ansatz_obj[::-1]

    return ansatz_obj, phi, metrics

def mutual_information(ket: csc_matrix):
    if hasattr(ket, 'todense'):
        ket = ket.todense()

    d = int(np.sqrt(ket.shape[0]))
    statevector = Statevector(ket, dims=(d, d))
    return qiskit_mi(statevector)

# Taken from https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab
def adam(
    fun,
    x0,
    jac,
    args=(),
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    startiter=0,
    maxiter=1000,
    gtol=1e-5,
    norm=np.inf,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps)
    f = sf.fun
    grad = sf.grad

    success = False
    for i in range(startiter, startiter + maxiter):
        g = grad(x)

        if np.linalg.norm(g, ord=norm) < gtol:
            success = True
            break

        if callback and callback(x):
            break

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

    i += 1
    if success:
        res = OptimizeResult(x=x, fun=f(x), jac=g, nit=i, nfev=i, success=success)
    else:
        # Max iteration achieved
        res = OptimizeResult(x=x, fun=f(x), jac=g, nit=i, nfev=i, success=success,
                            message='Maximum iterations reached')
        
    return res
