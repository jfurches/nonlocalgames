import warnings
import itertools
import pytest

import numpy as np
from scipy.sparse import SparseEfficiencyWarning
from scipy.linalg import eigvalsh

from nonlocalgames.hamiltonians import (
    CHSHHamiltonian,
    NPartiteSymmetricNLG,
    G14
)

from nonlocalgames.qinfo import (
    is_hermitian, is_diagonal, 
    commutator, is_antihermitian
)

class TestHamiltonians:
    @pytest.mark.parametrize('mode', ['optimal', 'normal'])
    def test_chsh(self, mode: str):
        ham = CHSHHamiltonian(init_mode=mode)
        ham.init()

        assert is_hermitian(ham.mat)

        if mode == 'optimal':
            # Check the optimal quantum violation is correct
            w = eigvalsh(ham.mat)
            min_eigval = min(w.real)
            assert np.isclose(min_eigval, -2*np.sqrt(2))
        elif mode == 'normal':
            # Check that seeding produces the same hamiltonian
            ham.init(seed=42)
            mat1 = ham.mat
            params1 = ham.params

            ham = CHSHHamiltonian(init_mode=mode)
            ham.init(seed=42)
            mat2 = ham.mat
            params2 = ham.params

            assert np.allclose(mat1, mat2, rtol=0)
            assert np.allclose(params1, params2, rtol=0)

    @pytest.mark.parametrize('n', [2,3,4,5])
    def test_Npartite(self, n: int, trials=5):
        ham = NPartiteSymmetricNLG(n)
        ham.init()

        is_zero = lambda A: np.allclose(A, 0j)

        for _ in range(trials):
            with warnings.catch_warnings():
                # Error if we generate a diagonal hamiltonian or we aren't being efficient
                # with sparse matrices (important for larger sizes since every optimization step
                # will regenerate the hamiltonian multiple times)
                warnings.filterwarnings(action='error', category=RuntimeWarning)
                warnings.filterwarnings(action='error', category=SparseEfficiencyWarning)

                # Generate new random starting hamiltonian (phi)
                ham.params = np.random.normal(scale=np.pi/2, size=2*n)
                H = ham.mat
                assert is_hermitian(H)
                assert not is_diagonal(H)

                # Check our pool commutators are not all zero
                pool_ops = ham.pool.spmat_ops
                commutators = [commutator(H, A) for A in pool_ops]
                assert any([not is_zero(c) for c in commutators])

                # Make sure the operators are antihermitian
                assert all(map(is_antihermitian, pool_ops))

                grads = []
                for c in commutators:
                    # Commutator of [H, A] needs to be an observable
                    assert is_hermitian(c)

                    # Compute manually the Adapt gradient expression
                    ket = ham.ref_ket
                    grad = ket.conj().transpose() @ c @ ket
                    grad = grad[0, 0]

                    # Gradient better be real
                    assert np.isclose(grad.imag, 0)
                    grads.append(grad.real)
                
                # Make sure some operator has nonzero gradient since the 
                # probability we chose optimal parameters is 0
                assert not np.allclose(grads, 0)


class TestG14:
    def test_g14_graph(self):
        graph = G14._get_graph()

        assert graph.nodes.shape[0] == 14
        assert graph.edge_links.min() == 0
        assert graph.edge_links.max() == 13

        # Check apex vertex
        for v in range(13):
            assert (13, v) in graph.edge_links

    def test_g14_pcc(self):
        pcc = G14._pcc(4)
        assert pcc.nnz == 4

        for c1, c2 in itertools.product(range(G14.chi_q), repeat=2):
            v1 = np.zeros(4)
            v2 = np.zeros(4)
            v1[c1] = 1
            v2[c2] = 1

            psi = np.kron(v1, v2)

            if c1 == c2:
                assert np.allclose(pcc @ psi, psi, rtol=0)
            else:
                assert np.allclose(pcc @ psi, 0, rtol=0)
    
    def test_g14(self):
        ham = G14(init_mode='normal')
        ham.init(seed=42)
        mat1 = ham.mat

        ham = G14(init_mode='normal')
        ham.init(seed=42)
        mat2 = ham.mat

        assert np.allclose(mat1, mat2)
        assert mat1.shape == (16, 16)
        assert is_hermitian(mat1)
