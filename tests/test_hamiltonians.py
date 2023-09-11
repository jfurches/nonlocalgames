import warnings
import itertools
import pytest

import numpy as np
from scipy.sparse import SparseEfficiencyWarning
from scipy.linalg import eigvalsh

from nonlocalgames.hamiltonians import (
    CHSHHamiltonian,
    NPartiteSymmetricNLG,
    G14,
    Ramsey
)

from nonlocalgames.hamiltonians.g14 import one_hot
from nonlocalgames.hamiltonians.ramsey import P17

from nonlocalgames import methods

from nonlocalgames.qinfo import (
    is_hermitian, is_diagonal, 
    commutator, is_antihermitian,
    Ry, tensor
)

class TestCHSH:
    @pytest.mark.parametrize('mode', ['optimal', 'normal'])
    def test_properties(self, mode: str):
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

    @pytest.mark.parametrize('layer', ('ry', 'u3'))
    def test_chsh_dpo(self, layer):
        ham = CHSHHamiltonian(measurement_layer=layer)
        *_, metrics = methods.dual_phase_optim(
            ham, seed=42, tol=1e-5, adapt_thresh=1e-3)

        assert np.isclose(metrics['energy'][-1], -2 * np.sqrt(2))


class TestNPS:
    @pytest.mark.parametrize('n', [2,3,4,5])
    def test_properties(self, n: int, trials=5):
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

        # Check node values
        assert graph.nodes.shape[0] == 14
        assert graph.edge_links.min() == 0
        assert graph.edge_links.max() == 13
        assert len(graph.edge_links) == 37 * 2

        # Check apex vertex
        for v in range(13):
            assert (13, v) in graph.edge_links
        
        # Check bidirectional edges
        for edge in graph.edge_links:
            assert edge[::-1] in graph.edge_links

    def test_g14_pcc_no_ancilla(self):
        ham = G14()
        pcc = ham._pcc()
        assert pcc.nnz == 4
        assert is_hermitian(pcc)

        for c1, c2 in itertools.product(range(G14.chi_q), repeat=2):
            v1 = one_hot(c1, G14.chi_q)
            v2 = one_hot(c2, G14.chi_q)

            psi = np.kron(v1, v2)

            if c1 == c2:
                assert np.allclose(pcc @ psi, psi, rtol=0)
            else:
                assert np.allclose(pcc @ psi, 0, rtol=0)
    
    @pytest.mark.parametrize('ancilla', range(1, 3))
    def test_g14_pcc_ancilla(self, ancilla):
        ham = G14(ancilla=ancilla)
        pcc = ham._pcc()

        assert pcc.nnz == 4 * (2 ** (2 * ancilla))
        assert is_hermitian(pcc)

        for c1, c2 in itertools.product(range(G14.chi_q), repeat=2):
            n_hidden_states = 2 ** ancilla
            for h1, h2 in itertools.product(range(n_hidden_states), repeat=2):
                v1 = np.kron(
                    one_hot(h1, n_hidden_states),
                    one_hot(c1, G14.chi_q)
                )
                v2 = np.kron(
                    one_hot(h2, n_hidden_states),
                    one_hot(c2, G14.chi_q)
                )

                psi = np.kron(v1, v2)

                # Make sure we get the same behavior even with ancilla qubits
                if c1 == c2:
                    assert np.allclose(pcc @ psi, psi, rtol=0)
                else:
                    assert np.allclose(pcc @ psi, 0, rtol=0)

    @pytest.mark.parametrize('constrain,layer', itertools.product(
            (True, False),
            ('ry', 'u3', 'cnotry', 'u10')
    ))
    def test_g14_properties(self, constrain, layer):
        ham = G14(init_mode='normal', constrain_phi=constrain, measurement_layer=layer)
        ham.init(seed=42)

        assert ham.mat.shape == (16, 16)
        assert is_hermitian(ham.mat)

        players = 1 if constrain else 2
        if layer == 'ry':
            base_shape = (players, 14, 2, 1)
        elif layer == 'u3':
            base_shape = (players, 14, 2, 3)
        elif layer == 'cnotry':
            base_shape = (players, 14, 2, 2)
        elif layer == 'u10':
            base_shape = (players, 14, 2, 5)

        assert ham.desired_shape == base_shape

    @pytest.mark.parametrize('seed', range(10))
    def test_seeding(self, seed):
        ham = G14(init_mode='normal')
        ham.init(seed=seed)
        mat1 = ham.mat

        ham = G14(init_mode='normal')
        ham.init(seed=seed)
        mat2 = ham.mat

        assert np.allclose(mat1, mat2)
    
    def test_conj(self):
        phi = np.random.uniform(-np.pi, np.pi, size=2)
        Uv = np.kron(Ry(phi[0]), Ry(phi[1]))
        
        # Bob's operator must be conj of Alice's, and therefore
        # must be real
        assert np.allclose(Uv, Uv.conj())
        assert np.all(np.isreal(Uv))


@pytest.fixture(scope='module')
def ramsey():
    '''Fixture to reuse ramsey module so we don't recompute unnecessary things. Just to
    speed up testing.'''
    return Ramsey()

class TestRamsey:
    def test_pvp(self, ramsey: Ramsey):
        pvp = ramsey.pvp
        assert is_hermitian(pvp)
        assert len(pvp.data) == len(P17)
        assert pvp.max() == 1

        for v in range(32):
            vec = one_hot(v, 32)
            vv = np.kron(vec, vec).reshape(-1, 1)
            prod = (vv.T @ pvp @ vv).item()
            if v < 17:
                assert prod == 1
            else:
                assert prod == 0

    def test_pep(self, ramsey: Ramsey):
        pep = ramsey.pep
        assert is_hermitian(pep)
        assert len(pep.data) == len(P17.edges)
        assert pep.max() == 1

        for u, v in P17.edges:
            uvec = one_hot(u, 32)
            vvec = one_hot(v, 32)
            vv = np.kron(uvec, vvec).reshape(-1, 1)
            prod = (vv.T @ pep @ vv).item()
            assert prod == 1

            # Test that the reverse edges are present in the projector
            vv = np.kron(vvec, uvec).reshape(-1, 1)
            prod = (vv.T @ pep @ vv).item()
            assert prod == 1
