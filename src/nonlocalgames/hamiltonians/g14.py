from functools import cache
import json
from importlib import resources

import numpy as np
from scipy.sparse import csc_matrix
from gymnasium.spaces import GraphInstance

from adaptgym.pools import AllPauliPool

from .nlg_hamiltonian import NLGHamiltonian
from ..qinfo import *

class G14(NLGHamiltonian):
    # Phi matrix, players x vertices
    desired_shape = (2, 14, 2)
    # Optimal quantum coloring
    chi_q = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self._qubits = int(np.ceil(np.log2(self.chi_q)))
        self._pool = AllPauliPool(qubits=2 * self._qubits)
    
    def _init_pool(self):
        self._pool.init(qubits=2 * self._qubits)
        self._pool.get_operators()
        self._pool.generate_sparse_ops()

    def _generate_hamiltonian(self) -> csc_matrix:
        g14 = G14._get_graph()
        phi = self._params

        # Projector onto matching color assignments, full subspace
        pcc = G14._pcc(2 * self._qubits)

        # Measurement operator for equal colors
        def M(*args):
            ops = [np.kron(Ry(phi[i, v, 0]), Ry(phi[i, v, 1])) for i, v in enumerate(args)]
            N = len(args)
            idx = list(range(N))

            U = tensor(ops, idx, N)

            return U.T.conj() @ pcc @ U

        d = 2 ** (2 * self._qubits)
        sp_ham = csc_matrix((d, d), dtype=complex)

        # Same vertex asked of each player,
        # p(c1 != c2 | v, v), which works out to
        # 1 - p(c1 = c2 | v, v)
        I = np.eye(d)
        for v in range(len(g14.nodes)):
            sp_ham += I - M(v, v)
        
        # Different vertices, still adjacent. This equivalent to
        # p(c1 = c2 | v1 != v2)
        for v1, v2 in g14.edge_links:
            sp_ham += M(v1, v2)
        
        return sp_ham

    @property
    def ref_ket(self):
        d = 2 ** (2 * self._qubits)
        # Equal superposition state
        ket = csc_matrix(np.full((d, 1), 1 / np.sqrt(d)), dtype=complex)

        return ket

    @cache
    @staticmethod
    def _pcc(qubits: int) -> csc_matrix:
        idx = list(map(lambda x: 5*x, range(G14.chi_q)))
        data = np.ones(len(idx))
        pcc = csc_matrix((data, (idx, idx)), shape=(2 ** qubits, 2 ** qubits), dtype=complex)
        return pcc

    @cache
    @staticmethod
    def _get_graph() -> GraphInstance:
        # First we get G13
        g13_file = resources.files('nonlocalgames.data').joinpath('g13.json')
        with open(g13_file, 'r', encoding='utf-8') as f:
            g13 = json.load(f)
        
        vertices = g13['vertices']
        edges = g13['edges']

        # Convert edges to numerical type
        edges = [
            (vertices.index(v1), vertices.index(v2))
            for v1, v2 in edges
        ]

        # Transform into G14
        apex = len(vertices)
        for vertex in range(len(vertices)):
            edges.append((apex, vertex))
        
        # Convert to GraphInstance
        return GraphInstance(
            nodes=np.ones((apex + 1, 1)),
            edges=None,
            edge_links=np.array(edges)
        )
