from functools import cache, cached_property

import numpy as np
from scipy.sparse import csc_matrix

import networkx as nx
from networkx.generators import complete_graph
from networkx.generators.expanders import paley_graph

from qiskit.quantum_info import Statevector
from adaptgym.pools import PauliPool

from .nlg_hamiltonian import NLGHamiltonian

K4: nx.Graph = complete_graph(4)
P17: nx.Graph = paley_graph(17)

class Ramsey(NLGHamiltonian):
    players = 2
    questions = len(K4)
    qubits = 5

    _system = players * qubits

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pool = PauliPool((2,3), ops='XYZ')
    
    def _generate_hamiltonian(self) -> csc_matrix:
        pvp = self.pvp
        pep = self.pep

        hdim = 2 ** self._system
        ham = csc_matrix((hdim, hdim), dtype=complex)
        for vk in K4:
            Uq = self._ml.uq((vk, vk))
            ham += Uq.T.conj() @ pvp @ Uq
        
        for v1, v2 in K4.edges:
            Uq = self._ml.uq((v1, v2))
            ham += Uq.T.conj() @ pep @ Uq

            Uq = self._ml.uq((v2, v1))
            ham += Uq.T.conj() @ pep @ Uq

        Q = len(K4.nodes) + 2 * len(K4.edges)
        p_q = 1 / Q
        ham *= p_q
        return -ham
    
    def _init_pool(self):
        self._pool.init(qubits=self._system)
        self._pool.get_operators()
        self._pool.generate_sparse_ops()

    @cached_property
    def pvp(self):
        idx = []
        stride = np.array([2 ** self.qubits, 1], dtype=int)
        vec = np.array([0, 0], dtype=int)
        for v in P17:
            vec[:] = v
            i = np.dot(stride, vec)
            idx.append(i)
        
        data = np.ones(len(idx))
        pvp = csc_matrix(
            (data, (idx, idx)),
            shape=(2 ** self._system, 2 ** self._system),
            dtype=complex
        )
        return pvp.todense()
    
    @cached_property
    def pep(self):
        idx = []
        stride = np.array([2 ** self.qubits, 1], dtype=int)
        vec = np.array([0, 0], dtype=int)
        for v1, v2 in P17.edges:
            vec[:] = (v1, v2)
            i = np.dot(stride, vec)
            idx.append(i)
        
        data = np.ones(len(idx))
        pep = csc_matrix(
            (data, (idx, idx)),
            shape=(2 ** self._system, 2 ** self._system),
            dtype=complex
        )
        return pep.todense()

    @cached_property
    def ref_ket(self):
        # |+> state
        label = '+' * self._system
        v = Statevector.from_label(label)
        ket = csc_matrix(v.data, dtype=complex).reshape(-1, 1)

        return ket
