from functools import cache, cached_property
from math import ceil

import numpy as np
from scipy.sparse import csc_matrix

import networkx as nx
from networkx.generators import complete_graph
from networkx.generators.expanders import paley_graph

from qiskit.quantum_info import Statevector
from adaptgym.pools import PauliPool, AllPauliPool

from .nlg_hamiltonian import NLGHamiltonian

K4: nx.Graph = complete_graph(4)
P17: nx.Graph = paley_graph(17)

def embedded_p17():
    G = nx.Graph()
    for i in range(6):
        G.add_edge(i, i+1 % 6)
    
    for i in range(0, 6, 2):
        G.add_edge(i, i+2 % 6)
    
    return G

R6: nx.Graph = embedded_p17()

class Ramsey(NLGHamiltonian):
    G_in = K4
    G_out = P17

    players = 2
    questions = len(G_in)
    qubits = ceil(np.log2(len(G_out)))

    _system = players * qubits

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pool = PauliPool((2, 3), ops='XYZ')

    def _generate_hamiltonian(self) -> csc_matrix:
        pvp = self.pvp
        pep = self.pep

        hdim = 2 ** self._system
        ham = csc_matrix((hdim, hdim), dtype=complex)
        for vk in self.G_in:
            Uq = self._ml.uq((vk, vk))
            ham += Uq.T.conj() @ pvp @ Uq
        
        for v1, v2 in self.G_in.edges:
            Uq = self._ml.uq((v1, v2))
            ham += Uq.T.conj() @ pep @ Uq

            Uq = self._ml.uq((v2, v1))
            ham += Uq.T.conj() @ pep @ Uq

        Q = len(self.G_in.nodes) + 2 * len(self.G_in.edges)
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
        for v in self.G_out:
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
        edges = list(self.G_out.edges)

        idx = []
        stride = np.array([2 ** self.qubits, 1], dtype=int)
        vec = np.array([0, 0], dtype=int)
        for v1, v2 in self.G_out.edges:
            vec[:] = (v1, v2)
            i = np.dot(stride, vec)
            idx.append(i)

            # Enumerating edges in undirected graphs sometimes doesn't
            # yield both (v1, v2) and (v2, v1), but for nonlocal games we
            # want that symmetry.
            if (v2, v1) in self.G_out.edges and \
               (v2, v1) not in edges:
                
                vec[:] = (v2, v1)
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


class MiniRamsey(Ramsey):
    G_in = K4
    G_out = R6

    players = 2
    questions = len(G_in)
    qubits = ceil(np.log2(len(G_out)))

    _system = players * qubits

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._pool = AllPauliPool(self._system)
    
    @cached_property
    def pep(self):
        return super().pep
    
    @cached_property
    def pvp(self):
        return super().pvp
