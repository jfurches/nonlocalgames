from functools import cached_property
from math import ceil

import numpy as np
from scipy.sparse import csc_matrix

import networkx as nx

from qiskit.quantum_info import Statevector
from adaptgym.pools import PauliPool

from .nlg_hamiltonian import NLGHamiltonian


class GraphHomomorphism(NLGHamiltonian):
    """Hamiltonian representing a graph homomorphism."""

    players = 2

    def __init__(self, G_in: nx.Graph, G_out: nx.Graph, **kwargs):
        super().__init__(**kwargs)

        self.G_in = G_in
        self.G_out = G_out
        self._pool = PauliPool((2, 3), ops="XYZ")

        self.questions = len(self.G_in)
        self.qubits = ceil(np.log2(len(self.G_out)))

        self._system = self.players * self.qubits

    def _generate_hamiltonian(self) -> csc_matrix:
        pvp = self.pvp
        pep = self.pep

        hdim = 2**self._system
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
        stride = np.array([2**self.qubits, 1], dtype=int)
        vec = np.array([0, 0], dtype=int)
        for v in self.G_out:
            vec[:] = v
            i = np.dot(stride, vec)
            idx.append(i)

        data = np.ones(len(idx))
        pvp = csc_matrix(
            (data, (idx, idx)),
            shape=(2**self._system, 2**self._system),
            dtype=complex,
        )
        return pvp.todense()

    @cached_property
    def pep(self):
        edges = list(self.G_out.edges)

        idx = []
        stride = np.array([2**self.qubits, 1], dtype=int)
        vec = np.array([0, 0], dtype=int)
        for v1, v2 in self.G_out.edges:
            vec[:] = (v1, v2)
            i = np.dot(stride, vec)
            idx.append(i)

            # Enumerating edges in undirected graphs sometimes doesn't
            # yield both (v1, v2) and (v2, v1), but for nonlocal games we
            # want that symmetry.
            if (v2, v1) in self.G_out.edges and (v2, v1) not in edges:
                vec[:] = (v2, v1)
                i = np.dot(stride, vec)
                idx.append(i)

        data = np.ones(len(idx))
        pep = csc_matrix(
            (data, (idx, idx)),
            shape=(2**self._system, 2**self._system),
            dtype=complex,
        )
        return pep.todense()

    @cached_property
    def ref_ket(self):
        # |+> state
        label = "+" * self._system
        v = Statevector.from_label(label)
        ket = csc_matrix(v.data, dtype=complex).reshape(-1, 1)

        return ket
