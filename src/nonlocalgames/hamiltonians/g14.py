import itertools
from functools import cache, cached_property
from importlib import resources
from typing import List, Tuple

import networkx as nx
import numpy as np
from adaptgym.pools import AllPauliPool
from qiskit.quantum_info import Statevector
from scipy.sparse import csc_matrix

from ..qinfo import *
from .nlg_hamiltonian import NLGHamiltonian


class G14(NLGHamiltonian):
    """Hamiltonian encoding G14 non-local game from [1].

    G14 is a non-local game where the referee asks 2 players,
    Alice and Bob, questions to determine if they've properly
    colored the graph G14. The questions and rules are as follows:

        1. A vertex question (v, v). Both must answer with the same color.
        2. An edge question (v1 ~ v2). Both must answer with different colors.

    Since it's known that G14 admits a quantum 4-coloring, each player
    only needs 2 qubits to represent all possible colors, so this
    hamiltonian acts on 4 qubits.

    The default mode is `violation`, which corresponds to constructing
    a hamiltonian whose energy measures how often the shared quantum
    strategy violates the rules of the game. Minimizing the energy of this
    results in a quantum strategy satisfying the rules of the game and using
    the optimal quantum coloring. Other modes are currently not implemented.

    References:
        [1] https://arxiv.org/abs/1801.03542
    """

    # Optimal quantum coloring
    chi_q = 4

    players = 2
    questions = 14
    qubits = int(np.ceil(np.log2(chi_q)))

    _system = 2 * qubits

    ham_types = ("violation", "nonviolation", "full")

    def __init__(
        self,
        ham_type: str = "violation",
        weighting: str | None = None,
        constrain_phi=True,
        ancilla=0,
        **kwargs,
    ):
        """Construct hamiltonian for G14

        Args:
            ham_type: The hamiltonian type. `violation` minimizes the probability
                that the quantum strategy violates the game rules. `full` both minimizes
                violations and maximizes winning strategies.

            weighting: How to weight the hamiltonian terms $H_v$ and $H_e$. Default is
                None, but passing `balanced` will scale the terms to account for
                imbalance in the number of questions.

            constrain_phi: Enforce the constraint that Bob's measurement operators
                are equal to the complex conjugate of Alice's

            ancilla: Number of auxiliary qubits per player, default 0
        """

        # Construct parameter shape before calling super init. If constrain_phi
        # is true, then we remove the player dimension from the parameters.
        self._constrain_phi = constrain_phi
        if self._constrain_phi:
            self.players = 1

        # Add ancilla qubits to each player
        self.ancilla = ancilla
        self.qubits = self.qubits + ancilla
        self._system = 2 * self.qubits

        # Call super __init__ to generate measurement layer
        super().__init__(**kwargs)

        self._pool = AllPauliPool(qubits=self._system, odd_Y=False)
        assert ham_type in self.ham_types
        self._ham_type = ham_type
        self._weighting = weighting

    def _init_pool(self):
        self._pool.init(qubits=self._system)
        self._pool.get_operators()
        self._pool.generate_sparse_ops()

    def _generate_hamiltonian(self) -> csc_matrix:
        g14 = G14._get_graph()

        # Projector onto matching color assignments, full subspace
        pcc = self._pcc()
        if self._ham_type == "full":
            pcc = 2 * pcc - np.eye(pcc.shape[0])
        elif self._ham_type == "nonviolation":
            pcc = np.eye(pcc.shape[0]) - pcc

        # Measurement operator for equal colors
        def M(*args):
            ops = []
            for i, v in enumerate(args):
                # If we constrain the operators, Bob (1) = Alice.conj()
                if self._constrain_phi and i == 1:
                    op = self._ml.to_unitary(0, v).conj()
                else:
                    op = self._ml.to_unitary(i, v)

                ops.append(op)

            # Make tensor product of unitaries for each player
            N = len(args)
            idx = list(range(N))
            U = tensor(ops, idx, N)
            Mq = U.T.conj() @ pcc @ U

            return Mq

        d = 2**self._system
        sp_ham = csc_matrix((d, d), dtype=complex)

        wv, we = 1, 1
        if self._weighting == "balanced":
            # Re-weight the hamiltonian terms to place more
            # emphasis on the infrequent (vertex) questions.
            nv, ne = len(g14), len(g14.edges)
            w = np.array([nv, ne], dtype=float)
            w /= w.sum()
            we, wv = w

        # Same vertex asked of each player,
        # p(c1 != c2 | v, v), which works out to
        # 1 - p(c1 = c2 | v, v)
        I = np.eye(d)
        for v in range(len(g14)):
            sp_ham += wv * (I - M(v, v))

        # Different vertices, still adjacent. This equivalent to
        # p(c1 = c2 | v1 != v2)
        for e in g14.edges:
            sp_ham += we * M(*e)
            sp_ham += we * M(*e[::-1])

        if self._ham_type == "nonviolation":
            Q = len(g14) + 2 * len(g14.edges)
            sp_ham *= -1 / Q

        return sp_ham

    @cached_property
    def ref_ket(self):
        # |+> state
        label = "+" * self._system
        v = Statevector.from_label(label)
        ket = csc_matrix(v.data, dtype=complex).reshape(-1, 1)

        return ket

    @cache
    def _pcc(self) -> csc_matrix:
        # Old no-ancilla code
        if self.ancilla == 0:
            # The equal color projector consists of
            #   |00><00| + |11><11| + ... + |33><33|.
            # This is a diagonal matrix, very sparse. |00><00| obviously
            # has index (0, 0). Color c projector |cc> is 4*c + c = 5c,
            # hence we can just construct the sparse matrix with 1-entries
            # at (5c, 5c) for c = (0, 1, 2, 3).
            idx = list(map(lambda x: 5 * x, range(G14.chi_q)))
            data = np.ones(len(idx))
            pcc = csc_matrix(
                (data, (idx, idx)),
                shape=(2**self._system, 2**self._system),
                dtype=complex,
            )

            return pcc

        # New color projector, has hidden qubits like |hc>a x |hc>b. Instead
        # of a fancy sparse matrix index method, we'll just compute the outer
        # products explicitly

        # Compute the number of qubits required to represent chi_q just to be general
        chi_q_qubits = int(np.ceil(np.log2(G14.chi_q)))

        pcc = csc_matrix((2**self._system, 2**self._system), dtype=complex)
        for c in range(G14.chi_q):
            # |c> vector
            c_ket = one_hot(c, 2**chi_q_qubits)

            # Add in all possible hidden states
            n_hidden_states = 2**self.ancilla
            for h in itertools.product(
                range(n_hidden_states), repeat=max(2, self.players)
            ):
                # Tensor the players together, forming
                # |hi c> x |hi c> ...
                ket = None
                for hi in h:
                    hi_ket = one_hot(hi, n_hidden_states)
                    if ket is None:
                        ket = np.kron(hi_ket, c_ket)
                    else:
                        ket = np.kron(ket, np.kron(hi_ket, c_ket))

                projector = np.outer(ket, ket)
                pcc += projector

        pcc = csc_matrix(pcc)
        return pcc

    @cache
    @staticmethod
    def _get_graph() -> nx.Graph:
        """Returns the G14 graph"""
        g14_file = resources.files("nonlocalgames.data").joinpath("g14.g6")
        return nx.read_graph6(g14_file)

    @staticmethod
    def get_questions() -> List[Tuple[int, int]]:
        """Enumerates all possible questions for the players
        (both edges and their reverse)"""
        G = G14._get_graph()
        questions = [(v, v) for v in G]
        for e in G.edges:
            questions.append(e)
            questions.append(e[::-1])

        return questions


def one_hot(i: int, N: int):
    x = np.zeros(N)
    x[i] = 1
    return x
