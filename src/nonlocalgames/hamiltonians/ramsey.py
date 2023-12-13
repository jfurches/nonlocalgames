import networkx as nx
from networkx.generators import complete_graph
from networkx.generators.expanders import paley_graph

from .quantum_clique import QuantumClique

K4: nx.Graph = complete_graph(4)
P17: nx.Graph = paley_graph(17)


def embedded_p17():
    G = nx.Graph()
    for i in range(6):
        G.add_edge(i, i + 1 % 6)

    for i in range(0, 6, 2):
        G.add_edge(i, i + 2 % 6)

    return G


R6: nx.Graph = embedded_p17()


class Ramsey(QuantumClique):
    def __init__(self, **kwargs):
        super().__init__(4, P17, **kwargs)

class MiniRamsey(QuantumClique):
    def __init__(self, **kwargs):
        super().__init__(4, R6, **kwargs)
