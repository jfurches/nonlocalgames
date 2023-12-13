import networkx as nx
from networkx.generators import complete_graph

from .graph_homomorphism import GraphHomomorphism


class QuantumClique(GraphHomomorphism):
    def __init__(self, clique_n: int, G: nx.Graph, **kwargs):
        G_in = complete_graph(clique_n)
        super().__init__(G_in, G, **kwargs)
