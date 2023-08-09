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
