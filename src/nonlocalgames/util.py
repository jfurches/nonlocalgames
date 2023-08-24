from functools import partial
from typing import Dict, Tuple, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
from matplotlib.collections import LineCollection, PathCollection

from nonlocalgames.hamiltonians import G14


def load_seeds(path = 'data/seeds.txt'):
    '''Loads some random seeds'''
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    seed_func = lambda l: int(l.strip())
    seeds = list(map(seed_func, lines))
    
    return seeds

T = TypeVar('T')

def from_ket_form(counts: Dict[str, T]) -> Dict[Tuple[int], T]:
    '''Transforms the qiskit counts vector from ket form with string keys
    to tuples of integer keys
    
    Example:
        >>> d = {'01 10': 7}
        >>> from_ket_form(d)
            {(2, 1): 7}
    '''
    postprocessed = {}
    from_bin = partial(int, base=2)
    for bitstring, count in counts.items():
        # Transform the binary strings back into integers per player.
        # Qiskit will output a bit string in the format
        # 'bn bn-1 ... b0', where bi is the bit string for classical register i.
        # Additionally, the bits are reversed, i.e. in little endian order
        # with the MSB on the left.
        answers = tuple(map(from_bin, reversed(bitstring.split(' '))))
        postprocessed[answers] = count

    return postprocessed

def draw_graph(G: nx.Graph, cmap: str | Colormap = 'PiYG'):
    '''Draws a graph and colors the nodes and vertices based on the `weight` attribute.
    
    Args:
        G: Graph to draw. Both nodes and vertices should have a `weight` attribute
        
        cmap: The colormap to use. Can be a matplotlib string or a Colormap instance.
    
    Returns:
        The Axes of the new figure
    '''
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    node_colors = list(map(lambda it: it[-1], G.nodes.data('weight')))
    edge_colors = list(map(lambda it: it[-1], G.edges.data('weight')))

    vmin, vmax = min([*node_colors, *edge_colors]), max([*node_colors, *edge_colors])

    pos = nx.kamada_kawai_layout(G)
    nodes: PathCollection = nx.draw_networkx_nodes(
        G,
        pos,
        node_color='white',
        edgecolors=cmap(node_colors),
        node_size=500,
    )
    edges: LineCollection = nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        edge_cmap=cmap,
        edge_vmin=vmin,
        edge_vmax=vmax,
        width=2,
    )

    labels = G.nodes.data('label')
    for n, label in labels:
        if label is None:
            G.nodes[n]['label'] = str(n)
    nx.draw_networkx_labels(G, pos, dict(G.nodes.data('label')), font_color="black")
    edges.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(edges, ax=ax)
    return ax
