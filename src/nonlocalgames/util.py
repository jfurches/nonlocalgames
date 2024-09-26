from functools import partial
from typing import Dict, Tuple, TypeVar
from queue import PriorityQueue
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
from matplotlib.collections import LineCollection, PathCollection


def load_seeds(path="data/seeds.txt"):
    """Loads some random seeds"""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    seed_func = lambda l: int(l.strip())
    seeds = list(map(seed_func, lines))

    return seeds


T = TypeVar("T")


def from_ket_form(counts: Dict[str, T]) -> Dict[Tuple[int], T]:
    """Transforms the qiskit counts vector from ket form with string keys
    to tuples of integer keys

    Example:
        >>> d = {'01 10': 7}
        >>> from_ket_form(d)
            {(2, 1): 7}
    """
    postprocessed = {}
    from_bin = partial(int, base=2)
    for bitstring, count in counts.items():
        # Transform the binary strings back into integers per player.
        # Qiskit will output a bit string in the format
        # 'bn bn-1 ... b0', where bi is the bit string for classical register i.
        # Additionally, the bits are reversed, i.e. in little endian order
        # with the MSB on the left.
        answers = tuple(map(from_bin, reversed(bitstring.split(" "))))
        postprocessed[answers] = count

    return postprocessed


def draw_graph(
    G: nx.Graph, cmap: str | Colormap = "PiYG", initial_pos=None, vmin=None, vmax=None
):
    """Draws a graph and colors the nodes and vertices based on the `weight` attribute.

    Args:
        G: Graph to draw. Both nodes and vertices should have a `weight` attribute

        cmap: The colormap to use. Can be a matplotlib string or a Colormap instance.

    Returns:
        The Axes of the new figure
    """
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    node_colors = list(map(lambda it: it[-1], G.nodes.data("weight")))
    edge_colors = list(map(lambda it: it[-1], G.edges.data("weight")))

    vmin = vmin or min([*node_colors, *edge_colors])
    vmax = vmax or max([*node_colors, *edge_colors])
    scaled = (np.array(node_colors) - vmin) / (vmax - vmin)
    bordercolors = cmap(np.clip(scaled, 0, 1))

    pos = nx.kamada_kawai_layout(G, weight=None, pos=initial_pos)
    nodes: PathCollection = nx.draw_networkx_nodes(
        G,
        pos,
        node_color="white",
        edgecolors=bordercolors,
        linewidths=2,
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

    labels = G.nodes.data("label")
    for n, label in labels:
        if label is None:
            G.nodes[n]["label"] = str(n)
    nx.draw_networkx_labels(G, pos, dict(G.nodes.data("label")), font_color="black")
    edges.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(edges, ax=ax)
    return ax


def df_to_graph(df: pd.DataFrame, labels=None) -> nx.Graph:
    G = nx.Graph()
    q = PriorityQueue()
    Row = namedtuple("Row", ["va", "vb", "win_rate"])

    # First we queue up everything
    for _, series in df.iterrows():
        row = Row(series.va, series.vb, series.win_rate)
        # Lower priority for edges
        priority = int(series.va != series.vb)
        q.put((priority, row))

    # Now pop from the queue. All edges should be
    # at the end of the queue, so we draw vertices
    # first
    while not q.empty():
        _, row = q.get()
        va, vb = row.va, row.vb
        win_rate = row.win_rate

        if va == vb:
            G.add_node(va, weight=win_rate)
        else:
            if G.has_edge(vb, va):
                w = G.edges[vb, va]["weight"]
                G.edges[vb, va]["weight"] = min(win_rate, w)
            else:
                G.add_edge(va, vb, weight=win_rate)

    if labels is not None:
        for n, label in enumerate(labels):
            G.nodes[n]["label"] = label

    return G
