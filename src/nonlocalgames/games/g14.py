from importlib import resources
from typing import List, Tuple
import json

import networkx as nx


class G14:
    """Utility class to access G14 graph without adapt-gym package"""

    @staticmethod
    def get_graph() -> nx.Graph:
        """Returns the G14 graph"""
        # Manually construct from G13 and adding an apex vertex
        g13_file = resources.files("nonlocalgames.data").joinpath("g13.json")
        with open(g13_file, "r") as f:
            g13_data = json.load(f)

        # Scan vertex list and form a mapping letter -> index
        vertices = g13_data["vertices"]
        vertices = {v: i for i, v in enumerate(vertices)}

        # Construct the graph
        G = nx.Graph()
        for e in g13_data["edges"]:
            G.add_edge(vertices[e[0]], vertices[e[1]])

        # Add apex vertex
        G.add_node(13)
        for i in range(len(vertices)):
            G.add_edge(i, 13)

        return G

    @staticmethod
    def get_questions() -> List[Tuple[int, int]]:
        """Enumerates all possible questions for the players
        (both edges and their reverse)"""
        G = G14.get_graph()
        questions = [(v, v) for v in G]
        for e in G.edges:
            questions.append(e)
            questions.append(e[::-1])

        return questions
