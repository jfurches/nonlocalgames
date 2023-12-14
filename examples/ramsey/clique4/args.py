from dataclasses import asdict, dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from nonlocalgames.hamiltonians import QuantumClique


@dataclass
class TaskArgs:
    seed: int
    graph_name: str
    graph: nx.Graph
    dpo_tol: float = 1e-6
    adapt_tol: float = 1e-3
    phi_tol: float = 1e-5
    theta_tol: float = 1e-9
    layer: str = "ry"

    def ham(self):
        return QuantumClique(4, self.graph, measurement_layer=self.layer)

    def to_dict(self):
        return asdict(self)


@dataclass
class TaskResult:
    df: pd.DataFrame
    state: Sequence[Any]
    phi: np.ndarray
    metrics: dict
    metadata: TaskArgs

    @property
    def energy(self):
        return self.df.energy.min()

    def __hash__(self):
        return self.metadata.seed + hash(self.metadata.graph_name)

    def __eq__(self, other):
        return self.metadata.seed == other.metadata.seed
