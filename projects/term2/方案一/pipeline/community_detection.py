from typing import Any, Dict, List

import igraph as ig
import leidenalg
import numpy as np


def build_weighted_graph(adjacency: Any) -> ig.Graph:
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square adjacency matrix.")
    return ig.Graph.Weighted_Adjacency(matrix.tolist(), mode="undirected", attr="weight", loops=False)


def detect_leiden_communities(adjacency: Any, resolution: float = 1.0) -> Dict[str, Any]:
    matrix = np.asarray(adjacency, dtype=float)
    node_count = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    if node_count == 0:
        return {"membership": [], "community_count": 0, "edge_count": 0, "modularity": 0.0}
    if node_count == 1:
        return {"membership": [0], "community_count": 1, "edge_count": 0, "modularity": 0.0}

    graph = build_weighted_graph(matrix)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
    )
    membership: List[int] = [int(item) for item in partition.membership]
    modularity = float(graph.modularity(membership, weights="weight")) if graph.ecount() else 0.0
    return {
        "membership": membership,
        "community_count": len(set(membership)),
        "edge_count": int(graph.ecount()),
        "modularity": modularity,
    }
