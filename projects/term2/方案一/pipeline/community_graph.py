from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix


def build_knn_adjacency(similarity: Any, k: int) -> Tuple[np.ndarray, Dict[str, int]]:
    matrix = np.asarray(similarity, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square similarity matrix.")

    node_count = matrix.shape[0]
    if node_count == 0:
        return np.zeros((0, 0), dtype=float), {"node_count": 0, "effective_k": 0, "edge_count": 0}

    effective_k = max(0, min(int(k), max(node_count - 1, 0)))
    adjacency = np.zeros_like(matrix, dtype=float)

    for row_index in range(node_count):
        row = matrix[row_index].copy()
        row[row_index] = -np.inf
        positive_indices = np.flatnonzero(row > 0.0)
        if effective_k and positive_indices.size > effective_k:
            ranked = positive_indices[np.argsort(row[positive_indices])[::-1][:effective_k]]
        else:
            ranked = positive_indices
        adjacency[row_index, ranked] = row[ranked]

    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    edge_count = int(np.count_nonzero(np.triu(adjacency > 0.0, 1)))
    return adjacency, {
        "node_count": node_count,
        "effective_k": effective_k,
        "edge_count": edge_count,
    }


def adjacency_to_sparse(adjacency: Any) -> csr_matrix:
    matrix = np.asarray(adjacency, dtype=float)
    return csr_matrix(matrix)
