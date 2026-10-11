from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def as_float_matrix(vectors: Any) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")
    return matrix


def compute_cosine_similarity_matrix(vectors: Any) -> np.ndarray:
    matrix = as_float_matrix(vectors)
    if matrix.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    similarity = cosine_similarity(matrix)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    similarity = np.clip(similarity, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity
