
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Measure the angle-based similarity of two vectors."""

    if a.shape != b.shape:
        raise ValueError("Vector dimensions must match")

    length = np.linalg.norm(a) * np.linalg.norm(b)

    if length == 0:
        raise ValueError("Cannot compare zero-length vectors")

    return float(np.dot(a, b) / length)
