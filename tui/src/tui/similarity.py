
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Measure the angle-based similarity of two vectors."""

    if a.shape != b.shape:
        raise ValueError("Vector dimensions must match")

    length = np.linalg.norm(a) * np.linalg.norm(b)

    if length == 0:
        raise ValueError("Cannot compare zero-length vectors")

    return float(np.dot(a, b) / length)


class SemanticIndex:
    def __init__(self):
        self.notes: list[str] = []
        self.vectors: list[np.ndarray] = []

    def add(self, text: str, vector: np.ndarray) -> int:
        """Store a note and its normalized embedding."""

        if not text.strip():
            raise ValueError("Note cannot be empty")

        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError("Embedding cannot be zero")

        vector = vector / norm

        if self.vectors and vector.shape != self.vectors[0].shape:
            raise ValueError("Embedding dimensions do not match")

        self.notes.append(text)
        self.vectors.append(vector)

        return len(self.notes)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[int, str, float]]:
        """Return notes ranked by cosine similarity."""

        if not self.notes:
            return []

        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_vector = np.asarray(
            query_vector, dtype=np.float32
        )

        norm = np.linalg.norm(query_vector)

        if norm == 0:
            raise ValueError("Query embedding cannot be zero")

        query_vector = query_vector / norm

        # Every row corresponds to one stored note embedding.
        matrix = np.stack(self.vectors)

        if matrix.shape[1] != query_vector.size:
            raise ValueError("Embedding dimensions do not match")

        # Dot product equals cosine similarity for unit vectors.
        scores = matrix @ query_vector

        # Sort highest scores first and keep the best K.
        best = np.argsort(scores)[::-1][:top_k]

        return [
            (int(i) + 1, self.notes[i], float(scores[i]))
            for i in best
        ]
