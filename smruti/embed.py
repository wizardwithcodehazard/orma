"""
embed.py - Text embeddings, the neural similarity primitive used for novelty, entity linking
and retrieval. Vectors are L2-normalized float32, so cosine similarity is a dot product.
"""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalized vectors."""


def normalize(vecs) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def to_blob(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def l2_to_cosine(distance: float) -> float:
    """sqlite-vec reports L2 distance; for unit vectors cos = 1 - d^2 / 2."""
    return 1.0 - (distance * distance) / 2.0


class FastEmbedder:
    """ONNX embeddings via fastembed, loaded lazily on first use."""

    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return normalize(np.asarray(list(self._model.embed(texts))))
