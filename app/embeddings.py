"""bge-small embeddings via fastembed (ONNX runtime, CPU-only).

bge models are asymmetric: queries need an instruction prefix, documents don't.
fastembed handles that split for us (`query_embed` vs `embed`). Vectors are
L2-normalized on both sides so cosine similarity is just a dot product.
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized: np.ndarray = (matrix / norms).astype(np.float32)
    return normalized


class Embedder:
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        # Imported lazily so the unit tests don't need the ONNX runtime.
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed KB passages (no prefix). Returns an (n, dim) normalized matrix."""
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return _l2_normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a user query with the bge query prefix. Returns a (dim,) vector."""
        vector = np.array(next(iter(self._model.query_embed([text]))), dtype=np.float32)
        query_vector: np.ndarray = _l2_normalize(vector[None, :])[0]
        return query_vector
