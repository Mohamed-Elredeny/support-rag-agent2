"""Embedding model wrapper around fastembed (ONNX runtime, CPU-only).

Why fastembed over sentence-transformers? fastembed runs bge-small-en-v1.5 via
ONNX and never pulls the torch+CUDA tree (~0.5-2 GB saved in the image) — we run
on CPU, so the GPU stack is dead weight.

The single most important correctness detail for BGE models is the
**query/document asymmetry**: queries must carry the instruction prefix
("Represent this sentence for searching relevant passages:") while documents
must NOT. fastembed handles this for us via `query_embed` vs `embed`; applying
the prefix to documents would be a silent correctness bug. We L2-normalize on
both sides so cosine similarity reduces to a dot product downstream.
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized: np.ndarray = (matrix / norms).astype(np.float32)
    return normalized


class Embedder:
    """Thin, typed wrapper. Loads the ONNX model once and reuses it."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        # Imported lazily so the unit tests (router/API contract) don't need the
        # ONNX runtime, and import time stays cheap.
        from fastembed import TextEmbedding

        self.model_name = model_name
        # cache_dir pins WHERE fastembed stores/loads the ONNX model. We set it
        # explicitly so the model can be baked into the image at a known path and
        # loaded fully offline at runtime (HF_HUB_OFFLINE=1).
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self._model = TextEmbedding(model_name=model_name, **kwargs)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed KB passages (no instruction prefix). Returns (n, dim), normalized."""
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return _l2_normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a user query WITH the bge query-instruction prefix. Returns (dim,)."""
        vector = np.array(next(iter(self._model.query_embed([text]))), dtype=np.float32)
        query_vector: np.ndarray = _l2_normalize(vector[None, :])[0]
        return query_vector
