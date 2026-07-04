"""In-memory cosine retriever over the small knowledge base.

At N=10 an exact brute-force dot product is sub-millisecond and always correct,
so there's no need for a vector DB. If the KB ever grows large, swap this class
for a Qdrant-backed one with the same `search` method (see README).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.embeddings import Embedder
from app.models import Hit, KBEntry


def load_kb(kb_path: str | Path) -> list[KBEntry]:
    raw = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    entries = [KBEntry(**e) for e in raw]
    if not entries:
        raise ValueError(f"Knowledge base at {kb_path} is empty.")
    return entries


class InMemoryRetriever:
    def __init__(self, entries: list[KBEntry], matrix: np.ndarray, embed_model: str) -> None:
        if matrix.shape[0] != len(entries):
            raise ValueError("Vector count does not match entry count.")
        self._entries = entries
        self._matrix = matrix.astype(np.float32)
        self.embed_model = embed_model

    @classmethod
    def from_kb(cls, kb_path: str | Path, embedder: Embedder) -> InMemoryRetriever:
        entries = load_kb(kb_path)
        matrix = embedder.embed_documents([e.document_text() for e in entries])
        return cls(entries, matrix, embedder.model_name)

    @classmethod
    def load(cls, index_path: str | Path) -> InMemoryRetriever:
        data = np.load(index_path, allow_pickle=False)
        meta = json.loads(str(data["meta"].item()))
        entries = [KBEntry(**e) for e in meta["entries"]]
        return cls(entries, data["vectors"], meta["embed_model"])

    def save(self, index_path: str | Path) -> None:
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"embed_model": self.embed_model, "entries": [vars(e) for e in self._entries]}
        np.savez(path, vectors=self._matrix, meta=np.array(json.dumps(meta)))

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        # Both sides are L2-normalized, so the dot product is cosine similarity.
        sims = self._matrix @ query_vector
        k = min(k, len(self._entries))
        top_idx = np.argsort(-sims)[:k]
        return [Hit(entry=self._entries[i], score=float(sims[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._entries)


def load_or_build_retriever(
    index_path: str | Path, kb_path: str | Path, embedder: Embedder
) -> InMemoryRetriever:
    """Load the baked index if it matches the current model and KB size, else rebuild."""
    path = Path(index_path)
    if path.exists():
        try:
            retriever = InMemoryRetriever.load(path)
            if retriever.embed_model == embedder.model_name and len(retriever) == len(
                load_kb(kb_path)
            ):
                return retriever
        except (ValueError, OSError, KeyError):
            pass  # corrupted/incompatible index — rebuild from the KB
    return InMemoryRetriever.from_kb(kb_path, embedder)
