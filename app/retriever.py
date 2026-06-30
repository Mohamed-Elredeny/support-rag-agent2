"""In-memory exact-cosine retriever over the (tiny) knowledge base.

DESIGN DECISION — why in-memory and not a vector DB:
At N=10, a brute-force dot product over a (10, 384) matrix is *exact* (recall=1
by construction), sub-millisecond, and has zero operational surface. Standing up
Qdrant/Chroma to ANN-index 10 vectors is over-engineering — approximate search
only helps when exact search is too slow, which happens around 10^5-10^6 vectors.

We hide this behind the `Retriever` protocol so the production swap (Qdrant with
HNSW, persistence, metadata filtering, horizontal scaling) is a drop-in: only
this file changes, never the agent. See README "Scaling to production".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from app.embeddings import Embedder
from app.models import Hit, KBEntry


class Retriever(Protocol):
    """Anything that can rank KB entries against a query vector."""

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]: ...


def load_kb(kb_path: str | Path) -> list[KBEntry]:
    raw = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    entries = [
        KBEntry(id=e["id"], category=e["category"], question=e["question"], answer=e["answer"])
        for e in raw
    ]
    if not entries:
        raise ValueError(f"Knowledge base at {kb_path} is empty.")
    return entries


class InMemoryRetriever:
    """Holds normalized document vectors and ranks by cosine similarity."""

    def __init__(self, entries: list[KBEntry], matrix: np.ndarray, embed_model: str) -> None:
        if matrix.shape[0] != len(entries):
            raise ValueError("Vector count does not match entry count.")
        self._entries = entries
        self._matrix = matrix.astype(np.float32)
        self.embed_model = embed_model

    # ---- construction ----

    @classmethod
    def from_kb(cls, kb_path: str | Path, embedder: Embedder) -> InMemoryRetriever:
        entries = load_kb(kb_path)
        matrix = embedder.embed_documents([e.document_text() for e in entries])
        return cls(entries, matrix, embedder.model_name)

    @classmethod
    def load(cls, index_path: str | Path) -> InMemoryRetriever:
        """Load a pre-baked index (fast, offline, deterministic startup)."""
        data = np.load(index_path, allow_pickle=False)
        meta = json.loads(str(data["meta"].item()))
        entries = [KBEntry(**e) for e in meta["entries"]]
        return cls(entries, data["vectors"], meta["embed_model"])

    def save(self, index_path: str | Path) -> None:
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "embed_model": self.embed_model,
            "entries": [vars(e) for e in self._entries],
        }
        np.savez(path, vectors=self._matrix, meta=np.array(json.dumps(meta)))

    # ---- query ----

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        # Both sides are L2-normalized, so the dot product IS cosine similarity.
        sims = self._matrix @ query_vector
        k = min(k, len(self._entries))
        top_idx = np.argsort(-sims)[:k]
        return [Hit(entry=self._entries[i], score=float(sims[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._entries)


def load_or_build_retriever(
    index_path: str | Path,
    kb_path: str | Path,
    embedder: Embedder,
) -> InMemoryRetriever:
    """Prefer the baked index; rebuild from the KB if it is missing or stale."""
    path = Path(index_path)
    if path.exists():
        retriever = InMemoryRetriever.load(path)
        if retriever.embed_model == embedder.model_name and len(retriever) == len(load_kb(kb_path)):
            return retriever
    retriever = InMemoryRetriever.from_kb(kb_path, embedder)
    return retriever
