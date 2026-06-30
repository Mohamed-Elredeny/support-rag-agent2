"""Build (or rebuild) the retrieval index from the knowledge base.

Idempotent: it recomputes the whole (N, 384) matrix from `kb.json` and writes
`data/index.npz`. Running it twice yields the same artifact — no append, no
delete-then-insert window. It is invoked at Docker build time to *bake* the index
into the image (fast, offline startup), and can also be run as a one-shot job:

    python -m ingest.ingest --kb data/kb.json --index data/index.npz

At larger scale this is exactly where the Kubernetes ingestion Job would upsert
into Qdrant instead of writing a local file (see README "Scaling to production").
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.embeddings import Embedder
from app.retriever import InMemoryRetriever


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the retrieval index from the KB.")
    parser.add_argument("--kb", default=settings.kb_path)
    parser.add_argument("--index", default=settings.index_path)
    parser.add_argument("--model", default=settings.embed_model)
    parser.add_argument("--cache-dir", default=settings.embed_cache_dir)
    args = parser.parse_args()

    embedder = Embedder(args.model, args.cache_dir)
    retriever = InMemoryRetriever.from_kb(args.kb, embedder)
    retriever.save(args.index)
    print(f"Ingested {len(retriever)} entries with '{args.model}' -> {args.index}")


if __name__ == "__main__":
    main()
