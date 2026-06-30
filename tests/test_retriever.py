"""Retrieval quality test with the REAL embedding model.

Proves semantic search works on paraphrases that share NO keywords with the KB
question ("I forgot my login" -> Q1, "get my money back" -> Q5). Skipped
automatically if fastembed isn't installed; it downloads the ONNX model on first
run, so it's slower than the unit tests (run it in CI / locally, not on every save).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastembed")

from app.config import get_settings
from app.embeddings import Embedder
from app.retriever import InMemoryRetriever

PARAPHRASES = [
    ("I forgot my login and can't get in", "Q1"),
    ("how do I switch the email on my account", "Q2"),
    ("can I get my money back", "Q5"),
    ("how do I stop my subscription", "Q6"),
    ("does it connect to Slack and Jira", "Q7"),
    ("is my data encrypted", "Q8"),
    ("how do I download a backup of everything", "Q9"),
]


@pytest.fixture(scope="module")
def retriever() -> InMemoryRetriever:
    settings = get_settings()
    return InMemoryRetriever.from_kb(settings.kb_path, Embedder(settings.embed_model))


def test_hit_at_1_on_paraphrases(retriever: InMemoryRetriever) -> None:
    embedder = Embedder(get_settings().embed_model)
    hits_at_1 = 0
    for query, expected in PARAPHRASES:
        hits = retriever.search(embedder.embed_query(query), k=3)
        if hits[0].entry.id == expected:
            hits_at_1 += 1
    # Allow one miss; report the rate. (We expect 7/7 in practice.)
    assert hits_at_1 >= len(PARAPHRASES) - 1, f"hit@1 = {hits_at_1}/{len(PARAPHRASES)}"


def test_scores_are_normalized_cosine(retriever: InMemoryRetriever) -> None:
    embedder = Embedder(get_settings().embed_model)
    hits = retriever.search(embedder.embed_query("how do I reset my password"), k=3)
    assert all(-1.0001 <= h.score <= 1.0001 for h in hits)
    assert hits[0].score >= hits[1].score >= hits[2].score
