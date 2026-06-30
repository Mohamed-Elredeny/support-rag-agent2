"""Shared fixtures. The API contract tests use a FAKE agent so they run fast and
fully offline — no embedding model, no Ollama. This isolates the HTTP contract
from the (separately tested) retrieval and routing logic.
"""

from __future__ import annotations

import pytest
from app.api import app
from app.metrics import Metrics
from app.models import ChatResponse, Decision, Scores, Source
from fastapi.testclient import TestClient


class FakeAgent:
    """Deterministic stand-in: maps a few keywords to each branch."""

    async def handle(self, question: str) -> ChatResponse:
        q = question.lower()
        if any(w in q for w in ("code", "script", "weather", "sql")):
            return ChatResponse(
                decision=Decision.decline,
                answer="That's outside the scope of this assistant.",
                sources=[Source(id="Q10", category="Out of Scope", score=0.21)],
                scores=Scores(top1=0.21, top2=0.18, margin=0.03),
                latency_ms=1,
                model="fake",
            )
        if any(w in q for w in ("can't", "cant", "access", "trouble")):
            return ChatResponse(
                decision=Decision.clarify,
                answer="Is this about Account or Billing?",
                sources=[Source(id="Q1", category="Account", score=0.58)],
                scores=Scores(top1=0.58, top2=0.56, margin=0.02),
                latency_ms=1,
                model="fake",
            )
        return ChatResponse(
            decision=Decision.answer,
            answer="Go to the login page and click Forgot Password.",
            sources=[Source(id="Q1", category="Account", score=0.74)],
            scores=Scores(top1=0.74, top2=0.41, margin=0.33),
            latency_ms=1,
            model="fake",
        )


@pytest.fixture
def client() -> TestClient:
    # NOTE: not used as a context manager => the real lifespan (which loads the
    # ONNX model + Ollama client) does NOT run. We inject state manually instead.
    app.state.agent = FakeAgent()
    app.state.metrics = Metrics()
    return TestClient(app)
