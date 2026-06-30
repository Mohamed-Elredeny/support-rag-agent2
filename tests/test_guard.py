"""Tests for the LLM grounding/scope guard on the answer path.

Locks the fix for the false-positive bug: the guard must only fire when the model
*starts* its reply with a sentinel, never when a legitimate answer merely contains
the word "insufficient" / "out of scope" mid-sentence.
"""

from __future__ import annotations

import numpy as np
from app.agent import SupportAgent
from app.config import Settings
from app.models import Decision, Hit, KBEntry


class _FakeEmbedder:
    model_name = "fake"

    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


class _FakeRetriever:
    """Returns a confident, in-scope top hit so route() picks ANSWER."""

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        e1 = KBEntry("Q1", "Account", "reset password", "Click Forgot Password ...")
        e2 = KBEntry("Q3", "Account", "suspended", "Check your email ...")
        return [Hit(e1, 0.80), Hit(e2, 0.50)]


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return self.response


def _agent(llm_response: str) -> SupportAgent:
    settings = Settings(t_high=0.59, t_low=0.56, t_margin=0.02, llm_grounding_guard=True)
    return SupportAgent(_FakeEmbedder(), _FakeRetriever(), _FakeLLM(llm_response), settings)


async def test_guard_declines_on_sentinel() -> None:
    resp = await _agent("OUT_OF_SCOPE").handle("write me code")
    assert resp.decision is Decision.decline


async def test_guard_declines_on_insufficient_sentinel() -> None:
    resp = await _agent("INSUFFICIENT").handle("who is your CEO?")
    assert resp.decision is Decision.decline


async def test_guard_does_not_false_positive_midsentence() -> None:
    # A legitimate answer that merely contains "insufficient" must still ANSWER.
    resp = await _agent("Your reset link is insufficient once it expires after 30 minutes.").handle(
        "how long is the reset link valid?"
    )
    assert resp.decision is Decision.answer
    assert "insufficient" in resp.answer


async def test_answer_passes_through() -> None:
    resp = await _agent("Go to the login page and click Forgot Password.").handle("reset password")
    assert resp.decision is Decision.answer
    assert resp.sources[0].id == "Q1"
