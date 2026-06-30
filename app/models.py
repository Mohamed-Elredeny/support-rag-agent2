"""API request/response schemas and the core domain types.

The ChatResponse deliberately exposes the *decision trace* (chosen action,
retrieved sources, similarity scores). This makes the agentic behaviour
observable in the demo and the interview — you can SEE why it answered,
clarified, or declined, instead of trusting a black box.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    """The three agentic branches."""

    answer = "answer"
    clarify = "clarify"
    decline = "decline"


# ---- Domain types (not serialized over the wire) ----


@dataclass(frozen=True)
class KBEntry:
    id: str
    category: str
    question: str
    answer: str

    OUT_OF_SCOPE_CATEGORY = "Out of Scope"

    @property
    def is_out_of_scope(self) -> bool:
        return self.category == self.OUT_OF_SCOPE_CATEGORY

    def document_text(self) -> str:
        """What we embed for retrieval: question + answer.

        We embed both because a user query may paraphrase either the question
        ("I forgot my login") or facts only present in the answer ("how long is
        the reset link valid"). Embedding the answer too widens recall.
        """
        return f"{self.question}\n{self.answer}"


@dataclass(frozen=True)
class Hit:
    entry: KBEntry
    score: float


# ---- Wire schemas ----


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="The user's support question.")


class Source(BaseModel):
    id: str
    category: str
    score: float


class Scores(BaseModel):
    top1: float = Field(description="Cosine similarity of the best-matching KB entry.")
    top2: float | None = Field(default=None, description="Second-best similarity.")
    margin: float | None = Field(default=None, description="top1 - top2; small => ambiguous.")


class ChatResponse(BaseModel):
    decision: Decision
    answer: str
    sources: list[Source]
    scores: Scores
    latency_ms: int
    model: str
