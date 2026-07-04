"""Domain types and the API request/response schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    answer = "answer"
    clarify = "clarify"
    decline = "decline"


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
        # Embed question + answer so a query can match either side.
        return f"{self.question}\n{self.answer}"


@dataclass(frozen=True)
class Hit:
    entry: KBEntry
    score: float


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    id: str
    category: str
    score: float


class Scores(BaseModel):
    top1: float
    top2: float | None = None
    margin: float | None = None


class ChatResponse(BaseModel):
    decision: Decision
    answer: str
    sources: list[Source]
    scores: Scores
    latency_ms: int
    model: str
