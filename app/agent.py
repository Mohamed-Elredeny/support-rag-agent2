"""The agentic state machine — the heart of the challenge.

Flow:  EMBED -> RETRIEVE -> ROUTE -> {ANSWER | CLARIFY | DECLINE} -> RESPOND

The routing decision is made in *deterministic Python code* on two retrieval
signals (top-1 similarity and the top-1/top-2 margin) BEFORE the LLM is ever
called. That makes the decision testable, auditable, and reliable on a weak 0.5B
model. The LLM only phrases the answer, and is strictly grounded in the single
retrieved entry. An optional LLM grounding/scope guard can only *downgrade*
answer -> decline/clarify (fail-safe), never invent an answer.

Out-of-scope is handled two ways, neither of which hardcodes Q10's text:
  1. Semantic: Q10 is embedded as an "Out of Scope" exemplar; a coding/off-topic
     query that lands on it routes to DECLINE.
  2. Low similarity: anything far from every entry (top1 < t_low) is declined.
  3. Guard: on the answer path, the grounded prompt also tells the model to emit
     OUT_OF_SCOPE for build/code requests (catches the adversarial Q7-vs-Q10 case
     where "write code to call the API" lexically resembles the integrations entry).
"""

from __future__ import annotations

import time

import structlog
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.embeddings import Embedder
from app.llm_client import OllamaClient, OllamaError
from app.models import ChatResponse, Decision, Hit, KBEntry, Scores, Source
from app.retriever import Retriever

log = structlog.get_logger(__name__)

DECLINE_MESSAGE = (
    "That's outside the scope of this assistant. I can help with questions about your "
    "account, billing, integrations, and data & security. For coding help or other topics, "
    "please refer to the developer documentation or your technical team."
)

_GUARD_OUT_OF_SCOPE = "OUT_OF_SCOPE"
_GUARD_INSUFFICIENT = "INSUFFICIENT"

SYSTEM_PROMPT = (
    "You are a customer-support assistant. Answer ONLY using the provided context.\n"
    "- If the user asks you to write, debug, or generate code, or the request is not about "
    "their account, billing, integrations, or data & security, reply with exactly: "
    f"{_GUARD_OUT_OF_SCOPE}\n"
    "- If the context does not contain the answer, reply with exactly: "
    f"{_GUARD_INSUFFICIENT}\n"
    "- Otherwise answer in 1-3 sentences using only facts from the context. Never invent "
    "emails, time periods, prices, or features that are not in the context."
)


def route(top1: float, margin: float, top_is_out_of_scope: bool, settings: Settings) -> Decision:
    """Pure routing policy — unit-tested in isolation on synthetic scores."""
    if top1 < settings.t_low:
        return Decision.decline
    if top_is_out_of_scope:
        return Decision.decline
    if top1 >= settings.t_high and margin >= settings.t_margin:
        return Decision.answer
    return Decision.clarify


def _clarify_message(hits: list[Hit]) -> str:
    """Ask a targeted question naming the top-2 retrieved topics."""
    cats: list[str] = []
    for hit in hits:
        if hit.entry.category not in cats and not hit.entry.is_out_of_scope:
            cats.append(hit.entry.category)
        if len(cats) == 2:
            break
    if len(cats) == 2:
        return (
            f"I want to point you to the right answer - is your question about "
            f"{cats[0]} or {cats[1]}? A little more detail will help."
        )
    return "Could you add a bit more detail so I can point you to the right answer?"


def _build_prompt(question: str, entry: KBEntry) -> str:
    return (
        f"Context (knowledge base entry {entry.id}, category: {entry.category}):\n"
        f"{entry.question}\n{entry.answer}\n\n"
        f"User question: {question}\n\n"
        f"Answer:"
    )


class SupportAgent:
    def __init__(
        self,
        embedder: Embedder,
        retriever: Retriever,
        llm: OllamaClient,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    async def handle(self, question: str) -> ChatResponse:
        started = time.perf_counter()

        # Embedding is CPU-bound/blocking — keep it off the event loop.
        query_vector = await run_in_threadpool(self._embedder.embed_query, question)
        hits = self._retriever.search(query_vector, self._settings.top_k)

        top1 = hits[0].score
        top2 = hits[1].score if len(hits) > 1 else None
        margin = top1 - top2 if top2 is not None else top1

        decision = route(top1, margin, hits[0].entry.is_out_of_scope, self._settings)
        answer_text = ""

        if decision is Decision.answer:
            decision, answer_text = await self._answer(question, hits[0].entry)
        if decision is Decision.clarify:
            answer_text = _clarify_message(hits)
        if decision is Decision.decline:
            answer_text = answer_text or DECLINE_MESSAGE

        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "decision",
            decision=decision.value,
            top1=round(top1, 4),
            margin=round(margin, 4),
            top_id=hits[0].entry.id,
            latency_ms=latency_ms,
        )

        return ChatResponse(
            decision=decision,
            answer=answer_text,
            sources=[
                Source(id=h.entry.id, category=h.entry.category, score=round(h.score, 4))
                for h in hits
            ],
            scores=Scores(
                top1=round(top1, 4),
                top2=round(top2, 4) if top2 is not None else None,
                margin=round(margin, 4),
            ),
            latency_ms=latency_ms,
            model=self._settings.llm_model,
        )

    async def _answer(self, question: str, entry: KBEntry) -> tuple[Decision, str]:
        """Grounded generation with an extractive fallback and an optional guard."""
        prompt = _build_prompt(question, entry)
        try:
            raw = await self._llm.generate(prompt, system=SYSTEM_PROMPT)
        except OllamaError:
            # Resilience: never 500 the user. Return the KB answer verbatim
            # (still fully grounded, just not paraphrased).
            log.warning("ollama_unavailable_extractive_fallback", entry_id=entry.id)
            return Decision.answer, entry.answer

        if self._settings.llm_grounding_guard:
            # The guard prompt tells the model to reply with EXACTLY one sentinel, so
            # match the START of the stripped reply — never a mid-answer mention (a
            # legitimate answer could contain the word "insufficient").
            head = raw.strip().upper()
            if head.startswith(_GUARD_OUT_OF_SCOPE):
                return Decision.decline, DECLINE_MESSAGE
            if head.startswith(_GUARD_INSUFFICIENT) or not raw.strip():
                return (
                    Decision.decline,
                    "I don't have that information in my knowledge base. "
                    "Please contact support@test.com for help.",
                )

        return Decision.answer, raw
