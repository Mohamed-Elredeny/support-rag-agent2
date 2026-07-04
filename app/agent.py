"""The agent: embed -> retrieve -> route -> {answer | clarify | decline}.

The route is decided in plain Python from two retrieval signals (top-1 similarity
and the top-1/top-2 margin) before the LLM is called, so it's deterministic and
testable. The LLM only phrases the answer and acts as a grounding/scope guard that
can downgrade answer -> decline, never the other way (fail-safe).
"""

from __future__ import annotations

import time

import structlog
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.embeddings import Embedder
from app.llm_client import OllamaClient, OllamaError
from app.models import ChatResponse, Decision, Hit, KBEntry, Scores, Source
from app.retriever import InMemoryRetriever

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
    """Pure routing policy — unit-tested on synthetic scores."""
    if top1 < settings.t_low:
        return Decision.decline
    if top_is_out_of_scope:
        return Decision.decline
    if top1 >= settings.t_high and margin >= settings.t_margin:
        return Decision.answer
    return Decision.clarify


def _clarify_message(hits: list[Hit]) -> str:
    """Ask about the top-2 in-scope entries: name them if they differ in category;
    if they share one (e.g. refund vs cancel), ask for the goal instead."""
    scoped = [h.entry for h in hits if not h.entry.is_out_of_scope]
    if len(scoped) < 2:
        return "Could you add a bit more detail so I can point you to the right answer?"
    first, second = scoped[0], scoped[1]
    if first.category != second.category:
        return (
            f"I want to point you to the right answer - is your question about "
            f"{first.category} or {second.category}? A little more detail will help."
        )
    return (
        f"Your question looks related to {first.category}, but it could mean a couple of "
        f"different things - could you tell me a bit more about exactly what you need?"
    )


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
        retriever: InMemoryRetriever,
        llm: OllamaClient,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    async def handle(self, question: str) -> ChatResponse:
        started = time.perf_counter()

        # Embedding is blocking/CPU-bound — keep it off the event loop.
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
        """Grounded generation with an extractive fallback and an optional scope guard."""
        prompt = _build_prompt(question, entry)
        try:
            raw = await self._llm.generate(prompt, system=SYSTEM_PROMPT)
        except OllamaError:
            # Never 500 the user: return the KB answer verbatim (still grounded).
            log.warning("ollama_unavailable_extractive_fallback", entry_id=entry.id)
            return Decision.answer, entry.answer

        if self._settings.llm_grounding_guard:
            # The guard replies with EXACTLY one sentinel, so match the START of the
            # reply — a real answer might contain "insufficient" mid-sentence.
            head = raw.strip().upper()
            if head.startswith(_GUARD_OUT_OF_SCOPE):
                return Decision.decline, DECLINE_MESSAGE
            if head.startswith(_GUARD_INSUFFICIENT) or not raw.strip():
                return (
                    Decision.decline,
                    "I don't have that information in my knowledge base. "
                    f"Please contact {self._settings.support_email} for help.",
                )

        return Decision.answer, raw
