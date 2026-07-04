"""FastAPI surface: /chat, /healthz, /readyz, /metrics, and a tiny chat UI.

Probe discipline (a classic interview point):
- /healthz (liveness): is the *process* alive? Never checks dependencies — a
  dependency hiccup must not trigger a restart storm.
- /readyz (readiness): are dependencies usable? Gates traffic. If Ollama is down
  the pod is pulled from the Service; in-flight requests still degrade gracefully
  via the extractive fallback in the agent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, PlainTextResponse

from app import storage
from app.admin import router as admin_router
from app.agent import SupportAgent
from app.config import Settings, get_settings
from app.embeddings import Embedder
from app.llm_client import OllamaClient
from app.logging import CorrelationIdMiddleware, configure_logging
from app.metrics import Metrics
from app.models import ChatRequest, ChatResponse
from app.retriever import load_or_build_retriever

log = structlog.get_logger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def build_llm(settings: Settings) -> OllamaClient:
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        num_ctx=settings.llm_num_ctx,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        timeout_s=settings.llm_timeout_s,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    storage.init_db()

    # Load everything ONCE per process (expensive: ONNX model + index build).
    embedder = Embedder(settings.embed_model, settings.embed_cache_dir)
    retriever = load_or_build_retriever(settings.index_path, settings.kb_path, embedder)
    llm = build_llm(settings)

    app.state.settings = settings
    app.state.llm = llm
    app.state.metrics = Metrics()
    app.state.agent = SupportAgent(embedder, retriever, llm, settings)
    log.info("startup_complete", kb_size=len(retriever), embed_model=settings.embed_model)

    try:
        yield
    finally:
        await llm.aclose()


app = FastAPI(
    title="Agentic Support Assistant",
    version="0.1.0",
    summary="RAG customer-support agent: answer / clarify / decline over a 10-entry KB.",
    lifespan=lifespan,
)
app.add_middleware(CorrelationIdMiddleware)
app.include_router(admin_router)


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    agent: SupportAgent = request.app.state.agent
    response = await agent.handle(req.question)
    request.app.state.metrics.record(response.decision, response.latency_ms)
    storage.record_chat(
        req.question,
        response.answer,
        response.decision.value,
        response.scores.top1,
        channel="web",
    )
    return response


@app.get("/tickets", tags=["support"])
async def tickets(status: str | None = None) -> list[dict[str, object]]:
    return [
        {
            "id": t.id,
            "channel": t.channel,
            "question": t.question,
            "decision": t.decision,
            "status": t.status,
            "created_at": t.created_at.isoformat(),
        }
        for t in storage.list_tickets(status)
    ]


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    """Liveness: process is up. No dependency checks by design."""
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
async def readyz(request: Request) -> PlainTextResponse:
    """Readiness: the index is loaded and Ollama is reachable."""
    llm: OllamaClient = request.app.state.llm
    ollama_ok = await llm.is_healthy()
    ready = ollama_ok  # the in-memory index is always present once startup succeeds
    body = f"index=ok ollama={'ok' if ollama_ok else 'down'}"
    return PlainTextResponse(body, status_code=200 if ready else 503)


@app.get("/metrics", tags=["ops"])
async def metrics(request: Request) -> PlainTextResponse:
    return PlainTextResponse(request.app.state.metrics.render(), media_type="text/plain")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
