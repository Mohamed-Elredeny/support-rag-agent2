"""FastAPI surface: /chat, health probes, /metrics, and the chat UI.

Probe split: /healthz is liveness (process only, never checks dependencies, so a
dependency hiccup can't cause a restart storm); /readyz is readiness (gates traffic
on Ollama being reachable).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, PlainTextResponse

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

    # Load everything once per process (the ONNX model + index build are expensive).
    embedder = Embedder(settings.embed_model, settings.embed_cache_dir)
    retriever = load_or_build_retriever(settings.index_path, settings.kb_path, embedder)
    llm = build_llm(settings)

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


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    agent: SupportAgent = request.app.state.agent
    response = await agent.handle(req.question)
    request.app.state.metrics.record(response.decision, response.latency_ms)
    return response


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    """Liveness: the process is up. No dependency checks by design."""
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
async def readyz(request: Request) -> PlainTextResponse:
    """Readiness: the index is loaded and Ollama is reachable."""
    llm: OllamaClient = request.app.state.llm
    ollama_ok = await llm.is_healthy()
    body = f"index=ok ollama={'ok' if ollama_ok else 'down'}"
    return PlainTextResponse(body, status_code=200 if ollama_ok else 503)


@app.get("/metrics", tags=["ops"])
async def metrics(request: Request) -> PlainTextResponse:
    return PlainTextResponse(request.app.state.metrics.render(), media_type="text/plain")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
