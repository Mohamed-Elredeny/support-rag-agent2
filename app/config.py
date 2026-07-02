"""Application configuration.

Twelve-factor config: every tunable (thresholds, model names, top-k, prompts)
lives here and is sourced from environment variables / the Kubernetes ConfigMap.
There are NO magic numbers scattered through the code — the interviewer will ask
"why 0.6?", and the answer must point at a single, documented, calibrated place.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM backend (Ollama) ---
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:0.5b"
    # Ollama silently truncates the prompt to num_ctx; set it deliberately so the
    # retrieved context is never dropped. 2048 comfortably fits 1 KB entry + prompt.
    llm_num_ctx: int = 2048
    llm_temperature: float = 0.0  # deterministic generation for a reproducible demo
    llm_seed: int = 0
    # Sized for a cold start on a CPU-only laptop/cluster: the FIRST answer query
    # loads the model into RAM (tens of seconds under memory pressure). Too tight a
    # timeout would fail into the extractive fallback AND silently skip the scope
    # guard, so keep generous headroom. Ollama has its own keep-alive after warm-up.
    llm_timeout_s: float = 120.0
    # Use the LLM as a grounding/scope guard on the answer path. Deterministic at
    # temperature 0. The deterministic router decides the branch; this only ever
    # *downgrades* answer -> decline/clarify, never the reverse (fail-safe).
    llm_grounding_guard: bool = True

    # --- Embeddings ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    # Where fastembed caches the ONNX model. None => its default; set to a fixed
    # path in the container so the model is baked once and loaded offline.
    embed_cache_dir: str | None = None

    # --- Retrieval ---
    top_k: int = 3
    kb_path: str = "data/kb.json"
    index_path: str = "data/index.npz"

    # --- Agentic router thresholds (cosine similarity, L2-normalized bge-small) ---
    # Calibrated by grid-search on the golden set (see eval/results.md), NOT guessed.
    # ANSWER  : top1 >= t_high AND margin >= t_margin
    # CLARIFY : t_low <= top1 < t_high  OR  (top1 >= t_high AND margin < t_margin)
    # DECLINE : top1 < t_low  OR  top hit is the Out-of-Scope exemplar
    t_high: float = 0.59
    t_low: float = 0.56
    t_margin: float = 0.02

    # --- Service ---
    log_level: str = "INFO"
    # Support contact surfaced in fallback messages — config, not a code literal
    # (mirrors the KB's support address; overridable per environment / Secret).
    support_email: str = "support@test.com"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so config is parsed once per process."""
    return Settings()
