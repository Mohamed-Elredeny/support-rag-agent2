"""Application settings. Every tunable lives here and is overridable via env vars
or the Kubernetes ConfigMap — no magic numbers scattered through the code."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM backend (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:0.5b"
    llm_num_ctx: int = 2048  # set explicitly; Ollama silently truncates to num_ctx
    llm_temperature: float = 0.0  # deterministic generation
    llm_seed: int = 0
    llm_timeout_s: float = 120.0  # generous headroom for a CPU cold start
    llm_grounding_guard: bool = True  # guard may only downgrade answer -> decline

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_cache_dir: str | None = None  # None => fastembed default cache

    # Retrieval
    top_k: int = 3
    kb_path: str = "data/kb.json"
    index_path: str = "data/index.npz"

    # Router thresholds (cosine similarity). Calibrated on the golden set — see eval/results.md.
    t_high: float = 0.59  # answer only if top1 >= t_high
    t_low: float = 0.56  # decline if top1 < t_low
    t_margin: float = 0.02  # and require this top1-top2 gap to answer instead of clarify

    # Storage
    db_path: str = "data/app.db"

    # Service
    log_level: str = "INFO"
    support_email: str = "support@test.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
