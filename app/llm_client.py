"""Async client for the Ollama LLM."""

from __future__ import annotations

import asyncio

import httpx
import structlog

log = structlog.get_logger(__name__)


class OllamaError(RuntimeError):
    """Raised when generation fails after exhausting retries."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int,
        temperature: float,
        seed: int,
        timeout_s: float,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s)

    async def generate(self, prompt: str, system: str | None = None) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": self._options,
        }
        if system is not None:
            payload["system"] = system

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post("/api/generate", json=payload)
                resp.raise_for_status()
                return str(resp.json().get("response", "")).strip()
            except httpx.HTTPStatusError as exc:
                # Retry server errors only; a 4xx is a bug, not a transient blip.
                if exc.response.status_code < 500 or attempt == self._max_retries:
                    raise OllamaError(f"Ollama returned {exc.response.status_code}") from exc
                last_exc = exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    break
            await asyncio.sleep(0.5 * (attempt + 1))

        raise OllamaError("Ollama unreachable after retries") from last_exc

    async def is_healthy(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
