"""Read/write helpers for the knowledge base file (data/kb.json).

The admin panel edits the KB through these; after any change the caller rebuilds
the retriever index so the edit takes effect immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings


def _path() -> Path:
    return Path(get_settings().kb_path)


def load() -> list[dict[str, str]]:
    return json.loads(_path().read_text(encoding="utf-8"))


def save(entries: list[dict[str, str]]) -> None:
    _path().write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def get(entry_id: str) -> dict[str, str] | None:
    return next((e for e in load() if e["id"] == entry_id), None)


def upsert(entry: dict[str, str]) -> None:
    """Add a new entry or replace an existing one with the same id."""
    entries = load()
    for i, e in enumerate(entries):
        if e["id"] == entry["id"]:
            entries[i] = entry
            break
    else:
        entries.append(entry)
    save(entries)


def delete(entry_id: str) -> None:
    entries = [e for e in load() if e["id"] != entry_id]
    save(entries)


def count() -> int:
    return len(load())
