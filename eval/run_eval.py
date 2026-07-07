"""Offline evaluation of retrieval quality and the routing policy.

Runs the REAL embedding model + the deterministic router (no Ollama needed — the
branch is decided before the LLM is called). Emits:
  - retrieval: recall@1, recall@3, MRR over the ANSWER-labeled queries
  - routing (cross-validated): the HONEST out-of-sample accuracy + confusion matrix.
    Thresholds are grid-searched on each train split and scored on the held-out fold,
    so no query is ever scored by a model that was tuned on it.
  - routing (in-sample): the same metrics at the shipped thresholds for reference —
    the gap to the cross-validated number is the calibration's optimism.
  - a grid-search calibration of (t_high, t_low, t_margin) so the thresholds are
    *derived from the data*, not guessed; the costliest error (decline routed as
    answer = a confident wrong answer) is the tie-breaker.

Writes a Markdown report to eval/results.md.

    python -m eval.run_eval              # report at the configured thresholds
    python -m eval.run_eval --calibrate  # grid-search and report at the best point
    python -m eval.run_eval --folds 5    # k for the cross-validation (default 5)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from app.agent import route
from app.config import Settings, get_settings
from app.embeddings import Embedder
from app.models import Decision
from app.retriever import InMemoryRetriever

BRANCHES = [Decision.answer, Decision.clarify, Decision.decline]


@dataclass
class Record:
    query: str
    expected_decision: str
    expected_id: str | None
    top1: float
    margin: float
    top_id: str
    gold_rank: int | None  # 0-based rank of expected_id within top-k, else None


def _embed_all(
    items: list[dict], retriever: InMemoryRetriever, embedder: Embedder, top_k: int
) -> list[Record]:
    records: list[Record] = []
    for it in items:
        hits = retriever.search(embedder.embed_query(it["query"]), top_k)
        top1 = hits[0].score
        top2 = hits[1].score if len(hits) > 1 else top1
        gold_rank = None
        if it.get("expected_id"):
            gold_rank = next(
                (i for i, h in enumerate(hits) if h.entry.id == it["expected_id"]), None
            )
        records.append(
            Record(
                query=it["query"],
                expected_decision=it["expected_decision"],
                expected_id=it.get("expected_id"),
                top1=round(top1, 4),
                margin=round(top1 - top2, 4),
                top_id=hits[0].entry.id,
                gold_rank=gold_rank,
            )
        )
    return records


def _predict(r: Record, settings: Settings) -> Decision:
    return route(r.top1, r.margin, r.top_id == "Q10", settings)


def _predictions(records: list[Record], settings: Settings) -> list[Decision]:
    return [_predict(r, settings) for r in records]


def retrieval_metrics(records: list[Record]) -> dict[str, float]:
    answerable = [r for r in records if r.expected_id]
    n = len(answerable)
    recall_at_1 = sum(1 for r in answerable if r.gold_rank == 0) / n
    recall_at_3 = sum(1 for r in answerable if r.gold_rank is not None) / n
    mrr = sum(1.0 / (r.gold_rank + 1) for r in answerable if r.gold_rank is not None) / n
    return {"n": n, "recall@1": recall_at_1, "recall@3": recall_at_3, "MRR": mrr}


def confusion(records: list[Record], preds: list[Decision]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {b.value: defaultdict(int) for b in BRANCHES}
    for r, p in zip(records, preds, strict=True):
        matrix[r.expected_decision][p.value] += 1
    return matrix


def accuracy(records: list[Record], preds: list[Decision]) -> float:
    correct = sum(1 for r, p in zip(records, preds, strict=True) if p.value == r.expected_decision)
    return correct / len(records)


def _frange(start: float, stop: float, step: float) -> list[float]:
    out, x = [], start
    while x < stop + 1e-9:
        out.append(round(x, 2))
        x += step
    return out


def calibrate(records: list[Record], base: Settings) -> Settings:
    """Grid-search thresholds. Objective: maximize accuracy; tie-break by fewest
    decline->answer mistakes (the costliest error), then fewest answer->decline."""
    best: tuple[tuple[float, int, int], Settings] | None = None
    for th in _frange(0.55, 0.74, 0.02):
        for tl in _frange(0.40, th - 0.02, 0.02):
            for tm in _frange(0.02, 0.09, 0.01):
                s = base.model_copy(update={"t_high": th, "t_low": tl, "t_margin": tm})
                preds = _predictions(records, s)
                cm = confusion(records, preds)
                acc = accuracy(records, preds)
                key = (acc, -cm["decline"]["answer"], -cm["answer"]["decline"])
                if best is None or key > best[0]:
                    best = (key, s)
    assert best is not None
    return best[1]


def stratified_folds(records: list[Record], k: int) -> list[list[int]]:
    """Deterministic stratified k-fold: group record indices by expected_decision, then
    round-robin them into folds so each fold keeps a similar answer/clarify/decline mix.
    No randomness — the golden-set order fixes the split, so the report is reproducible."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        buckets[r.expected_decision].append(i)
    folds: list[list[int]] = [[] for _ in range(k)]
    for _, idxs in sorted(buckets.items()):
        for j, idx in enumerate(idxs):
            folds[j % k].append(idx)
    return folds


def cross_val_predict(
    records: list[Record], base: Settings, k: int
) -> tuple[list[Decision], list[Settings]]:
    """Out-of-sample predictions. For each fold, calibrate thresholds on the OTHER folds
    and predict the held-out fold — so every query is scored by a model that never saw it.
    Returns predictions aligned to `records`, plus the per-fold calibrated Settings."""
    folds = stratified_folds(records, k)
    preds: list[Decision | None] = [None] * len(records)
    fold_settings: list[Settings] = []
    for test_idx in folds:
        held_out = set(test_idx)
        train = [records[i] for i in range(len(records)) if i not in held_out]
        tuned = calibrate(train, base)
        fold_settings.append(tuned)
        for i in test_idx:
            preds[i] = _predict(records[i], tuned)
    assert all(p is not None for p in preds)
    return [p for p in preds if p is not None], fold_settings


def _confusion_block(matrix: dict[str, dict[str, int]]) -> list[str]:
    lines = [
        "Confusion matrix (rows = expected, cols = predicted):",
        "",
        "| expected \\ predicted | answer | clarify | decline | recall |",
        "|---|---|---|---|---|",
    ]
    for b in BRANCHES:
        row = matrix[b.value]
        total = sum(row.values())
        recall = row[b.value] / total if total else 0.0
        lines.append(
            f"| **{b.value}** | {row['answer']} | {row['clarify']} | {row['decline']} "
            f"| {recall:.0%} |"
        )
    return lines


def render_report(
    records: list[Record],
    shipped: Settings,
    shipped_label: str,
    cv_preds: list[Decision],
    fold_settings: list[Settings],
    k: int,
) -> str:
    rm = retrieval_metrics(records)

    cv_matrix = confusion(records, cv_preds)
    cv_acc = accuracy(records, cv_preds)

    in_preds = _predictions(records, shipped)
    in_matrix = confusion(records, in_preds)
    in_acc = accuracy(records, in_preds)

    lines = [
        "# Evaluation Results",
        "",
        "_Generated by `python -m eval.run_eval --calibrate`. Reproducible: temperature 0, "
        "fixed KB, deterministic router, deterministic stratified folds._",
        "",
        f"Embedding model: `{shipped.embed_model}` · top_k={shipped.top_k}",
        "",
        "## Retrieval (over ANSWER-labeled paraphrases)",
        "",
        f"- queries: **{rm['n']}**",
        f"- recall@1: **{rm['recall@1']:.1%}**",
        f"- recall@3: **{rm['recall@3']:.1%}**",
        f"- MRR: **{rm['MRR']:.3f}**",
        "",
        "Perfect recall is expected at N=10: the hard problem here is *calibration / "
        "abstention*, not recall — which is why the routing analysis below matters more.",
        "",
        f"## Routing — cross-validated ({k}-fold, honest)",
        "",
        "Thresholds are calibrated on each train split and scored on the held-out fold, so "
        "every query is predicted by a model that never saw it. **This is the number to trust** "
        "— it estimates how the calibration generalizes, not how well it memorized this set.",
        "",
        f"- out-of-sample accuracy: **{cv_acc:.1%}** ({len(records)} queries)",
        "",
    ]
    lines += _confusion_block(cv_matrix)
    lines += [
        "",
        "### Threshold stability across folds",
        "",
        "Near-identical thresholds across folds ⇒ the calibration is robust, not an artefact of "
        "one particular split. Wide swings would be an honest signal the set is too small.",
        "",
        "| fold | t_high | t_low | t_margin |",
        "|---|---|---|---|",
    ]
    for i, s in enumerate(fold_settings):
        lines.append(f"| {i + 1} | {s.t_high} | {s.t_low} | {s.t_margin} |")
    lines += [
        "",
        f"## Routing — at shipped thresholds ({shipped_label}, in-sample reference)",
        "",
        f"Shipped thresholds: t_high={shipped.t_high}, t_low={shipped.t_low}, "
        f"t_margin={shipped.t_margin}.",
        "",
        f"- in-sample accuracy: **{in_acc:.1%}** — optimistic: the same data chose these "
        f"thresholds. The gap to the **{cv_acc:.1%}** cross-validated figure above is exactly "
        f"the calibration's optimism, and the number to quote is the cross-validated one.",
        "",
    ]
    lines += _confusion_block(in_matrix)
    lines += [
        "",
        "## Misroutes (cross-validated) — honest error analysis",
        "",
    ]
    misroutes = [
        (r, p) for r, p in zip(records, cv_preds, strict=True) if p.value != r.expected_decision
    ]
    if not misroutes:
        lines.append("None on this set.")
    else:
        lines.append("| query | expected | predicted | top1 | margin | top_id |")
        lines.append("|---|---|---|---|---|---|")
        for r, p in misroutes:
            lines.append(
                f"| {r.query} | {r.expected_decision} | {p.value} | "
                f"{r.top1} | {r.margin} | {r.top_id} |"
            )
    lines += [
        "",
        "### Why the residual errors are expected (and how the system handles them)",
        "",
        "1. **Ambiguous queries with a confident nearest neighbour.** bge-small over a 10-entry "
        "KB produces high absolute similarities, so a genuinely ambiguous query (e.g. *“Where is "
        "my data?”* → storage Q8 vs export Q9) still has a clear top-1 and a margin above "
        "`t_margin`. Cosine geometry alone cannot flag these — they need an intent/ambiguity "
        "signal beyond similarity.",
        "2. **Near-but-out-of-scope queries** (e.g. *“write a SQL query…”*, medical advice) share "
        "vocabulary with the KB and exceed `t_low`, so a pure threshold under-declines them.",
        "",
        "Both are inherent limits of *similarity-only* routing on a tiny corpus. The primary "
        "out-of-scope defenses are therefore DETERMINISTIC: the **Q10 out-of-scope exemplar** "
        "(coding intents that land on it decline) and the **low-similarity floor** `t_low`. On "
        "top, the answer path adds a best-effort **LLM grounding/scope guard** (deterministic at "
        "temperature 0) that downgrades answer→decline for build/code requests — but at 0.5B its "
        "reliability is limited (it catches e.g. *“write a Python script…”* and misses e.g. "
        "*“write a SQL query…”*), so a code request in the ambiguous band degrades to a "
        "clarifying question, never a wrong answer. This harness scores the deterministic router "
        "*in isolation* on purpose — to keep the calibration story honest.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _matrix_rows(matrix: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    """Confusion matrix as JSON-friendly rows, each carrying its per-branch recall."""
    rows: list[dict[str, object]] = []
    for b in BRANCHES:
        row = matrix[b.value]
        total = sum(row.values())
        rows.append(
            {
                "expected": b.value,
                "answer": row["answer"],
                "clarify": row["clarify"],
                "decline": row["decline"],
                "total": total,
                "recall": (row[b.value] / total) if total else 0.0,
            }
        )
    return rows


def results_dict(
    records: list[Record],
    shipped: Settings,
    shipped_label: str,
    cv_preds: list[Decision],
    fold_settings: list[Settings],
    k: int,
) -> dict[str, object]:
    """Structured mirror of the Markdown report — same numbers, machine-readable.

    Consumed by the admin dashboard's Evaluation page so the UI never re-runs the
    harness; both outputs derive from the identical (records, cv_preds, shipped)."""
    cv_matrix = confusion(records, cv_preds)
    in_preds = _predictions(records, shipped)
    in_matrix = confusion(records, in_preds)
    misroutes = [
        {
            "query": r.query,
            "expected": r.expected_decision,
            "predicted": p.value,
            "top1": r.top1,
            "margin": r.margin,
            "top_id": r.top_id,
        }
        for r, p in zip(records, cv_preds, strict=True)
        if p.value != r.expected_decision
    ]
    return {
        "embed_model": shipped.embed_model,
        "top_k": shipped.top_k,
        "retrieval": retrieval_metrics(records),
        "routing": {
            "cv": {
                "folds": k,
                "n": len(records),
                "accuracy": accuracy(records, cv_preds),
                "confusion": _matrix_rows(cv_matrix),
            },
            "in_sample": {
                "label": shipped_label,
                "accuracy": accuracy(records, in_preds),
                "confusion": _matrix_rows(in_matrix),
                "thresholds": {
                    "t_high": shipped.t_high,
                    "t_low": shipped.t_low,
                    "t_margin": shipped.t_margin,
                },
            },
        },
        "fold_thresholds": [
            {"fold": i + 1, "t_high": s.t_high, "t_low": s.t_low, "t_margin": s.t_margin}
            for i, s in enumerate(fold_settings)
        ],
        "misroutes": misroutes,
    }


def main() -> None:
    # Windows terminals default to cp1252; the report contains arrows/curly quotes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden.yaml")
    parser.add_argument("--out", default="eval/results.md")
    parser.add_argument("--json", default="eval/results.json", help="structured report for the UI")
    parser.add_argument("--calibrate", action="store_true", help="grid-search the thresholds")
    parser.add_argument("--folds", type=int, default=5, help="k for cross-validation")
    args = parser.parse_args()

    items = yaml.safe_load(Path(args.golden).read_text(encoding="utf-8"))
    embedder = Embedder(settings.embed_model, settings.embed_cache_dir)
    retriever = InMemoryRetriever.from_kb(settings.kb_path, embedder)
    records = _embed_all(items, retriever, embedder, settings.top_k)

    # Honest, out-of-sample estimate — always computed, independent of --calibrate.
    cv_preds, fold_settings = cross_val_predict(records, settings, args.folds)

    # Shipped thresholds for the in-sample reference: refit on ALL data with --calibrate,
    # otherwise the thresholds already configured in app/config.py (or the ConfigMap).
    if args.calibrate:
        shipped = calibrate(records, settings)
        shipped_label = "grid-searched, refit on all data"
        print(
            f"Calibrated thresholds -> t_high={shipped.t_high}, "
            f"t_low={shipped.t_low}, t_margin={shipped.t_margin}"
        )
    else:
        shipped = settings
        shipped_label = "from config"

    report = render_report(records, shipped, shipped_label, cv_preds, fold_settings, args.folds)
    Path(args.out).write_text(report, encoding="utf-8")

    data = results_dict(records, shipped, shipped_label, cv_preds, fold_settings, args.folds)
    Path(args.json).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(report)
    print(f"Wrote {args.out} and {args.json}")


if __name__ == "__main__":
    main()
