"""Unit tests for the routing policy — the single most important piece of logic.

Tested on SYNTHETIC scores so the branch boundaries are pinned down independently
of the embedding model. This is the TDD core of the agent design.
"""

from __future__ import annotations

import pytest
from app.agent import route
from app.config import Settings
from app.models import Decision

S = Settings(t_high=0.60, t_low=0.40, t_margin=0.05)


@pytest.mark.parametrize(
    ("top1", "margin", "oos", "expected"),
    [
        (0.80, 0.30, False, Decision.answer),  # confident, clear winner
        (0.62, 0.06, False, Decision.answer),  # just over both thresholds
        (0.75, 0.02, False, Decision.clarify),  # confident but near-tie => clarify
        (0.50, 0.20, False, Decision.clarify),  # in-domain but weak => clarify
        (0.39, 0.10, False, Decision.decline),  # nothing close => decline
        (0.90, 0.40, True, Decision.decline),  # out-of-scope exemplar wins => decline
        (0.60, 0.05, False, Decision.answer),  # exactly on the boundary (>=)
    ],
)
def test_route(top1: float, margin: float, oos: bool, expected: Decision) -> None:
    assert route(top1, margin, oos, S) is expected


def test_decline_takes_priority_over_oos_when_far() -> None:
    # Below t_low is declined regardless of the out-of-scope flag.
    assert route(0.10, 0.05, True, S) is Decision.decline
    assert route(0.10, 0.05, False, S) is Decision.decline
