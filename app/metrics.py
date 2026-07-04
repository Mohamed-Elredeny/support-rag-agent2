"""In-process metrics in Prometheus text format (no extra dependencies).

Exposes request volume, the distribution of agentic decisions, and average
latency — enough to see the agent actually branching across answer/clarify/decline.
"""

from __future__ import annotations

import threading

from app.models import Decision


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._decisions = {d.value: 0 for d in Decision}
        self._latency_sum_ms = 0.0

    def record(self, decision: Decision, latency_ms: int) -> None:
        with self._lock:
            self._requests += 1
            self._decisions[decision.value] += 1
            self._latency_sum_ms += latency_ms

    def render(self) -> str:
        with self._lock:
            avg = self._latency_sum_ms / self._requests if self._requests else 0.0
            lines = [
                "# HELP support_requests_total Total /chat requests served.",
                "# TYPE support_requests_total counter",
                f"support_requests_total {self._requests}",
                "# HELP support_decisions_total Agentic decisions by branch.",
                "# TYPE support_decisions_total counter",
            ]
            lines += [
                f'support_decisions_total{{decision="{name}"}} {count}'
                for name, count in self._decisions.items()
            ]
            lines += [
                "# HELP support_request_latency_ms_avg Average request latency.",
                "# TYPE support_request_latency_ms_avg gauge",
                f"support_request_latency_ms_avg {avg:.2f}",
            ]
            return "\n".join(lines) + "\n"
