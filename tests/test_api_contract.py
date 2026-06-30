"""HTTP contract tests with a mocked agent — fast, offline, no model/Ollama.

Asserts the *contract* each branch must honour: status code, the decision trace
shape, citations, and that a decline never leaks a fabricated answer.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_answer_branch_contract(client: TestClient) -> None:
    res = client.post("/chat", json={"question": "How do I reset my password?"})
    assert res.status_code == 200
    body = res.json()
    assert body["decision"] == "answer"
    assert body["sources"][0]["id"] == "Q1"
    assert 0.0 <= body["scores"]["top1"] <= 1.0
    assert "answer" in body and body["answer"]


def test_clarify_branch_contract(client: TestClient) -> None:
    res = client.post("/chat", json={"question": "I can't access my account"})
    assert res.status_code == 200
    body = res.json()
    assert body["decision"] == "clarify"
    assert body["answer"].endswith("?")


def test_decline_branch_contract(client: TestClient) -> None:
    res = client.post("/chat", json={"question": "Write me a Python script"})
    assert res.status_code == 200
    body = res.json()
    assert body["decision"] == "decline"
    assert "scope" in body["answer"].lower()


def test_validation_rejects_empty_question(client: TestClient) -> None:
    res = client.post("/chat", json={"question": ""})
    assert res.status_code == 422


def test_healthz_is_dependency_free(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_metrics_increment(client: TestClient) -> None:
    client.post("/chat", json={"question": "How do I reset my password?"})
    text = client.get("/metrics").text
    assert "support_requests_total" in text
    assert 'support_decisions_total{decision="answer"}' in text
