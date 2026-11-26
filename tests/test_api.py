"""API tests. They replace answer() with a fake, so no Ollama or database is needed."""

from fastapi.testclient import TestClient  # calls the app in memory, no server needed

from app import main

client = TestClient(main.app)


def fake_answer(question: str) -> dict:  # same shape as app.answer.answer() returns
    return {
        "answer": "36% per annum (BSP Circular No. 1242, p. 1).",
        "refused": False,
        "reason": "ok",
        "citations": ["BSP Circular No. 1242, p. 1"],
        "tokens": {"input": 10, "output": 5},
    }


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_citations(monkeypatch):
    monkeypatch.setattr(main, "answer", fake_answer)  # swap the real pipeline for the fake, for this test only
    response = client.post("/ask", json={"question": "What is the credit card interest cap?"})
    body = response.json()
    assert response.status_code == 200
    assert body["refused"] is False
    assert body["citations"] == ["BSP Circular No. 1242, p. 1"]
    assert "tokens" not in body  # tokens go to the logs, not to the user
    assert body["latency_ms"] >= 0


def test_ask_handles_refusal_without_tokens(monkeypatch):
    refusal = {"answer": "I couldn't find this in the BSP circulars I have.", "refused": True,
               "reason": "no passages", "citations": [], "tokens": None}  # model never called
    monkeypatch.setattr(main, "answer", lambda q: refusal)
    body = client.post("/ask", json={"question": "banana bread recipe"}).json()
    assert body["refused"] is True
    assert body["reason"] == "no passages"


def test_rejects_too_short_question():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422  # min_length=3


def test_rejects_missing_question():
    assert client.post("/ask", json={}).status_code == 422
