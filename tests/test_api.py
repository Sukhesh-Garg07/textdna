from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "TextDNA" in response.text


def test_analyze_endpoint():
    response = client.post(
        "/analyze",
        json={"text": "Paris is the capital of France. It is known for the Eiffel Tower."},
    )
    assert response.status_code == 200
    body = response.json()
    assert "textdna_score" in body
    assert body["text_stats"]["sentences"] == 2


def test_analyze_rejects_empty_text():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_rejects_blank_text():
    response = client.post("/analyze", json={"text": "   "})
    assert response.status_code == 422


def test_analyze_rejects_missing_field():
    response = client.post("/analyze", json={})
    assert response.status_code == 422
