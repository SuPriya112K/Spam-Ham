"""
test_api.py

Integration tests for the FastAPI app in src/api.py.
Uses FastAPI's TestClient, which runs the app in-memory (no need for a live uvicorn server).

Run with: pytest tests/test_api.py -v
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture(scope="module")
def client():
    """
    Using TestClient as a context manager ensures FastAPI's lifespan
    (our model-loading startup event) actually runs before tests execute.
    Without the 'with', lifespan events are skipped and model_pipeline stays None.
    """
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_obvious_spam(client):
    response = client.post("/predict", json={"text": "WIN FREE MONEY NOW CLICK HERE LIMITED OFFER"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Spam"
    assert data["spam_probability"] > 0.5


def test_predict_obvious_ham(client):
    response = client.post("/predict", json={"text": "Hi team, confirming our meeting is at 3pm tomorrow."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Ham"
    assert data["ham_probability"] > 0.5


def test_predict_response_has_all_fields(client):
    response = client.post("/predict", json={"text": "This is a normal test message"})
    data = response.json()
    expected_fields = {"label", "confidence", "spam_probability", "ham_probability", "processing_time_ms"}
    assert expected_fields.issubset(data.keys())


def test_predict_probabilities_sum_to_one(client):
    response = client.post("/predict", json={"text": "This is a normal test message"})
    data = response.json()
    total = data["spam_probability"] + data["ham_probability"]
    assert abs(total - 1.0) < 0.001


def test_predict_rejects_empty_text(client):
    # Pydantic's min_length=1 should reject this before it even reaches our logic
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Unprocessable Entity - validation error


def test_predict_rejects_missing_text_field(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_rejects_text_over_max_length(client):
    huge_text = "a" * 10001  # one over our max_length=10000 limit
    response = client.post("/predict", json={"text": huge_text})
    assert response.status_code == 422


def test_predict_rejects_meaningless_text(client):
    # Text that becomes empty after cleaning (just a URL, nothing else)
    response = client.post("/predict", json={"text": "http://example.com"})
    assert response.status_code == 400


def test_predict_handles_special_characters(client):
    response = client.post("/predict", json={"text": "Special chars: @#$%^&*() should not crash the API"})
    assert response.status_code == 200