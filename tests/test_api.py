from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_empty():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400


def test_predict_valid():
    response = client.post("/predict", json={"text": "I love this movie"})
    assert response.status_code in (200, 503)
