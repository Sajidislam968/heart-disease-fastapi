from fastapi.testclient import TestClient

from app import main


class FakeModel:
    def predict(self, frame):
        return [1]

    def predict_proba(self, frame):
        return [[0.15, 0.85]]


def payload():
    return {
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
    }


def test_health_and_info(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: FakeModel())
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/info").json()
    assert info["feature_count"] == 13
    assert info["model_loaded"] is True


def test_predict_returns_boolean_and_probability(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: FakeModel())
    client = TestClient(main.app)
    response = client.post("/predict", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["heart_disease"] is True
    assert body["prediction"] == "presence"
    assert body["probability"] == 0.85


def test_predict_validates_input(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: FakeModel())
    client = TestClient(main.app)
    bad_payload = payload()
    bad_payload["cp"] = 9
    assert client.post("/predict", json=bad_payload).status_code == 422
