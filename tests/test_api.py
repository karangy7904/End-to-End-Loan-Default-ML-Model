from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SAMPLE = {
    "income": 1303834, "age": 23, "experience": 3,
    "married_single": "single", "house_ownership": "rented",
    "car_ownership": "no", "profession": "Mechanical_engineer",
    "city": "Rewa", "state": "Madhya_Pradesh",
    "current_job_yrs": 3, "current_house_yrs": 13,
}


def test_health_and_readiness_require_loaded_model():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert client.get("/ready").status_code == 200


def test_prediction_is_valid_and_uses_optimized_threshold():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_probability"] <= 1
    assert body["threshold_used"] == 0.5780094861984253
    assert body["decision"] in {"APPROVE", "REVIEW / DECLINE"}


def test_invalid_category_is_rejected():
    response = client.post("/predict", json={**SAMPLE, "house_ownership": "mansion"})
    assert response.status_code == 422
