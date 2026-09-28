from fastapi.testclient import TestClient

from api.main import app
from src.schema import EmployeeFeatures

SAMPLE = EmployeeFeatures.model_config["json_schema_extra"]["example"]


def test_health():
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}


def test_predict_returns_probability_and_flag():
    with TestClient(app) as client:
        res = client.post("/api/predict", json=SAMPLE)
        assert res.status_code == 200
        body = res.json()
        assert 0.0 <= body["attrition_probability"] <= 1.0
        assert isinstance(body["attrition_predite"], bool)


def test_predict_rejects_invalid_categorical_value():
    with TestClient(app) as client:
        bad_sample = {**SAMPLE, "Gender": "Autre"}
        res = client.post("/api/predict", json=bad_sample)
        assert res.status_code == 422
