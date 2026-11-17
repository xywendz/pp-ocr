import os

os.environ.setdefault("STRICT_MODEL_LOAD", "false")
os.environ.setdefault("LOG_LEVEL", "warning")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def test_root_page():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "图文识别工作台" in response.text


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert isinstance(response.json()["ready"], bool)


def test_openapi_schema():
    with TestClient(app) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/analyze" in response.json()["paths"]
