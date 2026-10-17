
import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"CropWise" in response.data


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction_endpoint(client):
    response = client.post(
        "/prediction",
        json={
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 20.88,
            "humidity": 82.00,
            "ph": 6.50,
            "rainfall": 202.94,
        },
    )

    assert response.status_code == 200
    assert "recommended_crop" in response.get_json()


def test_prediction_rejects_invalid_input(client):
    response = client.post(
        "/prediction",
        json={
            "N": "not-a-number",
            "P": 42,
            "K": 43,
            "temperature": 20.88,
            "humidity": 82.00,
            "ph": 6.50,
            "rainfall": 202.94,
        },
    )

    assert response.status_code == 400
    assert "error" in response.get_json()
