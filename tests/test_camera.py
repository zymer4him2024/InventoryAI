"""Tests for Camera Agent endpoints."""

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ID", "batch_count")
os.environ.setdefault("DEVICE_ID", "TEST-001")
os.environ["SIMULATE_CAMERA"] = "true"

from src.camera.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["simulate"] is True
