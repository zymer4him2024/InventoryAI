"""Tests for Inference Agent endpoints."""

import os

import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ID", "batch_count")
os.environ.setdefault("DEVICE_ID", "TEST-001")

from src.inference.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_jpeg() -> bytes:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["simulation"] is True
    assert data["model"] == "mock"


def test_mock_inference_returns_detections(client, test_jpeg):
    resp = client.post("/inference", files={"image": ("test.jpg", test_jpeg, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["inference_ms"] > 0
    assert isinstance(data["detections"], list)
    assert len(data["detections"]) > 0


def test_mock_inference_detection_shape(client, test_jpeg):
    resp = client.post("/inference", files={"image": ("test.jpg", test_jpeg, "image/jpeg")})
    data = resp.json()
    det = data["detections"][0]
    assert "label" in det
    assert "score" in det
    assert "box" in det
    assert 0.0 <= det["score"] <= 1.0
    assert len(det["box"]) == 4


def test_mock_inference_uses_configured_labels(client, test_jpeg):
    resp = client.post("/inference", files={"image": ("test.jpg", test_jpeg, "image/jpeg")})
    data = resp.json()
    labels = {d["label"] for d in data["detections"]}
    expected = {"bolt_m6", "washer_m6", "nut_m6"}
    assert labels.issubset(expected)
