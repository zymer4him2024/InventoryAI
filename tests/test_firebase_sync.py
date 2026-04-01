"""Tests for Firebase Sync Agent endpoints."""

import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ID", "batch_count")
os.environ.setdefault("DEVICE_ID", "TEST-001")
os.environ["FIREBASE_SIMULATE"] = "true"


@pytest.fixture
def client(tmp_path):
    os.environ["EVENTS_LOG_PATH"] = str(tmp_path / "events.jsonl")
    from src.firebase_sync.main import app
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["simulation"] is True


def test_write_simulation_mode(client):
    payload = {
        "app_id": "batch_count",
        "collection": "inventory_batch_events",
        "data": {"device_id": "TEST-001", "result": "PASS"},
    }
    resp = client.post("/write", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "simulated"
    assert data["collection"] == "inventory_batch_events"

    from src.firebase_sync.main import EVENTS_LOG_PATH
    assert EVENTS_LOG_PATH.exists()
    lines = EVENTS_LOG_PATH.read_text().strip().split("\n")
    last_line = json.loads(lines[-1])
    assert last_line["collection"] == "inventory_batch_events"
    assert last_line["result"] == "PASS"
    assert "written_at" in last_line


def test_load_sku_simulation(client):
    resp = client.post("/load_sku?sku=BOLT-M6-30")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sku"] == "BOLT-M6-30"
    assert data["part_class"] == "bolt_m6"
    assert data["target_count"] == 10
    assert "required_classes" in data
    assert data["tolerance"] == 0


def test_load_sku_empty_rejected(client):
    resp = client.post("/load_sku?sku=")
    assert resp.status_code == 422
