"""Integration tests — gateway HTTP routes exercising the full request path."""

import os
import time
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("APP_ID", "batch_count")
os.environ.setdefault("DEVICE_ID", "TEST-001")
os.environ.setdefault("FIREBASE_SIMULATE", "true")
os.environ.setdefault("SIMULATE_CAMERA", "true")
os.environ.setdefault("DISPLAY_HEADLESS", "true")

import pytest
from fastapi.testclient import TestClient

from src.gateway.main import app
from src.gateway.state import gw
from src.gateway.modes.batch_count import BatchCountMode
from src.gateway.modes.bundle_check import BundleCheckMode
from src.gateway.modes.area_monitor import AreaMonitorMode


def _mock_httpx_client(response):
    """Create a patched httpx.AsyncClient context manager returning response."""
    mock_client = AsyncMock()
    mock_client.post.return_value = response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _sku_response(data, status_code=200):
    """Create a mock httpx response with sync .json() (httpx responses are sync)."""
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = str(data)
    return resp


_BOLT_SKU = {
    "sku": "BOLT-100",
    "part_class": "bolt_m6",
    "target_count": 10,
    "required_classes": ["bolt_m6"],
    "customer_id": "test",
    "tolerance": 0,
}

_KIT_SKU = {
    "sku": "KIT-A",
    "part_class": "bolt_m6",
    "target_count": 1,
    "required_classes": ["bolt_m6", "washer_m6", "nut_m6"],
    "customer_id": "test",
    "tolerance": 0,
}


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_mode():
    """Reset gateway mode to a fresh BatchCountMode before each test."""
    gw.mode = BatchCountMode()
    yield


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["app_id"] == "batch_count"
    assert "device_id" in data


# ---------------------------------------------------------------------------
# GET /status — initial state
# ---------------------------------------------------------------------------

def test_status_idle(client):
    r = client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "IDLE"
    assert data["display"]["app_id"] == "batch_count"
    assert data["display"]["live_count"] == 0
    assert data["display"]["target_count"] == 0


# ---------------------------------------------------------------------------
# POST /job — starts a counting job via handle_qr
# ---------------------------------------------------------------------------

def test_job_starts_counting(client):
    """POST /job loads SKU config and transitions mode to COUNTING."""
    mock_client = _mock_httpx_client(_sku_response(_BOLT_SKU))

    with patch("src.gateway.modes.batch_count.httpx.AsyncClient", return_value=mock_client):
        r = client.post("/job", json={"sku": "BOLT-100"})

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["sku"] == "BOLT-100"
    assert data["state"] == "COUNTING"


def test_job_then_status_shows_counting(client):
    """After /job, /status reflects the counting state."""
    sku_data = {**_BOLT_SKU, "target_count": 5}
    mock_client = _mock_httpx_client(_sku_response(sku_data))

    with patch("src.gateway.modes.batch_count.httpx.AsyncClient", return_value=mock_client):
        client.post("/job", json={"sku": "BOLT-100"})

    r = client.get("/status")
    data = r.json()
    assert data["state"] == "COUNTING"
    assert data["display"]["sku"] == "BOLT-100"
    assert data["display"]["target_count"] == 5


def test_job_rejected_when_not_idle(client):
    """Second /job is ignored while mode is COUNTING."""
    mock_client = _mock_httpx_client(_sku_response(_BOLT_SKU))

    with patch("src.gateway.modes.batch_count.httpx.AsyncClient", return_value=mock_client):
        r1 = client.post("/job", json={"sku": "BOLT-100"})
        assert r1.json()["state"] == "COUNTING"

        r2 = client.post("/job", json={"sku": "NUT-200"})
        assert r2.json()["state"] == "COUNTING"

    r = client.get("/status")
    assert r.json()["display"]["sku"] == "BOLT-100"


def test_job_invalid_sku_rejected(client):
    """Empty SKU is rejected by Pydantic validation."""
    r = client.post("/job", json={"sku": ""})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /job — SKU load failure leaves mode in IDLE
# ---------------------------------------------------------------------------

def test_job_sku_not_found_stays_idle(client):
    """If firebase returns 404 for SKU, mode stays IDLE."""
    mock_client = _mock_httpx_client(_sku_response({"error": "not found"}, status_code=404))

    with patch("src.gateway.modes.batch_count.httpx.AsyncClient", return_value=mock_client):
        r = client.post("/job", json={"sku": "NONEXISTENT"})

    assert r.json()["state"] == "IDLE"


# ---------------------------------------------------------------------------
# Full lifecycle: job -> inference results -> PASS/FAIL -> auto-reset
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_count_full_lifecycle(client):
    """Complete flow: start job, feed detections, get PASS, auto-reset."""
    mock_client = _mock_httpx_client(_sku_response({**_BOLT_SKU, "target_count": 3}))

    with patch("src.gateway.modes.batch_count.httpx.AsyncClient", return_value=mock_client):
        client.post("/job", json={"sku": "BOLT-100"})

    assert gw.mode.get_state() == "COUNTING"

    # Feed detections (3 bolt_m6 = matches target)
    detections = [{"label": "bolt_m6", "score": 0.9, "box": [0, 0, 10, 10]}] * 3
    gw.mode._counting_start = time.monotonic() - 10.0

    write_client = _mock_httpx_client(Mock(status_code=200))
    with patch("src.gateway.modes.batch_count.httpx.AsyncClient", return_value=write_client):
        await gw.mode.on_inference_result(detections)

    assert gw.mode.get_state() == "PASS"

    r = client.get("/status")
    assert r.json()["state"] == "PASS"
    assert r.json()["display"]["result"] == "PASS"

    # Auto-reset after RESULT_HOLD_SEC
    gw.mode._result_time = time.monotonic() - 10.0
    await gw.mode.on_inference_result([])
    assert gw.mode.get_state() == "IDLE"

    r = client.get("/status")
    assert r.json()["state"] == "IDLE"


@pytest.mark.asyncio
async def test_batch_count_fail(client):
    """Mismatched count produces FAIL."""
    mock_client = _mock_httpx_client(_sku_response({**_BOLT_SKU, "target_count": 10}))

    with patch("src.gateway.modes.batch_count.httpx.AsyncClient", return_value=mock_client):
        client.post("/job", json={"sku": "BOLT-100"})

    # Feed only 2 detections (target=10)
    detections = [{"label": "bolt_m6", "score": 0.9, "box": [0, 0, 10, 10]}] * 2
    gw.mode._counting_start = time.monotonic() - 10.0

    write_client = _mock_httpx_client(Mock(status_code=200))
    with patch("src.gateway.modes.batch_count.httpx.AsyncClient", return_value=write_client):
        await gw.mode.on_inference_result(detections)

    assert gw.mode.get_state() == "FAIL"

    r = client.get("/status")
    assert r.json()["display"]["result"] == "FAIL"


# ---------------------------------------------------------------------------
# Bundle check mode via /job
# ---------------------------------------------------------------------------

def test_bundle_check_job(client):
    """POST /job works with BundleCheckMode."""
    gw.mode = BundleCheckMode()
    mock_client = _mock_httpx_client(_sku_response(_KIT_SKU))

    with patch("src.gateway.modes.bundle_check.httpx.AsyncClient", return_value=mock_client):
        r = client.post("/job", json={"sku": "KIT-A"})

    assert r.json()["state"] == "SCANNING"

    r = client.get("/status")
    data = r.json()
    assert data["state"] == "SCANNING"
    assert data["display"]["checklist"] == {
        "bolt_m6": False,
        "washer_m6": False,
        "nut_m6": False,
    }


@pytest.mark.asyncio
async def test_bundle_check_pass(client):
    """All required classes detected produces PASS."""
    gw.mode = BundleCheckMode()
    mock_client = _mock_httpx_client(_sku_response(_KIT_SKU))

    with patch("src.gateway.modes.bundle_check.httpx.AsyncClient", return_value=mock_client):
        client.post("/job", json={"sku": "KIT-A"})

    detections = [
        {"label": "bolt_m6", "score": 0.9, "box": [0, 0, 10, 10]},
        {"label": "washer_m6", "score": 0.9, "box": [0, 0, 10, 10]},
        {"label": "nut_m6", "score": 0.9, "box": [0, 0, 10, 10]},
    ]

    write_client = _mock_httpx_client(Mock(status_code=200))
    with patch("src.gateway.modes.bundle_check.httpx.AsyncClient", return_value=write_client):
        await gw.mode.on_inference_result(detections)

    assert gw.mode.get_state() == "PASS"

    r = client.get("/status")
    assert r.json()["display"]["checklist"] == {
        "bolt_m6": True,
        "washer_m6": True,
        "nut_m6": True,
    }


# ---------------------------------------------------------------------------
# Area monitor mode
# ---------------------------------------------------------------------------

def test_area_monitor_status(client):
    """GET /status works with AreaMonitorMode."""
    gw.mode = AreaMonitorMode()

    r = client.get("/status")
    data = r.json()
    assert data["state"] == "MONITORING"
    assert data["display"]["total_count"] == 0
    assert data["display"]["alert"] is False


@pytest.mark.asyncio
async def test_area_monitor_alert(client):
    """AreaMonitorMode transitions to ALERT on low stock."""
    gw.mode = AreaMonitorMode()

    detections = [{"label": "part", "score": 0.9, "box": [0, 0, 10, 10]}] * 2

    write_client = _mock_httpx_client(Mock(status_code=200))
    with patch("src.gateway.modes.area_monitor.httpx.AsyncClient", return_value=write_client):
        await gw.mode.on_inference_result(detections)

    r = client.get("/status")
    data = r.json()
    assert data["state"] == "ALERT"
    assert data["display"]["alert"] is True
    assert data["display"]["total_count"] == 2
