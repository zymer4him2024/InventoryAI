"""Tests for BundleCheckMode state machine."""

import os
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

os.environ.setdefault("APP_ID", "bundle_check")
os.environ.setdefault("DEVICE_ID", "TEST-001")

from src.gateway.modes.bundle_check import BundleCheckMode
from src.gateway import config


def _make_detections(labels: list[str]) -> list[dict]:
    return [{"label": label, "score": 0.9, "box": [0, 0, 10, 10]} for label in labels]


def test_initial_state():
    mode = BundleCheckMode()
    assert mode.get_state() == "IDLE"


@pytest.mark.asyncio
async def test_get_display_state():
    mode = BundleCheckMode()
    state = await mode.get_display_state()
    assert state["app_id"] == "bundle_check"
    assert state["state"] == "IDLE"
    assert state["checklist"] == {}


@pytest.mark.asyncio
async def test_inference_ignored_in_idle():
    mode = BundleCheckMode()
    detections = _make_detections(["bolt_m6"])
    await mode.on_inference_result(detections)
    assert mode.get_state() == "IDLE"


@pytest.mark.asyncio
async def test_handle_qr_starts_scanning():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "sku": "KIT-A",
        "required_classes": ["bolt_m6", "washer_m6", "nut_m6"],
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        mode = BundleCheckMode()
        await mode.handle_qr("KIT-A")

    assert mode.get_state() == "SCANNING"
    assert mode._required_classes == ["bolt_m6", "washer_m6", "nut_m6"]


@pytest.mark.asyncio
async def test_handle_qr_ignored_when_not_idle():
    mode = BundleCheckMode()
    mode._state = "SCANNING"
    await mode.handle_qr("SOME-SKU")
    assert mode.get_state() == "SCANNING"


@pytest.mark.asyncio
async def test_all_classes_detected_pass():
    mode = BundleCheckMode()
    mode._state = "SCANNING"
    mode._sku = "KIT-A"
    mode._required_classes = ["bolt_m6", "washer_m6", "nut_m6"]
    mode._detected_classes = set()
    mode._scan_start = time.monotonic()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(["bolt_m6", "washer_m6", "nut_m6"]))

    assert mode.get_state() == "PASS"
    assert mode._result == "PASS"


@pytest.mark.asyncio
async def test_partial_detection_stays_scanning():
    mode = BundleCheckMode()
    mode._state = "SCANNING"
    mode._required_classes = ["bolt_m6", "washer_m6", "nut_m6"]
    mode._detected_classes = set()
    mode._scan_start = time.monotonic()

    await mode.on_inference_result(_make_detections(["bolt_m6", "washer_m6"]))

    assert mode.get_state() == "SCANNING"
    assert mode._detected_classes == {"bolt_m6", "washer_m6"}


@pytest.mark.asyncio
async def test_accumulation_across_inferences():
    mode = BundleCheckMode()
    mode._state = "SCANNING"
    mode._sku = "KIT-A"
    mode._required_classes = ["bolt_m6", "washer_m6", "nut_m6"]
    mode._detected_classes = set()
    mode._scan_start = time.monotonic()

    await mode.on_inference_result(_make_detections(["bolt_m6"]))
    assert mode.get_state() == "SCANNING"

    await mode.on_inference_result(_make_detections(["washer_m6"]))
    assert mode.get_state() == "SCANNING"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(["nut_m6"]))

    assert mode.get_state() == "PASS"


@pytest.mark.asyncio
async def test_timeout_fail():
    mode = BundleCheckMode()
    mode._state = "SCANNING"
    mode._sku = "KIT-A"
    mode._required_classes = ["bolt_m6", "washer_m6", "nut_m6"]
    mode._detected_classes = {"bolt_m6"}
    mode._scan_start = time.monotonic() - config.BUNDLE_TIMEOUT_SEC - 1

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(["bolt_m6"]))

    assert mode.get_state() == "FAIL"
    assert mode._result == "FAIL"


@pytest.mark.asyncio
async def test_auto_reset_after_hold():
    mode = BundleCheckMode()
    mode._state = "PASS"
    mode._result = "PASS"
    mode._result_time = time.monotonic() - config.RESULT_HOLD_SEC - 1

    await mode.on_inference_result([])

    assert mode.get_state() == "IDLE"
    assert mode._result is None


@pytest.mark.asyncio
async def test_display_state_during_scanning():
    mode = BundleCheckMode()
    mode._state = "SCANNING"
    mode._sku = "KIT-A"
    mode._required_classes = ["bolt_m6", "washer_m6"]
    mode._detected_classes = {"bolt_m6"}

    state = await mode.get_display_state()
    assert state["state"] == "SCANNING"
    assert state["checklist"] == {"bolt_m6": True, "washer_m6": False}


@pytest.mark.asyncio
async def test_write_event_missing_classes():
    mode = BundleCheckMode()
    mode._state = "SCANNING"
    mode._sku = "KIT-A"
    mode._required_classes = ["bolt_m6", "washer_m6", "nut_m6"]
    mode._detected_classes = {"bolt_m6"}
    mode._scan_start = time.monotonic() - config.BUNDLE_TIMEOUT_SEC - 1

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(["bolt_m6"]))

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")

    assert payload["data"]["result"] == "FAIL"
    assert sorted(payload["data"]["missing_classes"]) == ["nut_m6", "washer_m6"]
    assert payload["data"]["detected_classes"] == ["bolt_m6"]
