"""Tests for BatchCountMode state machine."""

import os
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

os.environ.setdefault("APP_ID", "batch_count")
os.environ.setdefault("DEVICE_ID", "TEST-001")

from src.gateway.modes.batch_count import BatchCountMode
from src.gateway import config


def _make_detections(label: str, count: int) -> list[dict]:
    return [{"label": label, "score": 0.9, "box": [0, 0, 10, 10]} for _ in range(count)]


def test_initial_state():
    mode = BatchCountMode()
    assert mode.get_state() == "IDLE"


@pytest.mark.asyncio
async def test_get_display_state():
    mode = BatchCountMode()
    state = await mode.get_display_state()
    assert state["app_id"] == "batch_count"
    assert state["state"] == "IDLE"
    assert state["live_count"] == 0


@pytest.mark.asyncio
async def test_inference_ignored_in_idle():
    mode = BatchCountMode()
    detections = _make_detections("bolt_m6", 5)
    await mode.on_inference_result(detections)
    assert mode.get_state() == "IDLE"


@pytest.mark.asyncio
async def test_handle_qr_starts_counting():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "sku": "BOLT-M6-30",
        "part_class": "bolt_m6",
        "target_count": 30,
        "tolerance": 0,
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        mode = BatchCountMode()
        await mode.handle_qr("BOLT-M6-30")

    assert mode.get_state() == "COUNTING"
    assert mode._sku == "BOLT-M6-30"
    assert mode._part_class == "bolt_m6"
    assert mode._target_count == 30


@pytest.mark.asyncio
async def test_handle_qr_ignored_when_not_idle():
    mode = BatchCountMode()
    mode._state = "COUNTING"
    await mode.handle_qr("SOME-SKU")
    assert mode.get_state() == "COUNTING"


@pytest.mark.asyncio
async def test_counting_window_pass():
    mode = BatchCountMode()
    mode._state = "COUNTING"
    mode._part_class = "bolt_m6"
    mode._target_count = 5
    mode._tolerance = 0
    mode._counting_start = time.monotonic() - config.COUNTING_WINDOW_SEC - 1

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections("bolt_m6", 5))

    assert mode.get_state() == "PASS"
    assert mode._result == "PASS"


@pytest.mark.asyncio
async def test_counting_window_fail():
    mode = BatchCountMode()
    mode._state = "COUNTING"
    mode._part_class = "bolt_m6"
    mode._target_count = 10
    mode._tolerance = 0
    mode._counting_start = time.monotonic() - config.COUNTING_WINDOW_SEC - 1

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections("bolt_m6", 5))

    assert mode.get_state() == "FAIL"
    assert mode._result == "FAIL"


@pytest.mark.asyncio
async def test_tolerance_boundary_pass():
    mode = BatchCountMode()
    mode._state = "COUNTING"
    mode._part_class = "bolt_m6"
    mode._target_count = 10
    mode._tolerance = 2
    mode._counting_start = time.monotonic() - config.COUNTING_WINDOW_SEC - 1

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections("bolt_m6", 8))

    assert mode.get_state() == "PASS"


@pytest.mark.asyncio
async def test_auto_reset_after_hold():
    mode = BatchCountMode()
    mode._state = "PASS"
    mode._result = "PASS"
    mode._result_time = time.monotonic() - config.RESULT_HOLD_SEC - 1

    await mode.on_inference_result([])

    assert mode.get_state() == "IDLE"
    assert mode._result is None


@pytest.mark.asyncio
async def test_display_state_during_counting():
    mode = BatchCountMode()
    mode._state = "COUNTING"
    mode._sku = "BOLT-M6-30"
    mode._live_count = 7
    mode._target_count = 30

    state = await mode.get_display_state()
    assert state["state"] == "COUNTING"
    assert state["live_count"] == 7
    assert state["target_count"] == 30
    assert state["sku"] == "BOLT-M6-30"


@pytest.mark.asyncio
async def test_write_event_payload():
    mode = BatchCountMode()
    mode._state = "COUNTING"
    mode._part_class = "bolt_m6"
    mode._target_count = 5
    mode._tolerance = 0
    mode._sku = "BOLT-M6-30"
    mode._counting_start = time.monotonic() - config.COUNTING_WINDOW_SEC - 1

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections("bolt_m6", 5))

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")

    assert payload["app_id"] == "batch_count"
    assert payload["collection"] == "inventory_batch_events"
    assert payload["data"]["result"] == "PASS"
    assert payload["data"]["detected_count"] == 5
    assert payload["data"]["target_count"] == 5
    assert payload["data"]["device_id"] == "TEST-001"
