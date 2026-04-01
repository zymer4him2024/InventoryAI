"""Tests for AreaMonitorMode state machine."""

import os
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

os.environ.setdefault("APP_ID", "area_monitor")
os.environ.setdefault("DEVICE_ID", "TEST-001")

from src.gateway.modes.area_monitor import AreaMonitorMode
from src.gateway import config


def _make_detections(count: int) -> list[dict]:
    return [{"label": "bolt_m6", "score": 0.9, "box": [0, 0, 10, 10]} for _ in range(count)]


def test_initial_state():
    mode = AreaMonitorMode()
    assert mode.get_state() == "MONITORING"


@pytest.mark.asyncio
async def test_get_display_state():
    mode = AreaMonitorMode()
    state = await mode.get_display_state()
    assert state["app_id"] == "area_monitor"
    assert state["state"] == "MONITORING"
    assert state["total_count"] == 0
    assert state["alert"] is False


@pytest.mark.asyncio
async def test_qr_ignored():
    mode = AreaMonitorMode()
    await mode.handle_qr("SOME-SKU")
    assert mode.get_state() == "MONITORING"


@pytest.mark.asyncio
async def test_count_updates():
    mode = AreaMonitorMode()
    mode._last_snapshot_time = time.monotonic()

    await mode.on_inference_result(_make_detections(12))
    assert mode._total_count == 12


@pytest.mark.asyncio
async def test_low_stock_alert():
    mode = AreaMonitorMode()
    mode._last_snapshot_time = time.monotonic()
    mode._alert = False

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(config.LOW_STOCK_THRESHOLD))

    assert mode.get_state() == "ALERT"
    assert mode._alert is True


@pytest.mark.asyncio
async def test_high_stock_alert():
    mode = AreaMonitorMode()
    mode._last_snapshot_time = time.monotonic()
    mode._alert = False

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(config.HIGH_STOCK_THRESHOLD))

    assert mode.get_state() == "ALERT"
    assert mode._alert is True


@pytest.mark.asyncio
async def test_normal_count_monitoring():
    mode = AreaMonitorMode()
    mode._last_snapshot_time = time.monotonic()
    normal_count = (config.LOW_STOCK_THRESHOLD + config.HIGH_STOCK_THRESHOLD) // 2

    await mode.on_inference_result(_make_detections(normal_count))
    assert mode.get_state() == "MONITORING"
    assert mode._alert is False


@pytest.mark.asyncio
async def test_return_to_monitoring():
    mode = AreaMonitorMode()
    mode._state = "ALERT"
    mode._alert = True
    mode._last_snapshot_time = time.monotonic()
    normal_count = (config.LOW_STOCK_THRESHOLD + config.HIGH_STOCK_THRESHOLD) // 2

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(normal_count))

    assert mode.get_state() == "MONITORING"
    assert mode._alert is False


@pytest.mark.asyncio
async def test_threshold_crossing_writes_snapshot():
    mode = AreaMonitorMode()
    mode._last_snapshot_time = time.monotonic()
    mode._alert = False

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(config.LOW_STOCK_THRESHOLD))

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")

    assert payload["collection"] == "inventory_area_snapshots"
    assert payload["data"]["state"] == "ALERT"
    assert payload["data"]["count"] == config.LOW_STOCK_THRESHOLD


@pytest.mark.asyncio
async def test_periodic_snapshot():
    mode = AreaMonitorMode()
    mode._last_snapshot_time = time.monotonic() - config.AREA_SNAPSHOT_INTERVAL_SEC - 1
    normal_count = (config.LOW_STOCK_THRESHOLD + config.HIGH_STOCK_THRESHOLD) // 2

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(normal_count))

        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_delta_calculation():
    mode = AreaMonitorMode()
    mode._last_snapshot_count = 10
    mode._last_snapshot_time = time.monotonic()
    mode._total_count = 0

    await mode.on_inference_result(_make_detections(15))

    state = await mode.get_display_state()
    assert state["delta"] == 5


@pytest.mark.asyncio
async def test_snapshot_event_payload():
    mode = AreaMonitorMode()
    mode._last_snapshot_time = time.monotonic()
    mode._last_snapshot_count = 10
    mode._alert = False

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        await mode.on_inference_result(_make_detections(config.LOW_STOCK_THRESHOLD))

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")

    data = payload["data"]
    assert data["device_id"] == "TEST-001"
    assert data["app_id"] == "area_monitor"
    assert data["delta"] == config.LOW_STOCK_THRESHOLD - 10
    assert "timestamp" in data
