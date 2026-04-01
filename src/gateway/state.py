"""Gateway runtime state and mode registry."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import httpx

from src.gateway import config
from src.gateway.modes.base import BaseMode
from src.gateway.modes.batch_count import BatchCountMode
from src.gateway.modes.bundle_check import BundleCheckMode
from src.gateway.modes.area_monitor import AreaMonitorMode

MODE_MAP: Dict[str, Type[BaseMode]] = {
    "batch_count": BatchCountMode,
    "bundle_check": BundleCheckMode,
    "area_monitor": AreaMonitorMode,
}


@dataclass
class GatewayState:
    """Holds gateway runtime state: active mode and shared HTTP client."""

    mode: BaseMode = field(default_factory=lambda: MODE_MAP[config.APP_ID]())
    http_client: Optional[httpx.AsyncClient] = None
    # Invariant: mode_lock guards all mode method calls (on_inference_result,
    # get_display_state, handle_qr, get_state).
    mode_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def client(self) -> httpx.AsyncClient:
        if self.http_client is None or self.http_client.is_closed:
            self.http_client = httpx.AsyncClient()
        return self.http_client


gw = GatewayState()
