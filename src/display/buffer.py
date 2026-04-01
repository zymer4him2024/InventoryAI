"""Double-buffered display state for flicker-free rendering."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from src.display.schemas import HUDUpdate


@dataclass
class DisplayState:
    hud: HUDUpdate = field(default_factory=HUDUpdate)
    # Invariant: _lock guards self.hud. Always acquire before reading or writing self.hud.
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, hud: HUDUpdate) -> None:
        with self._lock:
            self.hud = hud

    def snapshot(self) -> HUDUpdate:
        with self._lock:
            return self.hud.model_copy()


@dataclass
class SnapshotBuffer:
    """Holds the latest JPEG snapshot for the /snapshot endpoint."""

    # Invariant: _lock guards latest. Always acquire before reading or writing.
    latest: Optional[bytes] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def store(self, data: bytes) -> None:
        with self._lock:
            self.latest = data

    def read(self) -> Optional[bytes]:
        with self._lock:
            return self.latest
