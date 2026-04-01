"""Double-buffered display state for flicker-free rendering."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

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
