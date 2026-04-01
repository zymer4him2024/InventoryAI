"""Renderer for area_monitor mode — large count, delta, timestamp."""

from __future__ import annotations

import cv2
import numpy as np

from src.display.schemas import HUDUpdate

C_BG = (30, 30, 30)
C_WHITE = (255, 255, 255)
C_GREEN = (0, 200, 0)
C_RED = (0, 0, 220)
C_GRAY = (120, 120, 120)


def render(canvas: np.ndarray, hud: HUDUpdate) -> np.ndarray:
    h, w = canvas.shape[:2]
    canvas[:] = C_BG

    total = hud.total_count if hud.total_count is not None else 0
    delta = hud.delta if hud.delta is not None else 0
    alert = hud.alert or False
    location = hud.location or ""
    last_updated = hud.last_updated or ""

    color = C_RED if alert else C_GREEN
    state_text = "ALERT" if alert else "MONITORING"

    # Border
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 12)

    # State + location top
    cv2.putText(canvas, state_text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    if location:
        cv2.putText(canvas, location, (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_GRAY, 2)

    # Large count center
    count_text = str(total)
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 8.0, 10)[0]
    cx = (w - text_size[0]) // 2
    cy = (h + text_size[1]) // 2 - 30
    cv2.putText(canvas, count_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 8.0, color, 10)

    # Delta below count
    if delta != 0:
        sign = "+" if delta > 0 else ""
        delta_text = f"{sign}{delta}"
        delta_color = C_GREEN if delta > 0 else C_RED
        ds = cv2.getTextSize(delta_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
        cv2.putText(canvas, delta_text, ((w - ds[0]) // 2, cy + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, delta_color, 4)

    # Timestamp bottom
    if last_updated:
        cv2.putText(canvas, f"Updated: {last_updated}", (40, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GRAY, 2)

    return canvas
