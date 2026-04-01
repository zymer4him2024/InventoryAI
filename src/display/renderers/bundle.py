"""Renderer for bundle_check mode — checklist with checkmarks + state banner."""

from __future__ import annotations

import cv2
import numpy as np

from src.display.schemas import HUDUpdate

C_BG = (30, 30, 30)
C_WHITE = (255, 255, 255)
C_GREEN = (0, 200, 0)
C_RED = (0, 0, 220)
C_YELLOW = (0, 220, 220)
C_GRAY = (120, 120, 120)


def render(canvas: np.ndarray, hud: HUDUpdate) -> np.ndarray:
    h, w = canvas.shape[:2]
    canvas[:] = C_BG

    state = hud.state or "IDLE"
    sku = hud.sku or ""
    checklist = hud.checklist or {}
    result = hud.result

    # State banner
    if result == "PASS":
        color = C_GREEN
        text = "PASS — All Present"
    elif result == "FAIL":
        color = C_RED
        text = "FAIL — Missing Items"
    elif state == "SCANNING":
        color = C_YELLOW
        text = f"SCANNING: {sku}"
    else:
        color = C_YELLOW
        text = "IDLE — Scan QR"

    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 12)
    cv2.putText(canvas, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

    if sku:
        cv2.putText(canvas, f"SKU: {sku}", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_GRAY, 2)

    # Checklist
    y_start = 220
    row_h = 60
    for i, (cls_name, detected) in enumerate(checklist.items()):
        y = y_start + i * row_h
        if y + row_h > h - 40:
            break
        mark = "[Y]" if detected else "[X]"
        mark_color = C_GREEN if detected else C_RED
        cv2.putText(canvas, mark, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, mark_color, 3)
        cv2.putText(canvas, cls_name, (160, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, C_WHITE, 2)

    return canvas
