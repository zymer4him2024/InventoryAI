"""Renderer for batch_count mode — count display + target + PASS/FAIL banner."""

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
    live = hud.live_count if hud.live_count is not None else 0
    target = hud.target_count if hud.target_count is not None else 0
    result = hud.result

    # State banner
    if state == "IDLE":
        color = C_YELLOW
        text = "IDLE — Scan QR"
    elif state == "COUNTING":
        color = C_YELLOW
        text = f"COUNTING: {sku}"
    elif result == "PASS":
        color = C_GREEN
        text = "PASS"
    elif result == "FAIL":
        color = C_RED
        text = "FAIL"
    else:
        color = C_GRAY
        text = state

    # Border
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 12)

    # State text top
    cv2.putText(canvas, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

    # SKU
    if sku:
        cv2.putText(canvas, f"SKU: {sku}", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_GRAY, 2)

    # Large count center
    count_text = str(live)
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 6.0, 8)[0]
    cx = (w - text_size[0]) // 2
    cy = (h + text_size[1]) // 2
    cv2.putText(canvas, count_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 6.0, C_WHITE, 8)

    # Target below count
    target_text = f"Target: {target}"
    ts = cv2.getTextSize(target_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    cv2.putText(canvas, target_text, ((w - ts[0]) // 2, cy + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, C_GRAY, 3)

    return canvas
