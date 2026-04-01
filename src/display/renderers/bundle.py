"""Renderer for bundle_check mode — camera feed + checklist overlay."""

from __future__ import annotations

import base64

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

    # Decode camera frame as background
    if hud.frame_b64:
        raw = base64.b64decode(hud.frame_b64)
        nparr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            canvas = cv2.resize(frame, (w, h))
        else:
            canvas[:] = C_BG
    else:
        canvas[:] = C_BG

    # Draw detection bounding boxes
    if hud.detections:
        for det in hud.detections:
            if len(det.box) == 4:
                x, y_pos, bw, bh = [int(v) for v in det.box]
                cv2.rectangle(canvas, (x, y_pos), (x + bw, y_pos + bh), C_GREEN, 2)
                label = f"{det.label} {det.score:.2f}"
                cv2.putText(canvas, label, (x, y_pos - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREEN, 2)

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

    # Semi-transparent top bar
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    cv2.putText(canvas, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    if sku:
        cv2.putText(canvas, f"SKU: {sku}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GRAY, 2)

    # Checklist panel (right side, semi-transparent)
    if checklist:
        panel_w = 350
        panel_x = w - panel_w - 10
        panel_h = len(checklist) * 50 + 20
        overlay2 = canvas.copy()
        cv2.rectangle(overlay2, (panel_x, 130), (w - 10, 130 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, canvas, 0.4, 0, canvas)

        for i, (cls_name, detected) in enumerate(checklist.items()):
            y_row = 170 + i * 50
            mark = "[Y]" if detected else "[X]"
            mark_color = C_GREEN if detected else C_RED
            cv2.putText(canvas, mark, (panel_x + 10, y_row), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mark_color, 2)
            cv2.putText(canvas, cls_name, (panel_x + 80, y_row), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_WHITE, 2)

    # Border
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 4)

    return canvas
