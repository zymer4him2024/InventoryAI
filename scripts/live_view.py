"""Live camera view with Hailo-8 inference — single process, full screen HDMI."""

import sys
import time

import cv2
import httpx
import numpy as np

INFERENCE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
CAMERA_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Color palette — modern flat colors (BGR)
COLORS = [
    (255, 107, 53),   # coral
    (78, 205, 196),   # teal
    (255, 209, 102),  # gold
    (120, 111, 166),  # purple
    (0, 200, 140),    # mint
    (86, 204, 242),   # sky blue
    (255, 154, 162),  # pink
    (180, 220, 100),  # lime
    (100, 160, 255),  # blue
    (255, 183, 77),   # orange
]
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BAR_BG = (20, 20, 20)
ACCENT = (78, 205, 196)  # teal accent


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=8):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    # Straight edges
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)

    if thickness < 0:
        # Fill corners
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    else:
        # Corner arcs
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_label(img, text, org, color, scale=0.55):
    """Draw text with a filled rounded background pill."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = org
    pad_x, pad_y = 8, 5
    # Background pill
    overlay = img.copy()
    draw_rounded_rect(overlay,
                      (x, y - th - pad_y * 2),
                      (x + tw + pad_x * 2, y + baseline),
                      color, -1, radius=6)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    # Text
    cv2.putText(img, text, (x + pad_x, y - pad_y),
                font, scale, WHITE, 1, cv2.LINE_AA)


def draw_detection(img, det, color):
    """Draw a single detection: bbox with corner accents + label pill."""
    box = det.get("box", [])
    if len(box) != 4:
        return
    x, y, w, h = [int(v) for v in box]
    x2, y2 = x + w, y + h
    label = det.get("label", "?")
    score = det.get("score", 0.0)

    # Main box — thin line
    cv2.rectangle(img, (x, y), (x2, y2), color, 2, cv2.LINE_AA)

    # Corner accents — thick short lines at each corner
    corner_len = min(25, w // 4, h // 4)
    t = 3  # corner thickness
    # Top-left
    cv2.line(img, (x, y), (x + corner_len, y), color, t, cv2.LINE_AA)
    cv2.line(img, (x, y), (x, y + corner_len), color, t, cv2.LINE_AA)
    # Top-right
    cv2.line(img, (x2, y), (x2 - corner_len, y), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y), (x2, y + corner_len), color, t, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img, (x, y2), (x + corner_len, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x, y2), (x, y2 - corner_len), color, t, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, t, cv2.LINE_AA)

    # Label pill above box
    text = f"{label}  {int(score * 100)}%"
    draw_label(img, text, (x, y - 4), color)


def draw_status_bar(img, fps, inf_ms, det_count):
    """Draw modern translucent status bar at top."""
    h, w = img.shape[:2]
    bar_h = 48

    # Translucent bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), BAR_BG, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Accent line at bottom of bar
    cv2.line(img, (0, bar_h), (w, bar_h), ACCENT, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_DUPLEX
    y_text = 33

    # Logo / title
    cv2.putText(img, "InventoryAI", (16, y_text),
                font, 0.7, ACCENT, 1, cv2.LINE_AA)

    # Stats — right aligned
    stats = f"FPS {fps:.0f}   Hailo {inf_ms:.0f}ms   Objects {det_count}"
    (sw, _), _ = cv2.getTextSize(stats, font, 0.55, 1)
    cv2.putText(img, stats, (w - sw - 16, y_text),
                font, 0.55, WHITE, 1, cv2.LINE_AA)


# --- Main ---

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print(f"FATAL: Cannot open camera {CAMERA_INDEX}", file=sys.stderr)
    sys.exit(1)

print(f"Camera {CAMERA_INDEX} opened. Inference: {INFERENCE_URL}")
print("Press 'q' to quit.")

cv2.namedWindow("InventoryAI Live", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("InventoryAI Live", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

client = httpx.Client(timeout=5.0)
fps_time = time.monotonic()
frame_count = 0
fps_display = 0.0
label_colors = {}

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    # Encode frame as JPEG for inference
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # Send to inference agent
    detections = []
    inf_ms = 0.0
    try:
        resp = client.post(
            f"{INFERENCE_URL}/inference",
            files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                detections = data.get("detections", [])
                inf_ms = data.get("inference_ms", 0.0)
    except httpx.RequestError:
        pass

    # Draw detections with per-class colors
    for det in detections:
        label = det.get("label", "?")
        if label not in label_colors:
            label_colors[label] = COLORS[len(label_colors) % len(COLORS)]
        draw_detection(frame, det, label_colors[label])

    # FPS counter
    frame_count += 1
    now = time.monotonic()
    if now - fps_time >= 1.0:
        fps_display = frame_count / (now - fps_time)
        frame_count = 0
        fps_time = now

    # Status bar
    draw_status_bar(frame, fps_display, inf_ms, len(detections))

    cv2.imshow("InventoryAI Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
