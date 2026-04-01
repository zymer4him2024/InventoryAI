"""Live camera view with Hailo-8 inference — single process, full screen HDMI."""

import sys
import time

import cv2
import httpx
import numpy as np

INFERENCE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
CAMERA_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CONF_THRESHOLD = 0.35

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

GREEN = (0, 220, 0)
YELLOW = (0, 220, 220)
WHITE = (255, 255, 255)

cap = cv2.VideoCapture(CAMERA_INDEX)
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

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    # Encode frame as JPEG for inference
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpg_bytes = buf.tobytes()

    # Send to inference agent
    detections = []
    inf_ms = 0.0
    try:
        resp = client.post(
            f"{INFERENCE_URL}/inference",
            files={"image": ("frame.jpg", jpg_bytes, "image/jpeg")},
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                detections = data.get("detections", [])
                inf_ms = data.get("inference_ms", 0.0)
    except httpx.RequestError:
        pass

    # Draw bounding boxes
    for det in detections:
        box = det.get("box", [])
        if len(box) != 4:
            continue
        x, y, w, h = [int(v) for v in box]
        label = det.get("label", "?")
        score = det.get("score", 0.0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        text = f"{label} {score:.2f}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

    # FPS counter
    frame_count += 1
    now = time.monotonic()
    if now - fps_time >= 1.0:
        fps_display = frame_count / (now - fps_time)
        frame_count = 0
        fps_time = now

    # Status bar
    h_frame, w_frame = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_frame, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"FPS: {fps_display:.1f}  Inference: {inf_ms:.0f}ms  Detections: {len(detections)}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    cv2.imshow("InventoryAI Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
