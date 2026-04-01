# InventoryAI

[![CI](https://github.com/zymer4him2024/InventoryAI/actions/workflows/ci.yml/badge.svg)](https://github.com/zymer4him2024/InventoryAI/actions/workflows/ci.yml)

AI-powered parts counting and verification system for manufacturing QC, running on Raspberry Pi 5 with Hailo-8 ML accelerator.

Five FastAPI microservices orchestrate a camera-to-display pipeline: capture frames, run object detection inference, apply counting/verification logic, render a real-time HUD, and sync results to Firebase.

## Architecture

```
USB Camera                                          HDMI Display
    |                                                    ^
    v                                                    |
+----------+    +------------+    +---------+    +---------+
|  Camera  | -> | Inference  | -> | Gateway | -> | Display |
|  :8002   |    |   :8001    |    |  :8000  |    |  :8003  |
+----------+    +------------+    +---------+    +---------+
                  Hailo-8 NPU         |
                  (26 TOPS)           v
                               +--------------+
                               | Firebase Sync|
                               |    :8004     |
                               +--------------+
                                      |
                                      v
                                  Firestore
```

**Data flow:** Gateway pulls frames from Camera, sends them to Inference, processes detections through the active mode's state machine, pushes HUD updates to Display, and writes events to Firebase Sync.

## Modes

The system supports three operating modes, selected via the `APP_ID` environment variable:

| Mode | States | Purpose |
|------|--------|---------|
| `batch_count` | IDLE → COUNTING → PASS/FAIL | Count N parts of a single class against a target quantity |
| `bundle_check` | IDLE → SCANNING → PASS/FAIL | Verify all required part types are present in a kit |
| `area_monitor` | MONITORING ↔ ALERT | Continuous stock level monitoring with threshold alerts |

Jobs are triggered by QR code scan (batch_count, bundle_check) or run continuously (area_monitor). Results auto-reset after a configurable hold period.

## Quick Start

Run all 5 agents locally in simulation mode (no hardware required):

```bash
./scripts/demo.sh                  # default: batch_count
./scripts/demo.sh bundle_check     # or bundle_check
./scripts/demo.sh area_monitor     # or area_monitor
```

Opens your browser to:
- **Display HUD:** http://localhost:8003/snapshot (refresh to see updates)
- **Gateway status:** http://localhost:8080/status

Press `Ctrl+C` to stop all agents.

### Prerequisites

```bash
pip install -r requirements.txt
```

## Docker Deployment

### Production (RPi5 + Hailo-8)

```bash
cp .env.example .env
# Edit .env: set DEVICE_ID, place firebase-credentials.json, set HEF_PATH
docker compose up -d
```

### Mac Development

```bash
docker compose -f docker-compose.mac.yml up -d
```

All agents start in simulation mode: synthetic camera frames, mock inference detections, and events logged to `/app/data/events.jsonl`.

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ID` | _(required)_ | Operating mode: `batch_count`, `bundle_check`, `area_monitor` |
| `DEVICE_ID` | _(required)_ | Unique device identifier |
| `FIREBASE_SIMULATE` | `false` | Write events to local file instead of Firestore |
| `SIMULATE_CAMERA` | `false` | Generate synthetic test frames |
| `HEF_PATH` | `""` | Path to Hailo `.hef` model (empty = mock inference) |
| `COUNTING_WINDOW_SEC` | `5` | Batch count: detection averaging window |
| `COUNT_TOLERANCE` | `0` | Batch count: allowed deviation from target |
| `BUNDLE_TIMEOUT_SEC` | `10` | Bundle check: max time to detect all parts |
| `LOW_STOCK_THRESHOLD` | `5` | Area monitor: low stock alert threshold |
| `HIGH_STOCK_THRESHOLD` | `50` | Area monitor: high stock alert threshold |
| `INFERENCE_INTERVAL_SEC` | `0.5` | Gateway: inference loop interval |

## API Reference

### Gateway (:8000)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/job` | Start a job — `{"sku": "BOLT-100"}` |
| GET | `/status` | Current mode state and display data |
| GET | `/health` | Health check |

### Inference (:8001)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/inference` | Run detection on uploaded JPEG image |
| GET | `/health` | Health check (includes simulation status) |

### Camera (:8002)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/frame` | Latest camera frame (JPEG) |
| GET | `/health` | Health check (includes camera status) |

### Display (:8003)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/hud` | Update HUD overlay data |
| GET | `/snapshot` | Current rendered HUD frame (JPEG) |
| GET | `/health` | Health check |

### Firebase Sync (:8004)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/write` | Write event to Firestore (or local file) |
| POST | `/load_sku` | Load SKU config — `?sku=BOLT-100` |
| GET | `/health` | Health check |

## Testing

```bash
# Run all 57 tests
pytest tests/ -v

# Run only integration tests
pytest tests/test_gateway_integration.py -v

# Run only unit tests for a specific mode
pytest tests/test_batch_count.py -v
```

## Project Structure

```
src/
  gateway/          # Orchestrator — routes, state machine, background loops
    modes/          # batch_count, bundle_check, area_monitor
  inference/        # Hailo-8 inference or mock simulation
  camera/           # USB camera capture or synthetic frames
  display/          # HUD rendering (OpenCV) with mode-specific renderers
  firebase_sync/    # Firestore writes and SKU config lookups
tests/              # 57 tests — unit + integration
scripts/
  demo.sh           # Local 5-agent simulation launcher
```

## Tech Stack

- **Runtime:** Python 3.11, FastAPI, Pydantic v2, uvicorn
- **Inference:** Hailo-8 HAT+ (26 TOPS NPU) via HailoRT SDK
- **Vision:** OpenCV (capture, rendering, HUD display)
- **Backend:** Firebase Admin SDK (Firestore)
- **HTTP:** httpx (async inter-agent communication)
- **CI:** GitHub Actions (ruff + pytest)
- **Deploy:** Docker Compose with non-root containers, resource limits, capability-based security
