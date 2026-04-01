# InventoryAI — Project Instructions

## Overview

5-agent FastAPI microservice stack for AI-powered parts counting on RPi5 + Hailo-8.
Agents: gateway (:8000), inference (:8001), camera (:8002), display (:8003), firebase_sync (:8004).
Modes: `batch_count`, `bundle_check`, `area_monitor` — selected via `APP_ID` env var.

## Architecture

```
Camera (:8002) → Inference (:8001) → Gateway (:8000) → Display (:8003)
                                          ↓
                                   Firebase Sync (:8004)
```

Gateway orchestrates the pipeline: pulls frames, sends to inference, processes detections through the active mode, updates display, writes events to Firebase.

## Project Structure

```
src/
  gateway/           # Orchestrator
    main.py          # FastAPI routes only
    state.py         # GatewayState dataclass + mode registry
    loops.py         # inference_loop, qr_scan_loop
    config.py        # Env var validation
    schemas.py       # Pydantic request/response models
    modes/           # batch_count.py, bundle_check.py, area_monitor.py, base.py
  inference/         # Hailo-8 or mock inference
  camera/            # USB camera or simulated frames
  display/           # HUD rendering (OpenCV)
    renderers/       # Mode-specific renderers
    buffer.py        # DisplayState + SnapshotBuffer dataclasses
  firebase_sync/     # Firestore writes + SKU lookups
tests/               # 57 tests (unit + integration)
scripts/demo.sh      # Local 5-agent simulation launcher
```

## Running Tests

```bash
# Full suite (set env vars to avoid gateway config.validate() fatal exit)
APP_ID=batch_count SIMULATE_CAMERA=true DISPLAY_HEADLESS=true FIREBASE_SIMULATE=true \
  python3 -m pytest tests/ -v

# Do NOT set DEVICE_ID — tests use setdefault("DEVICE_ID", "TEST-001")
```

## Running Locally

```bash
./scripts/demo.sh              # batch_count mode
./scripts/demo.sh bundle_check # or bundle_check / area_monitor
```

## Key Conventions

- **Python 3.9 compat:** Local Mac runs 3.9. Use `Optional[X]`, `Dict[str, X]`, `Union[A, B]` from the `typing` module — never `X | None` or `dict[str, X]` in type annotations. Even with `from __future__ import annotations`, Pydantic evaluates these at runtime and will fail on 3.9.
- **Typed state:** All mutable state lives in dataclasses (CameraState, InferenceState, GatewayState, etc.) — no `global` variables for domain state.
- **Lock invariants:** Every `threading.Lock` / `asyncio.Lock` has a comment naming what it guards.
- **Specific exceptions:** Never `except Exception:` — always name the types (e.g., `except (httpx.RequestError, ValueError)`).
- **Response models:** Every FastAPI endpoint has `response_model=` parameter.
- **Config validation:** Each agent validates env vars at startup and exits with FATAL message on invalid config.
- **No bare except:** Never use `except:` or `except Exception:`. Always catch specific exception types.
- **No hardcoded secrets:** All secrets via `os.getenv()` at startup. Never mid-request.
- **Pydantic at boundaries:** All HTTP endpoints use Pydantic models for request/response validation.

## Docker

- 4 Dockerfiles: `Dockerfile` (gateway/camera), `Dockerfile.inference`, `Dockerfile.display`, `Dockerfile.firebase_sync`
- All containers run as non-root user 1001:1001
- All have `mem_limit` and `cpus` set
- Inference uses `cap_add: SYS_ADMIN` (not `privileged: true`) for Hailo PCIe DMA
- Never use `privileged: true` without explicit justification
- Port 8000 may be occupied on Mac — demo uses 8080 for gateway

## Hardware (RPi5)

- **SSH:** `digioptics_br001@192.168.0.4`
- **Hailo-8:** PCIe device at `/dev/hailo0`, firmware 4.20.0
- **Models:** `/usr/share/hailo-models/yolov8s_h8.hef` (COCO 80-class, 309 FPS)
- **Python:** Use venv with `--system-site-packages` to inherit hailo-platform
- **HEF_PATH env var:** Set to model path to enable real inference (empty = mock mode)

## CI

GitHub Actions: ruff lint + pytest on push/PR to main. Uses JUnit XML to work around OpenCV thread cleanup crash on CI runner.

## Coding Standards

### TIER 1 — Hard Blocks
- **No hardcoded secrets** — use `os.getenv()` at startup, validate, exit on missing
- **No bare except** — always name exception types, log with context
- **Validate all input at boundaries** — Pydantic models for every HTTP endpoint
- **No `privileged: true`** in Docker — use specific `cap_add` with justification

### TIER 2 — Architecture Rules
- **Single responsibility per file** — one-sentence module docstring, split at 400 lines
- **Typed dataclasses for state** — no module-level `global` variables
- **Lock invariants documented** — comment naming what each lock protects
- **Non-root containers** — all containers have `user: "1001:1001"`, `mem_limit`, `cpus`

### TIER 3 — Linter-Enforced
- Ruff with rules: E, F, C90, S, B, UP
- Max complexity: 12
- `# noqa` requires justification comment
