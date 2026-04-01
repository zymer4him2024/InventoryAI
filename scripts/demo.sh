#!/usr/bin/env bash
# Launch all 5 InventoryAI agents locally in simulation mode.
# Opens browser to the display snapshot and gateway status.
#
# Usage:  ./scripts/demo.sh [batch_count|bundle_check|area_monitor]
# Stop:   Ctrl+C (kills all background agents)

set -euo pipefail

MODE="${1:-batch_count}"

export APP_ID="$MODE"
export DEVICE_ID="DEMO-001"
export SIMULATE_CAMERA="true"
export DISPLAY_HEADLESS="true"
export FIREBASE_SIMULATE="true"
export EVENTS_LOG_PATH="/tmp/inventoryai_events.jsonl"

# Point gateway at localhost agents
export CAMERA_URL="http://localhost:8002"
export INFERENCE_URL="http://localhost:8001"
export DISPLAY_URL="http://localhost:8003"
export FIREBASE_SYNC_URL="http://localhost:8004"

# Faster inference loop for demo
export INFERENCE_INTERVAL_SEC="1.0"

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down agents..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

echo "=== InventoryAI Demo (mode=$MODE) ==="
echo ""

# Start agents in order: dependencies first
echo "Starting inference agent on :8001..."
python3 -m uvicorn src.inference.main:app --port 8001 --log-level warning &
PIDS+=($!)

echo "Starting camera agent on :8002..."
python3 -m uvicorn src.camera.main:app --port 8002 --log-level warning &
PIDS+=($!)

echo "Starting display agent on :8003..."
python3 -m uvicorn src.display.main:app --port 8003 --log-level warning &
PIDS+=($!)

echo "Starting firebase_sync agent on :8004..."
python3 -m uvicorn src.firebase_sync.main:app --port 8004 --log-level warning &
PIDS+=($!)

# Wait for agents to be ready
echo "Waiting for agents to start..."
sleep 3

GATEWAY_PORT="${GATEWAY_PORT:-8080}"
echo "Starting gateway on :${GATEWAY_PORT}..."
python3 -m uvicorn src.gateway.main:app --port "$GATEWAY_PORT" --log-level info &
PIDS+=($!)

sleep 2

echo ""
echo "=== All agents running ==="
echo ""
echo "  Camera frame:     http://localhost:8002/frame"
echo "  Inference health: http://localhost:8001/health"
echo "  Display snapshot: http://localhost:8003/snapshot"
echo "  Firebase health:  http://localhost:8004/health"
echo "  Gateway status:   http://localhost:${GATEWAY_PORT}/status"
echo "  Gateway health:   http://localhost:${GATEWAY_PORT}/health"
echo ""
echo "  Auto-refresh HUD: http://localhost:8003/snapshot (refresh to see updates)"
echo ""

# Open browser
if command -v open &>/dev/null; then
    open "http://localhost:8003/snapshot"
    open "http://localhost:${GATEWAY_PORT}/status"
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:8003/snapshot"
    xdg-open "http://localhost:${GATEWAY_PORT}/status"
fi

echo "Press Ctrl+C to stop all agents."
echo ""

# Keep running until interrupted
wait
