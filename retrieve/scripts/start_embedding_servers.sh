#!/bin/bash
# Start embedding servers (one per GPU)
#
# Usage: bash retrieve/scripts/start_embedding_servers.sh [NUM_SERVERS] [BASE_PORT]
#
# Prerequisites: run from MemDreamerV2 root directory

set -e

EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-Qwen/Qwen3-Embedding-8B}"
NUM_SERVERS="${1:-8}"
BASE_PORT="${2:-8001}"

echo "============================================"
echo "Embedding Server Launcher"
echo "Model:    $EMBED_MODEL_PATH"
echo "Servers:  $NUM_SERVERS"
echo "Ports:    $BASE_PORT - $((BASE_PORT + NUM_SERVERS - 1))"
echo "============================================"

PIDS=()
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    DEVICE="cuda:$i"
    python3 retrieve/scripts/embedding_server.py \
        --port $PORT \
        --device $DEVICE \
        --model-path "$EMBED_MODEL_PATH" &
    PIDS+=($!)
    echo "  Server $i: PID=$! port=$PORT device=$DEVICE"
done

# Wait for servers to be ready
echo ""
echo "Waiting for servers to be ready..."
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    for attempt in $(seq 1 60); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "  http://localhost:$PORT ready"
            break
        fi
        if [ $attempt -eq 60 ]; then
            echo "  WARNING: Server on port $PORT not ready after 120s"
        fi
        sleep 2
    done
done

echo ""
echo "All servers ready."
echo "PIDs: ${PIDS[*]}"
echo "To stop: kill ${PIDS[*]}"
