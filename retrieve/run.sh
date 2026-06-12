#!/bin/bash
# Run retrieval evaluation on videos
#
# Usage: bash run.sh [OUTPUT_DIR]
#   OUTPUT_DIR defaults to results/retrieve_<timestamp>
#
# Prerequisites: embeddings precomputed, embedding servers running

set -e

# --- Configuration (fill in before running) ---
MEMORY_DIR=""           # Root directory containing memory graphs
DATA_FILE=""            # Path to question data JSONL file
MODEL="gemini-3.1-pro-preview"                # Model name
API_KEY=""              # API key (or set LLM_API_KEY env var)
BASE_URL=""             # LLM API base URL (OpenAI-compatible)
EMBED_MODEL_PATH=""     # Embedding model path (e.g., Qwen/Qwen3-Embedding-8B)

# --- Parameters ---
MAX_ROUNDS=${MAX_ROUNDS:-12}       # Maximum agent iteration rounds per question
TOP_K=${TOP_K:-10}                 # Default top_k for search tools
MAX_CONCURRENT=${MAX_CONCURRENT:-10}  # Max concurrent agent processes

# --- Embedding servers ---
NUM_SERVERS=8
BASE_PORT=8001

# --- Directory names (under MEMORY_DIR/<video>/) ---
PHASE2_NAME="phase2_topdown"
PHASE3_NAME="phase4"

# --- Output ---
OUTPUT_DIR="${1:-results/retrieve_$(date +%Y%m%d_%H%M%S)}"

# --- Validate ---
if [ -z "$MEMORY_DIR" ] || [ -z "$DATA_FILE" ]; then
    echo "ERROR: MEMORY_DIR and DATA_FILE must be set before running."
    echo "  Edit this script or export them as environment variables."
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"
> "$OUTPUT_DIR/pids.txt"

# Build embedding server URL list
SERVER_URLS=""
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    SERVER_URLS="$SERVER_URLS http://localhost:$PORT"
done

# Discover valid videos
VIDEOS=()
for d in $(ls "$MEMORY_DIR"); do
    if [ -d "$MEMORY_DIR/$d/$PHASE2_NAME" ] && [ -d "$MEMORY_DIR/$d/$PHASE3_NAME" ]; then
        VIDEOS+=("$d")
    fi
done

echo "============================================"
echo "MemDreamer Agentic Retrieval Evaluation"
echo "Output:       $OUTPUT_DIR"
echo "Model:        $MODEL"
echo "Memory:       $MEMORY_DIR"
echo "Data:         $DATA_FILE"
echo "Videos:       ${#VIDEOS[@]}"
echo "Max rounds:   $MAX_ROUNDS / Top-K: $TOP_K"
echo "Concurrent:   $MAX_CONCURRENT"
echo "============================================"

AGENT_PIDS=()
FAILED=0
LAUNCHED=0

for VIDEO in "${VIDEOS[@]}"; do
    EMB_FILE="$MEMORY_DIR/$VIDEO/embeddings.json"

    if [ ! -f "$EMB_FILE" ]; then
        echo "  WARNING: No embeddings for $VIDEO, skipping"
        continue
    fi

    # Wait if we've hit the concurrency limit
    while [ ${#AGENT_PIDS[@]} -ge $MAX_CONCURRENT ]; do
        NEW_PIDS=()
        for PID in "${AGENT_PIDS[@]}"; do
            if kill -0 $PID 2>/dev/null; then
                NEW_PIDS+=($PID)
            fi
        done
        AGENT_PIDS=("${NEW_PIDS[@]}")
        if [ ${#AGENT_PIDS[@]} -ge $MAX_CONCURRENT ]; then
            sleep 5
        fi
    done

    # Launch this video's agent
    python -m retrieve.run \
        --video-keys "$VIDEO" \
        --memory-root "$MEMORY_DIR" \
        --data-path "$DATA_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL" \
        --api_key "$API_KEY" \
        --base-url "$BASE_URL" \
        --phase2-name "$PHASE2_NAME" \
        --phase3-name "$PHASE3_NAME" \
        --precomputed-embeddings "$EMB_FILE" \
        --embed-server-urls $SERVER_URLS \
        --max-rounds "$MAX_ROUNDS" \
        --top-k "$TOP_K" \
        > "$OUTPUT_DIR/logs/${VIDEO}.log" 2>&1 &

    AGENT_PIDS+=($!)
    echo $! >> "$OUTPUT_DIR/pids.txt"
    LAUNCHED=$((LAUNCHED + 1))
    echo "  [$LAUNCHED/${#VIDEOS[@]}] $VIDEO (PID=$!, running=${#AGENT_PIDS[@]})"
done

echo "Launched $LAUNCHED agent processes"

# Wait for all remaining processes
echo ""
echo "Waiting for all agents to finish..."

for PID in "${AGENT_PIDS[@]}"; do
    if ! wait $PID 2>/dev/null; then
        FAILED=$((FAILED + 1))
    fi
done

echo "All agents finished. Failed: $FAILED"

# Aggregate results
echo ""
echo "Aggregating results..."

python3 - "$OUTPUT_DIR" << 'PYEOF'
import json
from pathlib import Path
import sys

output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/retrieve")

total = 0
correct = 0
per_video = {}

for summary_file in sorted(output_dir.glob("*/summary.json")):
    with open(summary_file) as f:
        data = json.load(f)
    v_total = data.get("total", 0)
    v_correct = data.get("correct", 0)
    v_name = data.get("video_key", summary_file.parent.name)
    if v_name in per_video:
        per_video[v_name]["total"] += v_total
        per_video[v_name]["correct"] += v_correct
    else:
        per_video[v_name] = {"total": v_total, "correct": v_correct}
    total += v_total
    correct += v_correct

for v in per_video:
    s = per_video[v]
    s["accuracy"] = s["correct"] / s["total"] if s["total"] else 0

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total:.2%}" if total else "No results found")
print(f"\nPer-video breakdown:")
for v, stats in sorted(per_video.items()):
    print(f"  {v}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")

agg = {"total": total, "correct": correct, "accuracy": correct/total if total else 0, "per_video": per_video}
with open(output_dir / "aggregated_results.json", "w") as f:
    json.dump(agg, f, indent=2)
print(f"\nSaved to {output_dir / 'aggregated_results.json'}")
PYEOF

echo "Done!"
