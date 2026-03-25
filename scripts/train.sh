#!/bin/bash

# --- 1. Environment & Path Setup ---
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Setup Go environment for Chromebook performance
export GOGC=20
export GOMAXPROCS=2
export GOEXPERIMENT=simd

# Ensure critical directories exist
mkdir -p logs
mkdir -p research_logs
mkdir -p data/models/checkpoints
mkdir -p data/models/gob_models
touch logs/training.csv

# --- 2. Housekeeping ---
rm -f data/models/gob_models/seq2seq_output_vocab.gob
rm -f data/models/gob_models/moe_classification_model.gob
rm -f data/models/gob_models/moe_classification_model_best.gob

# --- 3. Audit & Trend Analysis ---
echo "🔍 Scanning Gollemer Evolution & Commitment Trends..."
printf "| %-15s | %-7s | %-12s | %-7s |\n" "File" "Steps" "Commitment" "Trend"
echo "|-----------------|---------|--------------|---------|"

PREV_IQ=0
BEST_SCORE=0
BEST_FILE=""
CHECKPOINT_DIR="data/models/checkpoints"

for f in $(ls $CHECKPOINT_DIR/*.gob 2>/dev/null | sort); do
    go run cmd/inspect/inspect_model.go --export "$f" > /dev/null
    JSON_FILE="${f}.json"
    
    if [ -f "$JSON_FILE" ]; then
        STEPS=$(jq '.total_steps' "$JSON_FILE")
        IQ=$(jq '.commitment_pct' "$JSON_FILE")
        
        if (( $(echo "$IQ > $PREV_IQ" | bc -l) )); then
            TREND="🚀 ↑"
        elif (( $(echo "$IQ < $PREV_IQ" | bc -l) )); then
            TREND="⚠️  ↓"
        else
            TREND="↔️  -"
        fi

        if (( $(echo "$IQ > $BEST_SCORE" | bc -l) )); then
            BEST_SCORE=$IQ
            BEST_FILE=$f
        fi

        printf "| %-15s | %-7s | %-12s | %-7s |\n" "$(basename "$f")" "$STEPS" "${IQ}%" "$TREND"
        PREV_IQ=$IQ
        mv "$JSON_FILE" "./research_logs/"
    fi
done

# --- 4. Intelligent Disk Pruning ---
LATEST_FILE=$(ls -t $CHECKPOINT_DIR/*.gob 2>/dev/null | head -n 1)

if [ -n "$BEST_FILE" ]; then
    echo -e "\n🏆 Top Performer: $(basename "$BEST_FILE") (${BEST_SCORE}%)"
    for f in $CHECKPOINT_DIR/*.gob; do
        if [ "$f" != "$BEST_FILE" ] && [ "$f" != "$LATEST_FILE" ]; then
            rm "$f"
        fi
    done
fi

# --- 5. The Gatekeeper: Promotion & ChromeOS Notification ---
THRESHOLD=2.0
if [ -n "$BEST_FILE" ]; then
    if (( $(echo "$BEST_SCORE >= $THRESHOLD" | bc -l) )); then
        echo "🎖️  Threshold Met ($BEST_SCORE% >= $THRESHOLD%). Promoting..."
        cp "$BEST_FILE" "data/models/gob_models/moe_classification_model_best.gob"
        ln -sf "$(pwd)/data/models/gob_models/moe_classification_model_best.gob" "data/models/gob_models/moe_active.gob"
        
        # 🔔 ChromeOS Desktop Notification
        if command -v notify-send >/dev/null 2>&1; then
            notify-send "Gollemer AI" "Best Model Promoted! IQ: ${BEST_SCORE}%" --icon=utilities-terminal
        fi
    else
        echo "🛑  Threshold NOT Met ($BEST_SCORE%)."
    fi
fi

# --- 6. Launch Training ---
echo -e "\n🚀 Starting Gollemer Training..."
GOEXPERIMENT=simd go run cmd/gollemer/main.go \
    -train-chat \
    -rebalance \
    -overfit \
    -auto-heal \
    -wd 0.01 \
    -lr 0.001

# --- 7. Completion Notification ---
echo -e "\a"
if command -v notify-send >/dev/null 2>&1; then
    notify-send "Gollemer AI" "Training Session Complete." --urgency=critical
fi