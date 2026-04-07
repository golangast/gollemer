#!/bin/bash

## --- 1. Environment & Path Setup ---
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Detect OS
IS_WINDOWS=false
case "$(uname -s)" in
    CYGWIN*|MINGW*|MSYS*) IS_WINDOWS=true;;
esac

# Setup Go environment for pure Go performance
export GO_CMD="go"
export CGO_ENABLED=0
export GOEXPERIMENT=simd


# Use system 'go' if the hardcoded path doesn't exist
GO_CMD="/usr/local/go/bin/go"
if ! command -v "$GO_CMD" >/dev/null 2>&1; then
    GO_CMD="go"
fi
export GO_CMD
export GOEXPERIMENT=simd

# Handle Extensions
BIN_EXT=""
if [ "$IS_WINDOWS" = true ]; then
    BIN_EXT=".exe"
fi

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


W2V_PATH="data/models/gob_models/word2vec_model.gob"
if [ ! -f "$W2V_PATH" ]; then
    echo "⚠️  Word2Vec Dictionary missing. Regenerating from data..."
    GOEXPERIMENT=simd "$GO_CMD" run cmd/gollemer/main.go -train-word2vec
fi

# --- 3. Audit & Trend Analysis ---
HAS_JQ=false
if command -v jq >/dev/null 2>&1; then HAS_JQ=true; fi
HAS_BC=false
if command -v bc >/dev/null 2>&1; then HAS_BC=true; fi

echo "🔍 Scanning Gollemer Evolution & Commitment Trends..."
printf "| %-15s | %-7s | %-12s | %-7s |\n" "File" "Steps" "Commitment" "Trend"
echo "|-----------------|---------|--------------|---------|"

PREV_IQ=0
BEST_SCORE=0
BEST_FILE=""
CHECKPOINT_DIR="data/models/checkpoints"

for f in $(ls $CHECKPOINT_DIR/*.gob 2>/dev/null | sort); do
    "$GO_CMD" run cmd/inspect/inspect_model.go --export "$f" > /dev/null
    JSON_FILE="${f}.json"
    
    if [ -f "$JSON_FILE" ]; then
        # Use sed as fallback if jq is missing
        if [ "$HAS_JQ" = true ]; then
            STEPS=$(jq '.StepCount' "$JSON_FILE" 2>/dev/null)
            IQ_RAW=$(jq '.Commitment' "$JSON_FILE" 2>/dev/null)
        else
            STEPS=$(grep -o '"StepCount": *[0-9]*' "$JSON_FILE" | cut -d':' -f2)
            IQ_RAW=$(grep -o '"Commitment": *[0-9.]*' "$JSON_FILE" | cut -d':' -f2)
        fi
        
        IQ_RAW=$(echo "$IQ_RAW" | sed 's/[^0-9.]//g')
        STEPS=$(echo "$STEPS" | sed 's/[^0-9]//g')
        [ -z "$IQ_RAW" ] && IQ_RAW="0.0"
        [ -z "$STEPS" ] && STEPS="0"
        
        # Convert to percentage (IQ = IQ_RAW * 100)
        IQ=$(awk "BEGIN {printf \"%.2f\", $IQ_RAW * 100}")

        TREND="↔️  -"
        if [ "$HAS_BC" = true ]; then
            if (( $(echo "$IQ > $PREV_IQ" | bc -l) )); then TREND="🚀 ↑"
            elif (( $(echo "$IQ < $PREV_IQ" | bc -l) )); then TREND="⚠️  ↓"
            fi
            if (( $(echo "$IQ > $BEST_SCORE" | bc -l) )); then
                BEST_SCORE=$IQ; BEST_FILE=$f
            fi
        else
            # Fallback to awk for float math
            if awk "BEGIN {exit !($IQ > $PREV_IQ)}"; then TREND="🚀 ↑"
            elif awk "BEGIN {exit !($IQ < $PREV_IQ)}"; then TREND="⚠️  ↓"
            fi
            if awk "BEGIN {exit !($IQ > $BEST_SCORE)}"; then
                BEST_SCORE=$IQ; BEST_FILE=$f
            fi
        fi

        printf "| %-15s | %-7s | %-12s | %-7s |\n" "$(basename "$f")" "$STEPS" "${IQ}%" "$TREND"
        PREV_IQ=$IQ
        mv "$JSON_FILE" "./research_logs/"
    else
        printf "| %-15s | %-7s | %-12s | %-7s |\n" "$(basename "$f")" "???" "???" "???"
        [ -f "$JSON_FILE" ] && rm "$JSON_FILE"
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

# --- 5. The Gatekeeper
THRESHOLD=2.0
if [ -n "$BEST_FILE" ]; then
    IS_PROMOTABLE=false
    if [ "$HAS_BC" = true ]; then
        if (( $(echo "$BEST_SCORE >= $THRESHOLD" | bc -l) )); then IS_PROMOTABLE=true; fi
    else
        if awk "BEGIN {exit !($BEST_SCORE >= $THRESHOLD)}"; then IS_PROMOTABLE=true; fi
    fi

    if [ "$IS_PROMOTABLE" = true ]; then
        echo "🎖️  Threshold Met ($BEST_SCORE% >= $THRESHOLD%). Promoting..."
        cp "$BEST_FILE" "data/models/gob_models/moe_classification_model_best.gob"
        
        if [ "$IS_WINDOWS" = true ]; then
             cp "data/models/gob_models/moe_classification_model_best.gob" "data/models/gob_models/moe_active.gob"
        else
             ln -sf "$(pwd)/data/models/gob_models/moe_classification_model_best.gob" "data/models/gob_models/moe_active.gob"
        fi
        
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

# Max Gradient Norm: controls global L2 clipping threshold.
# Increase if norm is always exactly at the cap (over-clipped).
# Decrease if loss is exploding or unstable.
# Pass as first arg: ./scripts/train.sh 5.0
# Training log (FRESH START for each audit)
TRAIN_LOG="logs/training_full.log"
rm -f "$TRAIN_LOG"

# Hyperparameters
LR=0.0005
MAX_GRAD_NORM=${1:-10.0}
# Dim 768 matches the requested GPT-3 Small configuration
EMBEDDING_DIM=768
NUM_EXPERTS=4
ACCUMULATION_STEPS=16
echo "📐 Max Gradient Norm: $MAX_GRAD_NORM"
echo "📉 Adjusted LR for 768d stability: $LR"

# Removed redundant 'go run' block here. Training now handled by pre-built binary below.

# --- 7. Stability Audit: scan the run for NaN / Inf signals ---
echo -e "\n🔬 Scanning training log for stability issues..."
if [ -s "$TRAIN_LOG" ] && grep -q "NaN\|Inf\|loss exploded" "$TRAIN_LOG"; then
    echo "⚠️  STABILITY ISSUE DETECTED in $TRAIN_LOG"
    echo "   → Possible causes:"
    echo "     1. LR too high - try: ./scripts/train.sh $MAX_GRAD_NORM (lower --lr)"
    echo "     2. Clip too loose - try: ./scripts/train.sh $(awk "BEGIN {print $MAX_GRAD_NORM * 0.5}")"
    echo "     3. Expert collapse - re-run with -rebalance"
else
    echo "✅ No NaN/Inf detected."
fi

CLIP_COUNT=$(grep "\[CLIPPED" "$TRAIN_LOG" 2>/dev/null | wc -l | xargs)
TOTAL_STEPS=$(grep "Weights updated" "$TRAIN_LOG" 2>/dev/null | wc -l | xargs)

if [ "${TOTAL_STEPS:-0}" -gt 0 ]; then
    PERCENT=$(awk "BEGIN {print ($CLIP_COUNT * 100) / $TOTAL_STEPS}")
    if [ "$CLIP_COUNT" -eq "$TOTAL_STEPS" ]; then
        echo "⚠️  Gradient clip fired on ALL $TOTAL_STEPS update steps."
    else
        echo "📐 Clip fired on $CLIP_COUNT / $TOTAL_STEPS steps (${PERCENT}%). Target is < 30%."
    fi
else
    echo "ℹ️  No update steps completed yet (too early to audit)."
fi

if command -v notify-send >/dev/null 2>&1; then
    notify-send "Gollemer AI" "Training Session Complete." --urgency=critical
fi

# --- 8. Stability & Diversity Audit ---
echo -e "\n📊 Generating MoE Stability Report..."
LOG_FILE="logs/training.csv"

if [ -f "$LOG_FILE" ]; then
    # Grab the last 5 entries to check for "Alpha Dominance"
    DOMINANCE=$(tail -n 5 "$LOG_FILE" | awk -F',' '{if($4 > 0.45) print $1}' | wc -l)
    
    DIVERSITY_SCORE="N/A"
    if [ "$HAS_JQ" = true ]; then
        DIVERSITY_SCORE=$(tail -n 1 "$LOG_FILE" | jq '.diversity_score' 2>/dev/null || echo "N/A")
    fi

    echo "--------------------------------------------------"
    if [ "$DOMINANCE" -gt 3 ]; then
        echo "⚠️  CRITICAL: High Alpha Dominance detected ($DOMINANCE/5 recent epochs)."
        echo "💡 Advice: Increase -diversity-coeff or check for Data Leaks."
    else
        echo "✨ Stability: Healthy Expert Distribution."
    fi
    echo "📈 Current Diversity Score: $DIVERSITY_SCORE%"
    echo "--------------------------------------------------"
fi
echo "🧹 Cleaning old Gollemer artifacts..."
rm -f static/main.wasm "gollemer_server${BIN_EXT}" "gollemer${BIN_EXT}"

echo "🚀 Building WASM Dashboard..."
GOOS=js GOARCH=wasm "$GO_CMD" build -o static/main.wasm ./examples/learningfolder/wasm

echo "⚙️ Compiling Go Server (EMA + Cooling + SIMD)..."
"$GO_CMD" build -o "gollemer_server${BIN_EXT}" ./examples/learningfolder

echo "⚙️ Compiling Main Gollemer binary..."
"$GO_CMD" build -o "gollemer${BIN_EXT}" ./cmd/tools/train_moe

# --- 9. Port Cleanup ---
echo "🛑 Clearing port :8080..."
if [ "$IS_WINDOWS" = true ]; then
    # PowerShell-based cleanup is more reliable on Windows
    powershell.exe -Command "Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"
else
    fuser -k 8080/tcp 2>/dev/null
fi

echo "🌐 System Live at http://localhost:8080"
"./gollemer_server${BIN_EXT}" --ema_alpha=0.001 --shake_threshold=0.01 & SERVER_PID=$!

# Step 2: Calculate decayed temperature for the current epoch
# Formula: TEMP = START_TEMP * (DECAY_RATE ^ EPOCH)
EPOCH=${EPOCH:-1}
START_TEMP=1.0
DECAY_RATE=0.95
CURRENT_TEMP=$(awk "BEGIN {print $START_TEMP * ($DECAY_RATE ^ $EPOCH)}")

echo "--- Starting Epoch $EPOCH with Temperature: $CURRENT_TEMP ---"

# Run a single training epoch with the pre-built binary
"./gollemer${BIN_EXT}" -train-chat -lr "$LR" -max_grad_norm="$MAX_GRAD_NORM" -batch-size 4 -acc-steps 16
# --- Functions ---
function bench() {
    go test -bench=. -benchmem ./...
}
function profile-cpu() {
    go test -bench=BenchmarkSparseMoE -cpuprofile=cpu.out
    go tool pprof -http=:8080 cpu.out
}
function check-race() {
    go test -race -v ./...
}
function clean() {
    rm -f *.out *.test cpu.prof mem.prof
}