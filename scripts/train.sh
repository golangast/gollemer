#!/bin/bash

# --- 1. Environment & Path Setup ---
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Setup Go environment for Chromebook performance
export GOGC=20
export GOMAXPROCS=2
export GO_CMD="/usr/local/go/bin/go"
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
    $GO_CMD run cmd/inspect/inspect_model.go --export "$f" > /dev/null
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

# --- 5. The Gatekeeper: Promotion & ChromeOS Notification ---
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
        ln -sf "$(pwd)/data/models/gob_models/moe_classification_model_best.gob" "data/models/gob_models/moe_active.gob"
        
        # �� ChromeOS Desktop Notification
        if command -v notify-send >/dev/null 2>&1; then
            notify-send "Gollemer AI" "Best Model Promoted! IQ: ${BEST_SCORE}%" --icon=utilities-terminal
        fi
    else
        echo "🛑  Threshold NOT Met ($BEST_SCORE%)."
    fi
fi

# --- 6. Launch Training ---
echo -e "\n🚀 Starting Gollemer Training..."
GOEXPERIMENT=simd $GO_CMD run cmd/gollemer/main.go \
    -train-chat \
    -rebalance \
    -auto-heal \
    -wd 0.01 \
    -lr 0.001

# --- 7. Completion Notification ---
echo -e "\a"
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
rm -f static/main.wasm gollemer_server

echo "🚀 Building WASM Dashboard..."
GOOS=js GOARCH=wasm $GO_CMD build -o static/main.wasm ./examples/learningfolder/wasm

echo "⚙️ Compiling Go Server (EMA + Cooling + SIMD)..."
$GO_CMD build -o gollemer_server ./examples/learningfolder

echo "🌐 System Live at http://localhost:5500"
./gollemer_server --ema_alpha=0.001 --shake_threshold=0.01
# Step 2: Calculate decayed temperature for the current epoch
# Formula: TEMP = START_TEMP * (DECAY_RATE ^ EPOCH)
START_TEMP=1.0
DECAY_RATE=0.95
CURRENT_TEMP=$(echo "scale=4; $START_TEMP * ($DECAY_RATE ^ $EPOCH)" | bc -l)

echo "--- Starting Epoch $EPOCH with Temperature: $CURRENT_TEMP ---"

# Pass the calculated temperature into your Go binary
./gollemer train --data="./data/train.bin" --temp=$CURRENT_TEMP
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
