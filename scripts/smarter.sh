#!/bin/bash
# smarter.sh — the one-command Gollemer upgrade.
#   1. Backs up the current models
#   2. Expands social + Go training data (idempotent: the quality gate
#      quarantines duplicates, so re-runs only add what's new)
#   3. Retrains the social brain (wider 128/256 dims) and the Go brain
#   4. Runs the eval suites and prints a before/after report
#
# The Go knowledge base (internal/ai/training/chat/goknowledge.go) is code,
# not data — it's already in the repo and needs no retraining.
#
# Hardened: set -euo pipefail so any failure stops the run; training
# output goes to log files (not through tail, which would mask a
# non-zero exit); GOMAXPROCS follows the actual CPU count.
set -euo pipefail
cd "$(dirname "$0")/.."

# Go toolchain location (not always on PATH in background sessions).
export PATH="$PATH:/home/hatch/golang/bin:/usr/local/go/bin"

BACKUP_DIR="$HOME/workspace/goals/get-gollemer-actually-working-as-an-llm/hidden_files"
LOG_DIR="$HOME/workspace/goals/get-gollemer-actually-working-as-an-llm/hidden_files"
STAMP=$(date +%Y%m%d_%H%M%S)
NPROC=$(nproc 2>/dev/null || echo 2)

echo "=== [1/4] Backing up current models ==="
mkdir -p "$BACKUP_DIR"
for d in social go gocode gocli makefile; do
  f="data/models/gob_models/real_tiny_seq2seq_${d}.gob"
  if [ -f "$f" ]; then
    cp "$f" "$BACKUP_DIR/real_tiny_seq2seq_${d}_pre_smarter_${STAMP}.gob"
    echo "  backed up $d"
  fi
done

echo "=== [2/4] Expanding training data ==="
# The quality gate quarantines duplicates, so re-importing is safe and
# idempotent — only genuinely new pairs are admitted.
export GOEXPERIMENT=simd CGO_ENABLED=1
for f in data/training/social_expansion_v*.jsonl data/training/go_expansion_*.jsonl; do
  [ -f "$f" ] || continue
  echo "  importing $f"
  go run main.go -import-pairs="$f" 2>&1 | grep -a -E "admitted|quarantined" | tail -2
done

echo "=== [3/4] Retraining social + go brains (this takes a while) ==="
echo "  (-resume: a killed run continues from its last checkpoint instead of epoch 0)"
echo "  using GOMAXPROCS=$NPROC (nproc)"
export GOEXPERIMENT=simd CGO_ENABLED=1
GOMEMLIMIT=2500MiB GOGC=100 GOMAXPROCS="$NPROC" go run main.go -train-real-seq2seq -resume -domain social > "$LOG_DIR/smarter_${STAMP}_social.log" 2>&1
echo "  social done (log: smarter_${STAMP}_social.log)"
tail -3 "$LOG_DIR/smarter_${STAMP}_social.log"
GOMEMLIMIT=2500MiB GOGC=100 GOMAXPROCS="$NPROC" go run main.go -train-real-seq2seq -resume -domain go > "$LOG_DIR/smarter_${STAMP}_go.log" 2>&1
echo "  go done (log: smarter_${STAMP}_go.log)"
tail -3 "$LOG_DIR/smarter_${STAMP}_go.log"

echo "=== [4/4] Evals ==="
echo "--- social multiturn ---"
python3 scripts/social_multiturn_eval.py 2>&1 | tail -3
echo "--- social targeted ---"
python3 scripts/social_targeted_eval.py 2>&1 | tail -3
echo "--- go concepts ---"
python3 scripts/goconcept_eval_run.py 2>&1 | tail -5
echo "--- gocode ---"
python3 scripts/gocode_eval_run.py 2>&1 | tail -5
echo "--- gocli ---"
python3 scripts/gocli_eval_run.py 2>&1 | tail -5

echo ""
echo "Done. Previous models are in $BACKUP_DIR (*_pre_smarter_${STAMP}.gob)."
echo "If anything regressed, copy a backup back over data/models/gob_models/."
