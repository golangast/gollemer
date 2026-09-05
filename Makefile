# Gollemer Makefile
# -----------------------------------------------------------------------------

# Configuration
export GOEXPERIMENT=simd
export CGO_ENABLED=1

# Runtime Tuning
MEM_LIMIT    = 2500MiB
GOGC         = 50
GOMAXPROCS   = 8
MAIN_CMD     = go run main.go

.PHONY: train train-fresh clean clean-all help install-hooks \
       metrics export-labels train-small train-small-seq2seq \
       test-small-seq2seq seq2seq-prompt seq2seq-chat chat

## install-hooks: Install Gollemer Git pre-commit validation hook
install-hooks:
	@bash scripts/install_git_hook.sh

# --- Training ---

## train: Start a fresh curriculum training (clears MoE models, preserves word2vec)
train: clean
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-multiphase $(ARGS)

# train-resume: Start training without cleaning existing model checkpoints
train-resume:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-multiphase $(ARGS)

## train-fresh: Full fresh start — clears ALL models including word2vec, then trains
train-fresh: clean-all
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-multiphase $(ARGS)

## train-small: Run the small social dataset, print loss + memory, and test the model
train-small:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-small

## train-small-seq2seq: Run a strict pure Q→A seq2seq tiny demo
train-small-seq2seq:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-small-seq2seq

## test-small-seq2seq: Load the tiny seq2seq model and probe a few prompts
test-small-seq2seq:
	$(MAIN_CMD) -test-small-seq2seq

PROMPT ?= "hello"
## seq2seq-prompt: Send a custom prompt to the saved tiny seq2seq model (Usage: make seq2seq-prompt PROMPT="hello")
seq2seq-prompt:
	$(MAIN_CMD) -seq2seq-prompt="$(PROMPT)"

## seq2seq-chat: Start an interactive tiny seq2seq chat loop with the saved model
seq2seq-chat:
	$(MAIN_CMD) -seq2seq-chat

## chat: Start an interactive full MoE chat loop with conversation history and reasoning
chat:
	$(MAIN_CMD) -chat

# --- Analytics ---

## metrics: Run metrics aggregation and CSV export for edit logs
metrics:
	@echo "📊 Generating edit metrics and CSV..."
	@go run scripts/compute_edit_metrics.go || true
	@go run scripts/edits_to_csv.go || true
	@echo "✅ metrics written to logs/edits/"

## export-labels: Export training examples to CSV for manual labeling
export-labels:
	@echo "📝 Exporting edits_failed.jsonl -> data/training/edits_for_labeling.csv"
	@go run scripts/export_for_labeling.go || true


# --- Maintenance ---

## clean: Remove MoE model checkpoints (preserves word2vec to avoid cold restarts)
clean:
	rm -f data/models/gob_models/*.gob
	@if [ -f data/models/gob_models/word2vec_model.gob.bak ]; then \
		cp data/models/gob_models/word2vec_model.gob.bak data/models/gob_models/word2vec_model.gob 2>/dev/null || true; \
	fi

## clean-all: Remove ALL model files including word2vec (full cold start)
clean-all:
	rm -f data/models/gob_models/*.gob

## help: Display available commands
help:
	@echo "Available commands:"
	@grep -E '^##' Makefile | sed 's/## //'

# --- Dataset generation ---

## conversing-pb: Convert data/training/trainingdata/conversing.yaml -> conversing.pb
conversing-pb:
	go run ./cmd/tools/gen_conversing_yaml_pb
