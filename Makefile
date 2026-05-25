# Gollemer Makefile
# -----------------------------------------------------------------------------
# High-performance Mixture of Experts (MoE) Framework
# -----------------------------------------------------------------------------

# Configuration
# Use GOEXPERIMENT=simd for native AVX2/SSE acceleration
export GOEXPERIMENT=simd
export CGO_ENABLED=1

# Runtime Tuning
MEM_LIMIT = 5000MiB
GOGC      = 50
MAIN_CMD  = go run cmd/tools/train_moe/main.go

.PHONY: train train-social chat clean help

# --- Training ---

## train: Start a fresh Social Curriculum training (clears old state)
train: clean
	@echo "🚀 Starting Fresh Social Curriculum Training..."
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) $(MAIN_CMD) -train-social

## train-social: Resume existing Social Curriculum training
train-social:
	@echo "🚀 Resuming Social Curriculum Training..."
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) $(MAIN_CMD) -train-social

# --- Interaction ---

## chat: Launch the interactive LLM chat shell
chat:
	@echo "💬 Starting Interactive Chat..."
	GOMEMLIMIT=4000MiB GOGC=100 $(MAIN_CMD) -llm -talk

llm: chat

# --- Maintenance ---

## clean: Remove all model checkpoints and cached vocabularies
clean:
	@echo "🧹 Cleaning model state..."
	rm -f data/models/gob_models/*.gob

## help: Display available commands
help:
	@echo "Available commands:"
	@grep -E '^##' Makefile | sed 's/## //'
