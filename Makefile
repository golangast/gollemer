# Gollemer Makefile
# -----------------------------------------------------------------------------
# High-performance Mixture of Experts (MoE) Framework
# -----------------------------------------------------------------------------

# Configuration
# Use GOEXPERIMENT=simd for native AVX2/SSE acceleration (x86 only — NOT for Pi)
export GOEXPERIMENT=simd
export CGO_ENABLED=1

# Runtime Tuning
MEM_LIMIT = 5000MiB
GOGC      = 50
MAIN_CMD  = go run cmd/tools/train_moe/main.go

.PHONY: train train-social chat clean help \
        build-pi build-pi64 pi pi-social pi-chat pi-llm

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

# =============================================================================
# 🥧 Pi 3B targets  (Raspberry Pi 3B, ~900 MB RAM, ARM Cortex-A53)
# =============================================================================
# NOTE: GOEXPERIMENT=simd and CGO are intentionally disabled here — the Pi has
# no x86 SIMD and cross-compilation with CGO requires a separate toolchain.

## build-pi: Cross-compile a 32-bit ARM binary for Raspberry Pi 3B (Raspberry Pi OS 32-bit)
build-pi:
	@echo "🔨 Cross-compiling for Pi 3B (linux/arm, ARMv7)..."
	GOEXPERIMENT= CGO_ENABLED=0 GOOS=linux GOARCH=arm GOARM=7 \
		go build -o gollemer-pi ./cmd/tools/train_moe/main.go
	@echo "✅ Binary ready: gollemer-pi  (copy to Pi and run with ./gollemer-pi -pi ...)"

## build-pi64: Cross-compile a 64-bit ARM binary for Pi 3B running a 64-bit OS
build-pi64:
	@echo "🔨 Cross-compiling for Pi 3B (linux/arm64)..."
	GOEXPERIMENT= CGO_ENABLED=0 GOOS=linux GOARCH=arm64 \
		go build -o gollemer-pi64 ./cmd/tools/train_moe/main.go
	@echo "✅ Binary ready: gollemer-pi64  (copy to Pi and run with ./gollemer-pi64 -pi ...)"

# Pi runtime settings:
#   GOMEMLIMIT=700MiB  — leave ~200 MB headroom for the OS out of the 900 MB total
#   GOGC=10            — GC fires very aggressively to keep heap below limit
#   GOMAXPROCS=1       — set by -pi in code, but also enforced here at the OS level
PI_MEM   = 700MiB
PI_GOGC  = 10
PI_FLAGS = -pi

## pi: Resume Pi 3B social training (recommended first command on the Pi)
pi: pi-social

## pi-social: Resume social-only curriculum training in Pi 3B mode
pi-social:
	@echo "🥧 Pi 3B: Resuming Social Curriculum Training (900 MB safe mode)..."
	GOMEMLIMIT=$(PI_MEM) GOGC=$(PI_GOGC) GOMAXPROCS=1 \
		./gollemer $(PI_FLAGS) -train-social

## pi-social-fresh: Fresh social training on Pi (clears existing model first)
pi-social-fresh: clean pi-social

## pi-chat: Resume chat training in Pi 3B mode
pi-chat:
	@echo "🥧 Pi 3B: Resuming Chat Training (900 MB safe mode)..."
	GOMEMLIMIT=$(PI_MEM) GOGC=$(PI_GOGC) GOMAXPROCS=1 \
		./gollemer $(PI_FLAGS) -train-chat

## pi-llm: Launch the interactive LLM on the Pi (inference only — no -pi needed)
pi-llm:
	@echo "🥧 Pi 3B: Starting interactive LLM (inference mode)..."
	GOMEMLIMIT=$(PI_MEM) GOGC=50 GOMAXPROCS=2 \
		./gollemer -llm
