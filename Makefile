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
        build-pi build-pi64 pi pi-social pi-chat pi-llm \
        pi-social-master pi-social-worker pi-chat-master pi-chat-worker

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
	GOMEMLIMIT=4000MiB GOGC=100 $(MAIN_CMD) -llm -talk -listen

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

# Dynamically select the correct Pi binary based on what was built
PI_BIN ?= $(firstword $(wildcard ./gollemer-pi64 ./gollemer-pi ./gollemer))
ifeq ($(PI_BIN),)
PI_BIN = ./gollemer
endif

## pi: Resume Pi 3B social training (recommended first command on the Pi)
pi: pi-social

## pi-social: Resume social-only curriculum training in Pi 3B mode
pi-social:
	@echo "🥧 Pi 3B: Resuming Social Curriculum Training (900 MB safe mode)... using $(PI_BIN)"
	GOMEMLIMIT=$(PI_MEM) GOGC=$(PI_GOGC) GOMAXPROCS=1 \
		$(PI_BIN) $(PI_FLAGS) -train-social

## pi-social-fresh: Fresh social training on Pi (clears existing model first)
pi-social-fresh: clean pi-social

## pi-chat: Resume chat training in Pi 3B mode
pi-chat:
	@echo "🥧 Pi 3B: Resuming Chat Training (900 MB safe mode)... using $(PI_BIN)"
	GOMEMLIMIT=$(PI_MEM) GOGC=$(PI_GOGC) GOMAXPROCS=1 \
		$(PI_BIN) $(PI_FLAGS) -train-chat

## pi-llm: Launch the interactive LLM on the Pi (inference only — no -pi needed)
pi-llm:
	@echo "🥧 Pi 3B: Starting interactive LLM (inference mode)... using $(PI_BIN)"
	GOMEMLIMIT=$(PI_MEM) GOGC=50 GOMAXPROCS=2 \
		$(PI_BIN) -llm

# =============================================================================
# 🌐 Distributed Pi Targets — two-Pi parallel training over Ethernet
# =============================================================================
# Architecture:
#   Master Pi  — runs training + HTTP server on port 8080.
#                Receives weight updates from workers and writes the .gob files.
#   Worker Pi  — runs training only.  Does NOT write .gob files.
#                Sends weight tensors to the master after every 1000 batches.
#
# Usage (run each command on the respective Pi):
#   Master:  make pi-social-master                          (uses default port 8080)
#   Worker:  make pi-social-worker DIST_MASTER_IP=192.168.1.X
#
# The two Pis must be on the same LAN (e.g., both plugged into the same TP-Link
# router via Ethernet).  Set DIST_MASTER_IP to the master Pi's LAN address.

DIST_PORT        ?= 8080
DIST_MASTER_IP   ?= 192.168.1.100

## pi-social-master: Run distributed master Pi (trains + serves HTTP weight-sync endpoint)
pi-social-master:
	@echo "🌐 Pi Master: Starting distributed social training (port $(DIST_PORT))... using $(PI_BIN)"
	GOMEMLIMIT=$(PI_MEM) GOGC=$(PI_GOGC) GOMAXPROCS=1 \
		$(PI_BIN) $(PI_FLAGS) -train-social \
		-dist-mode=master -dist-addr=:$(DIST_PORT)

## pi-social-worker: Run distributed worker Pi (trains + streams weights to master)
pi-social-worker:
	@echo "🌐 Pi Worker: Starting distributed social training -> master $(DIST_MASTER_IP):$(DIST_PORT)... using $(PI_BIN)"
	GOMEMLIMIT=$(PI_MEM) GOGC=$(PI_GOGC) GOMAXPROCS=1 \
		$(PI_BIN) $(PI_FLAGS) -train-social \
		-dist-mode=worker -dist-addr=$(DIST_MASTER_IP):$(DIST_PORT)

## pi-chat-master: Distributed master for chat training
pi-chat-master:
	@echo "🌐 Pi Master: Starting distributed chat training (port $(DIST_PORT))... using $(PI_BIN)"
	GOMEMLIMIT=$(PI_MEM) GOGC=$(PI_GOGC) GOMAXPROCS=1 \
		$(PI_BIN) $(PI_FLAGS) -train-chat \
		-dist-mode=master -dist-addr=:$(DIST_PORT)

## pi-chat-worker: Distributed worker for chat training
pi-chat-worker:
	@echo "🌐 Pi Worker: Starting distributed chat training -> master $(DIST_MASTER_IP):$(DIST_PORT)... using $(PI_BIN)"
	GOMEMLIMIT=$(PI_MEM) GOGC=$(PI_GOGC) GOMAXPROCS=1 \
		$(PI_BIN) $(PI_FLAGS) -train-chat \
		-dist-mode=worker -dist-addr=$(DIST_MASTER_IP):$(DIST_PORT)
