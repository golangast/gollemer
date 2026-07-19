# Gollemer Makefile
# -----------------------------------------------------------------------------
# High-performance Mixture of Experts (MoE) Framework
# -----------------------------------------------------------------------------

# Configuration
# Use GOEXPERIMENT=simd for native AVX2/SSE acceleration (x86 only — NOT for Pi)
export GOEXPERIMENT=simd
export CGO_ENABLED=1

# Runtime Tuning
MEM_LIMIT    = 5500MiB
GOGC         = 90
GOMAXPROCS   = 8
TRAIN_BIN    = ./.build/gollemer-train
MAIN_CMD     = go run cmd/tools/train_moe/main.go

.PHONY: train train-fresh train-social chat clean clean-all help dashboard \
        build-pi build-pi64 pi pi-social pi-chat pi-llm \
        pi-social-master pi-social-worker pi-chat-master pi-chat-worker

# --- Training ---

## train: Start a fresh Social Curriculum training (clears MoE models, preserves word2vec)
train: clean
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-multiphase $(ARGS)

## train-fresh: Full fresh start — clears ALL models including word2vec, then trains
train-fresh: clean-all
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-multiphase $(ARGS)

## train-social: Resume existing Social Curriculum training
## Suspends ALL dashboard services before training to free ~1.5 GiB RAM, restarts on completion.
train-social:
	@echo "⏸  Suspending dashboard services to free RAM for training..."
	@for f in /tmp/dashboard.pid /tmp/observability.pid /tmp/dashboard_injector.pid; do \
	    [ -f $$f ] && kill -9 $$(cat $$f) 2>/dev/null || true; rm -f $$f; done
	@sleep 2
	@echo "🔨 Building training binary..."
	@mkdir -p .build
	go build -o .build/gollemer-train ./cmd/tools/train_moe/main.go
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=20 GOMAXPROCS=$(GOMAXPROCS) .build/gollemer-train -train-social $(ARGS)
	@echo "▶️  Restarting dashboard (http://localhost:8765)..."
	@bash scripts/start_dashboard.sh

# --- Interaction ---

## chat: Launch the interactive LLM chat shell
chat:
	@echo "💬 Starting Interactive Chat..."
	GOMEMLIMIT=4000MiB GOGC=100 $(MAIN_CMD) -llm -talk -listen

llm: chat

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

word2vec:
	go run ./cmd/tools/train_word2vec
	@cp data/models/gob_models/word2vec_model.gob data/models/gob_models/word2vec_model.gob.bak 2>/dev/null || true

## help: Display available commands
help:
	@echo "Available commands:"
	@grep -E '^##' Makefile | sed 's/## //'

## dashboard: Show how to start the live training dashboard
dashboard:
	bash scripts/start_dashboard.sh
	@echo "🖥️  Gollemer Training Dashboard → http://localhost:8765"
	@echo "🧭 Advanced Observability → http://localhost:8765/observability"
	@echo "To start the dashboard and demo observability services, run:"
	@echo "  bash scripts/start_dashboard.sh"
	@echo "Or run detached via Make: make dashboard.start"

## dashboard.start: Start observability+injector+dashboard detached (background)
dashboard.start:
	@echo "🔁 Launching demo observability server and live injector (detached)..."
	@pkill -f cmd/tools/observability_example/main.go || true
	@pkill -f scripts/dashboard_injector.sh || true
	@sh -c 'nohup bash scripts/start_dashboard.sh > /dev/null 2>&1 & echo \$! > /tmp/start_dashboard_launcher.pid; exit 0'
	@echo "✅ Launched (check /tmp/*.pid and /tmp/*.log)"

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
