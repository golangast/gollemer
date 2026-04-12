#!/bin/bash
# Build script to enable GPU acceleration via born-ml and goffi
# This enables WebGPU backend through C bindings

set -e

echo "🚀 Building with GPU acceleration (CGO_ENABLED=1)..."
echo "This enables WebGPU/goffi backend for born-ml framework"

# Export CGO_ENABLED to allow C bindings (required for GPU)
export CGO_ENABLED=1

# Optional: Set WGPU backend for GPU dispatch
export WGPU_BACKEND_TYPE="vulkan"
export WGPU_POWER_PREFERENCE="high-performance"

# Build with GPU support
go build -mod=mod -o bin/train_gpu ./cmd/train/main.go

echo "✅ GPU-accelerated build complete: bin/train_gpu"
echo "   Run with: ./bin/train_gpu -experts 16 -dim 256"
