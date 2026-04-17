#!/bin/bash

# Configuration
export OPENBLAS_NUM_THREADS=1
export CGO_ENABLED=1
export GOEXPERIMENT=simd
export WGPU_BACKEND_TYPE="vulkan"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# Ensure data directories exist
mkdir -p data/models/checkpoints docs/logs

echo "🚀 Starting GPU-Accelerated MoE Training Pipeline..."
echo "📍 Backend: Vulkan (via wgpu-native)"
echo "📍 Accelerator: OpenBLAS (via Netlib/CGO)"

# Build the optimized binary
/usr/local/go/bin/go build -mod=mod -o bin/gollemer_gpu ./cmd/tools/train_moe/main.go

if [ $? -eq 0 ]; then
    echo "✅ Build Successful. Launching training..."
    ./bin/gollemer_gpu -train-social -batch-size 128 -acc-steps 4 -gpu "$@"
else
    echo "❌ Build failed."
    exit 1
fi
