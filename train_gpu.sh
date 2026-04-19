#!/bin/bash

# Configuration
# Configuration
export OPENBLAS_NUM_THREADS=1
export CGO_ENABLED=0
export GOEXPERIMENT=simd
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH

# Ensure data directories exist
mkdir -p data/models/checkpoints docs/logs

echo "🚀 Starting High-Performance MoE Training Pipeline..."
echo "📍 Backend: OpenCL (via goffi)"
echo "📍 Accelerator: Radeon GPU + Native OpenCL Dispatch (CGO_ENABLED=0)"

# Build the optimized binary
CGO_ENABLED=0 /usr/local/go/bin/go build -tags gpu -mod=mod -o bin/gollemer_final_gpu ./cmd/tools/train_moe/main.go

if [ $? -eq 0 ]; then
    echo "✅ Build Successful. Launching training..."
    ./bin/gollemer_final_gpu -train-social -gpu -batch-size 64 -acc-steps 8 "$@"
else
    echo "❌ Build failed."
    exit 1
fi
