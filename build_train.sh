#!/bin/bash
# Build and run training with batch support

set -e

echo "📦 Building Gollemer training with batch support..."

# Uncomment the line below to enable GPU acceleration with CGO
# export CGO_ENABLED=1

go build -mod=mod -o bin/train_moe ./cmd/train/main.go

echo "✅ Build complete"
echo ""
echo "📊 Available flags:"
echo "  -dim INT        Input dimension (default: 128)"
echo "  -experts INT    Number of experts (default: 8)"
echo "  -k INT          Top-k experts per sample (default: 2)"
echo "  -lr FLOAT       Learning rate (default: 0.01)"
echo "  -batch INT      Batch size for training (default: 1)"
echo ""
echo "🚀 Example commands:"
echo "  ./bin/train_moe -batch 1 -experts 8 -dim 128          (baseline)"
echo "  ./bin/train_moe -batch 32 -experts 16 -dim 256        (medium)"
echo "  ./bin/train_moe -batch 64 -experts 24 -dim 512        (large)"
echo ""
echo "⚡ For GPU acceleration, enable CGO:"
echo "  CGO_ENABLED=1 go build -mod=mod -o bin/train_gpu ./cmd/train/main.go"
echo "  ./bin/train_gpu -batch 64 -experts 16 -dim 256"
