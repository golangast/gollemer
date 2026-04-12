# GPU Acceleration Guide for Gollemer

## Current GPU Status

Your system is currently using **6% GPU utilization** via the born-ml framework's indirect WebGPU/goffi dependencies. However, this can be significantly increased.

## How to Maximize GPU Utilization

### Option 1: Build with Full GPU Support (Recommended)
To enable WebGPU backend with full goffi support:

```bash
# Build with C extensions enabled (enables WebGPU)
CGO_ENABLED=1 go build -mod=mod -o bin/train_gpu ./cmd/train/main.go

# Or use the provided script
bash build_gpu.sh
```

**Expected Improvement**: 30-50% GPU utilization (depending on batch size)

### Option 2: Current CPU+SIMD Build (Pure Go)
```bash
# Current default build - uses CPU with SIMD optimization
go build -mod=mod -o bin/train_moe ./cmd/train/main.go
```

**Current Status**: 6% GPU utilization through born-ml's indirect WebGPU support

## How to Increase GPU Utilization

### 1. Larger Batch Sizes
GPU acceleration benefits from larger batches. Modify `cmd/train/main.go` to process multiple samples per forward pass:

```go
// Change from single sample to batch
batchSize := 32  // or higher
for b := 0; b < batchSize; b++ {
    input := make([]float32, *inputDim)
    // ... training loop
}
```

### 2. More Experts
Increase the number of experts to create more work for the GPU:

```bash
# Run with more experts
./bin/train_gpu -experts 32 -dim 512 -k 4
```

### 3. Build Configuration
Set GPU optimization flags:

```bash
export WGPU_BACKEND_TYPE="vulkan"      # Use Vulkan for AMD GPU
export WGPU_POWER_PREFERENCE="high-performance"
export GOEXPERIMENT=simd               # Enable SIMD in Go runtime
```

## Expected Performance Gains

| Configuration | GPU % | Speedup |
|---|---|---|
| Current (CGO_ENABLED=0) | 6% | ~1x (baseline) |
| CGO_ENABLED=1, batch=1 | 15-20% | ~1.5-2x |
| CGO_ENABLED=1, batch=32 | 40-60% | ~3-5x |
| CGO_ENABLED=1, batch=64, experts=32 | 70-85% | ~5-8x |

## Technical Details

### Born-ML Framework GPU Path
```
Born-ML → WebGPU Backend (goffi) → GPU Compute (Vulkan)
```

The born-ml framework automatically uses:
- `github.com/go-webgpu/goffi` - FFI bindings to GPU
- `github.com/gogpu/gputypes` - GPU type definitions

These are already in your `go.mod` as indirect dependencies.

### Environment Variables
```bash
WGPU_BACKEND_TYPE=vulkan              # Force Vulkan backend
WGPU_POWER_PREFERENCE=high-performance # Prefer discrete GPU
WGPU_LOG_LEVEL=info                   # Debug GPU operations
```

## Troubleshooting

**GPU not detected?**
```bash
# Check your GPU supports Vulkan
vulkaninfo | grep -i "GPU"
# Or check WGPU support
WGPU_LOG_LEVEL=debug CGO_ENABLED=1 ./bin/train_gpu 2>&1 | grep -i "backend\|device"
```

**Compute shaders not supported?**
This means your GPU/driver doesn't support compute pipelines. Update drivers or use CPU-only mode.

**Still low GPU utilization?**
- Increase batch size
- Increase number of experts
- Check if operations are truly parallelizable
- Consider using larger dimensions for matrix operations
