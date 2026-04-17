# GPU Optimization Guide for Gollemer

## ⚠️ IMPORTANT: Benchmark vs Real Training

This guide covers GPU optimization for **performance benchmarking** (`cmd/train/main.go`).

**For Real LLM Training**, use:
```bash
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go -train-chat -gpu -batch-size 4 -epochs 30
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for LLM training with actual data.

---

## Overview

Gollemer achieves **~2000 samples/sec** on AMD RX 6600 using:
1. **libgoffi** - Foreign Function Interface for GPU calls
2. **WebGPU/Vulkan** - Cross-platform GPU compute
3. **Goroutine Pipelining** - CPU data preparation in parallel with GPU compute
4. **Expert Parallelization** - Multiple experts updating weights concurrently

---

## Architecture

### GPU Stack
```
Born-ML Framework (v0.7.14)
  ↓
libgoffi/gogpu (FFI bindings)
  ↓
go-webgpu/goffi (WebGPU wrapper)
  ↓
wgpu-native (Vulkan/D3D12/Metal)
  ↓
GPU Hardware (AMD/NVIDIA/Intel Arc)
```

### Execution Model
- **Sequential GPU Lock** (required for Vulkan stability on AMD)
- **Parallel Data Preparation** (CPU goroutine pre-generates batches)
- **Expert Gradient Updates** (each active expert launches in goroutine)
- **Prefetch Buffer** (channel keeps N batches ready)

---

## Setup & Installation

### 1. Download wgpu_native Library

**Linux/macOS:**
```bash
cd /tmp
curl -LO https://github.com/gfx-rs/wgpu-native/releases/latest/download/wgpu-linux-x86_64-release.zip
unzip wgpu-linux-x86_64-release.zip
sudo cp lib/libwgpu_native.so /usr/local/lib/
sudo ldconfig
```

**Windows:**
```powershell
# Download from: https://github.com/gfx-rs/wgpu-native/releases
# Extract and copy wgpu_native.dll to System32 or alongside executable
```

### 2. Build with GPU Support

```bash
cd gollemer
export CGO_ENABLED=1
go build -mod=mod -o bin/train_gpu ./cmd/train/main.go
```

### 3. Verify GPU Available

```bash
# Linux: Check Vulkan support
vulkaninfo 2>/dev/null | grep "GPU\|Device"

# Or run training - it will print backend info
./bin/train_gpu -batch 64 -experts 8 -dim 256
# Output: ✅ Born-ml backend ready (libgoffi GPU acceleration via indirect dependencies)

# Training runs indefinitely - press Ctrl+C to stop and save checkpoint
# [Ctrl+C]
# [!] Interrupt received. Finishing current batch and saving...
# [✓] Finalizing weights and exiting.
```

---

## Running & Controlling Training

### Continuous Training (Runs Until Interrupted)

```bash
# Training runs indefinitely, continuously processing batches
./bin/train_gpu -batch 256 -experts 8 -dim 256 -prefetch 4

# Output shows epoch progress:
# Epoch 0 | Loss: 0.030476 | Batch Size: 256 | Throughput: 1994 samples/sec
# Epoch 1 | Loss: 0.025123 | Batch Size: 256 | Throughput: 2104 samples/sec
# Epoch 2 | Loss: 0.019845 | Batch Size: 256 | Throughput: 2098 samples/sec
# ... continues until Ctrl+C
```

### Graceful Interruption

```bash
# Press Ctrl+C to gracefully stop training and save checkpoint
# [Ctrl+C in terminal]
# [!] Interrupt received. Finishing current batch and saving...
# [✓] Finalizing weights and exiting.
```

### Time-Limited Training

```bash
# Run training for exactly 60 seconds then exit
timeout 60 ./bin/train_gpu -batch 256 -experts 8 -dim 256 -prefetch 4

# Run training for 1 hour (3600 seconds)
timeout 3600 ./bin/train_gpu -batch 256 -experts 8 -dim 256 -prefetch 4

# Useful for overnight training or CI/CD pipelines
```

### Background Training

```bash
# Run training in background and check progress later
./bin/train_gpu -batch 256 -experts 8 -dim 256 -prefetch 4 > training.log 2>&1 &

# Monitor progress in real-time
tail -f training.log

# Stop background training
pkill -f train_gpu
```

---

## Tuning Performance

### Key Parameters

```bash
./bin/train_gpu \
  -batch 256      # Batch size: larger = better GPU utilization, more memory
  -experts 8      # Expert count: fewer wide experts > many thin experts
  -dim 256        # Hidden dimension: balance between capacity and compute
  -prefetch 4     # Prefetch buffer: keep GPU fed with queued batches
```

### Optimization Guidelines

| Parameter | Impact | Recommendation |
|-----------|--------|-----------------|
| `-batch` | GPU parallelism | 256+ for GPU, 64+ for CPU |
| `-experts` | Kernel launches | 8 (fewer, wider) beats 32 |
| `-dim` | Compute intensity | 256-512 optimal for RX 6600 |
| `-prefetch` | Pipeline efficiency | 2-4 batches |

### Throughput vs Model Capacity

```
Expert Count: 8         Expert Count: 32
Dim: 256               Dim: 512
Batch: 256             Batch: 64
---------              ---------
2000 samples/sec       567 samples/sec
(4x faster)            (smaller per-expert compute)
```

**Tradeoff**: Small model trains fast; large model learns better but slower.

---

## Benchmarks

### AMD RX 6600 (Gollemer Measured)

**With Goroutine Pipelining:**
- 8 experts, dim 256, batch 256, prefetch 4: **2000 samples/sec** ✅
- 8 experts, dim 256, batch 64: **1253 samples/sec**
- 32 experts, dim 512, batch 64: **567 samples/sec**
- 32 experts, dim 512, batch 64 (no pipelining): **304 samples/sec**

**Without GPU (CPU-only):**
- 8 experts, dim 256, batch 64: ~50 samples/sec (40x slower)

### Expected on Other GPUs

| GPU | Expected Throughput | Notes |
|-----|-------------------|-------|
| RTX 3080 | 5000-8000 samples/sec | 2.5-4x faster |
| RTX 4090 | 10000-15000 samples/sec | 5-7.5x faster |
| RX 7900 XTX | 3000-4000 samples/sec | 1.5-2x faster |
| Intel Arc A770 | 1500-2000 samples/sec | Similar to RX 6600 |

---

## Debugging & Troubleshooting

### GPU Not Detected

```bash
# Check if wgpu_native loaded
ldd ./bin/train_gpu | grep wgpu

# Missing? Reinstall library:
sudo cp /tmp/lib/libwgpu_native.so /usr/local/lib/
sudo ldconfig
```

### Low Throughput

1. **Check batch size**: Increase to 256+
2. **Reduce experts**: Use 8 experts instead of 32
3. **Increase prefetch**: `-prefetch 4` or `-prefetch 8`
4. **Monitor GPU**: `radeontop -d /dev/dri/card0` (AMD)

### Crashes or Hangs

- **Keep GPU lock sequential** (required for Vulkan stability)
- Expert parallelism is via goroutines, not parallel GPU kernels
- If hanging: reduce batch size or number of experts

---

## Advanced Optimization

### Gradient Accumulation (Training Stability)

```bash
# Simulate batch 512 with memory for batch 64
./bin/train_gpu -batch 64 -prefetch 8  # Chains 8 batches = 512 effective
```

### Mixed Precision Training (Future)

Currently not implemented, but architecture supports:
- float32 GPU compute (current)
- float16 storage (reduce memory)
- Matrix multiplication in float16, accumulation in float32

### Multi-GPU Training (Future)

Multiple GPUs would require:
1. Distributed expert sharding
2. Gradient synchronization between GPUs
3. Currently single-GPU only

---

## Goroutine Pipelining Implementation

The core optimization runs:

```go
// GPU goroutine (sequential, locked)
Expert.Forward() → GPU kernel launch
Expert.Backward() → GPU kernel launch

// Parallel with above
DataGenerator goroutine: Pre-compute next batch in channel buffer
Expert.UpdateWeights() goroutine: Each active expert updates simultaneously
```

**Result**: GPU never idles waiting for data preparation.

---

## Monitoring

### Build with Profiling

```bash
go build -mod=mod -o bin/train_gpu_prof -gcflags="-cpuprofile=cpu.prof" ./cmd/train/main.go
./bin/train_gpu_prof -batch 64 -experts 8 -dim 256
go tool pprof cpu.prof
```

### Runtime Metrics

During training, monitor:
- **Samples/sec** reported in console
- **GPU utilization** (radeontop, nvidia-smi)
- **Memory usage** (watch -n 1 'free -h')
- **CPU usage** (prefetch should be ~10-20% CPU, rest GPU)

---

## FAQ

**Q: Why not parallel GPU kernels?**
A: Vulkan on AMD requires sequential execution. Parallel kernels crash driver. Pipelined goroutines solve this elegantly.

**Q: Can I use CUDA instead of Vulkan?**
A: Not currently in born-ml v0.7.14. Windows uses D3D12, macOS uses Metal.

**Q: What's the memory overhead?**
A: ~200MB for GPU buffers (Born-ML) + batch buffers. Negligible on modern GPUs.

**Q: How do I optimize for inference?**
A: Use smaller experts (8 vs 32), smaller dims (128 vs 512), and disabled training mode.

**Q: Can I use `CGO_ENABLED=0` for GPU?**
A: No - GPU requires C bindings for wgpu_native library. Must set `CGO_ENABLED=1`.
