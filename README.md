# Gollemer

Gollemer is a high-performance **Mixture of Experts (MoE)** neural network framework and training pipeline written entirely in **Go**. It is designed for efficient conversational AI and intent classification without heavy external ML dependencies, featuring native GPU acceleration via WebGPU/Vulkan.

---

## 🚀 Key Features

- **Mixture of Experts Architecture**: Efficient multi-expert routing for increased model capacity with low inference latency.
- **Native GPU Acceleration**: High-performance training and inference powered by `goffi` + WebGPU/Vulkan (Linux-first).
- **Two-Phase Training Pipeline**:
  - **Phase 0: MLM Pre-training**: Masked Language Modeling (fill-in-the-blank) to teach fundamental grammar and word relationships.
  - **Phase 1: Seq2Seq Fine-tuning**: Intent-based supervised training for conversational and technical tasks.
- **Pure Go Core**: Built from the ground up in Go, leveraging goroutines for data pipelining and concurrency.
- **No Python/C++ Dependencies**: Avoids the complexity of PyTorch, TensorFlow, or large C++ runtimes.

---

## ⚡ Quick Start

### 1. Installation
Ensure you have Go 1.22+ and a working C compiler (GCC) for GPU support.

```bash
git clone https://github.com/golangast/gollemer
cd gollemer
go mod tidy
```

### 2. GPU Setup (Linux)
Gollemer uses WebGPU via `wgpu-native`.

```bash
# Install wgpu-native binaries
./build_gpu.sh
```

### 3. Training the Model
Gollemer features an automated training cycle that handles both grammar (MLM) and chat (Seq2Seq).

```bash
# Start the full training cycle (MLM followed by Seq2Seq)
CGO_ENABLED=1 go run cmd/tools/train_moe/main.go -train-chat -gpu -batch-size 4 -acc-steps 16

# Train a specialized Social model for natural conversation
CGO_ENABLED=1 go run cmd/tools/train_moe/main.go -train-social -gpu -batch-size 4 -epochs 30
```

### 4. Running Inference
Once trained, launch the interactive LLM shell to chat with the model.

```bash
# Run the interactive assistant
CGO_ENABLED=0 go run cmd/tools/train_moe/main.go -llm
```

---

## 🛠️ Advanced Usage

### GPU Benchmarking
Test the raw throughput of the MoE implementation on your hardware using synthetic data.

```bash
# Build and run the benchmark
go build -o bin/train_gpu ./cmd/train/main.go
./bin/train_gpu -batch 256 -experts 8 -dim 256 -prefetch 4
```

### Configuration Flags
| Flag | Description | Recommended |
|---|---|---|
| `-gpu` | Enable WebGPU/Vulkan acceleration | Required for speed |
| `-batch-size` | Number of samples per training step | 4-8 for 8GB VRAM |
| `-acc-steps` | Gradient accumulation steps | 8-16 |
| `-lr` | Learning rate | 0.0001 |
| `-epochs` | Number of training passes | 20-50 |

---

## 🏗️ Project Structure

```text
.
├── cmd/                # Entry points
│   ├── train/          # GPU Benchmarking tool
│   └── tools/
│       └── train_moe/  # Main training and LLM entry point
├── internal/           # Core Logic
│   └── ai/             # MoE, RNN, MLM, and Optimizer implementations
├── data/               # Assets
│   ├── models/         # Saved .gob model weights
│   └── training/       # Training datasets (CSV/TXT)
├── build_gpu.sh        # GPU dependency setup script
└── TRAINING_GUIDE.md   # Detailed training workflows
```

---

## 🚪 Exit
- **Interactive Shell**: Type `exit` to shut down.
- **Interrupt**: `Ctrl+C` terminates training or inference loops safely.
