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

```json
  /* Model Architecture 
     Defines the "physical" limits of your MoE network.
  */
  "num_experts": 12, // The number of specialized FFN sub-networks.
  // Why: More experts = better nuances but higher RAM/VRAM cost.
  // Change when: Lower if you hit OOM; raise if the model can't differentiate complex intents.
  "model_dim": 256, // Embedding and hidden layer vector size.
  // Why: 256 is efficient for Go/CPU; fits well in cache.
  // Change when: Raise (e.g. 512) for "deeper" logic; lower to 128 for ultra-fast inference.
  /* Training Loop 
     Controls how the weights are updated over time.
  */
  "epochs": 1200, // Iterations through the full dataset.
  // Why: MoE requires high epochs for the router to stabilize expert selection.
  // Change when: Increase if loss is still trending down; decrease if the model overfits.
  "learning_rate": 0.0001, // The magnitude of weight updates (1e-4).
  // Why: Prevents "gradient explosion" in custom Go implementations.
  // Change when: Raise if training stalls; lower if the loss becomes 'NaN'.
  "batch_size": 1, // Real-time samples processed per iteration.
  // Why: Maximizes RAM efficiency on your 16GB setup.
  // Change when: Raise if you have plenty of free RAM to speed up training.
  "accumulate_steps": 4, // Virtual batching (Effective Batch = batch_size * accumulate_steps).
  // Why: Provides gradient stability of a batch of 4 without the memory overhead.
  // Change when: Increase if the loss curve is too "noisy" or jagged.
  /* MoE Logic: Routing & Stability 
     Crucial for ensuring the "Mixture" part of the MoE actually works.
  */
  "context_multiplier": 1.0, // Weight of the context window influence.
  // Why: Adjusts how much previous tokens dictate current routing.
  // Change when: Raise if the model loses the "thread" of long sentences.
  "router_noise": 0.8, // Stochastic jitter added to expert selection.
  // Why: Forces the router to explore all experts during training.
  // Change when: Lower if the router is already balanced; raise if only 1 expert is active.
  "expert_dropout": 0.3, // Randomly skips experts during a training pass.
  // Why: Encourages redundancy and prevents "lazy" experts.
  // Change when: Increase if the model is memorizing training data verbatim.
  "collapse_threshold": 0.4, // The minimum usage rate for an expert before it's considered "dead".
  // Why: Triggers your 'auto-heal' or reset logic for underutilized experts.
  // Change when: Adjust based on how many experts you expect to be "generalists."
  "label_smoothing": 0.1, // Distribution of probability across labels.
  // Why: Softens hard targets to help the model generalize.
  // Change when: Lower if the model is too "unsure" (flat probabilities).
  "weight_decay": 0.0001, // L2 regularization penalty.
  // Why: Prevents any single weight from becoming a "bottleneck" outlier.
  // Change when: Increase if weights are ballooning; decrease if weights are too suppressed.
  "max_grad_norm": 1.0, // Gradient clipping threshold.
  // Why: Hard cap on update size to prevent mathematical instability in Go.
  // Change when: Lower if you encounter frequent 'NaN' or '
```