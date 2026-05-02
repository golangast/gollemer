# Gollemer

Gollemer is a high-performance **Mixture of Experts (MoE)** neural network framework and training pipeline written entirely in **Go**. It is designed for efficient conversational AI and intent classification without heavy external ML dependencies, featuring native GPU acceleration via WebGPU/Vulkan.

---

## 🚀 Key Features

- **Mixture of Experts Architecture**: Efficient multi-expert routing (K=1) for increased model capacity with low inference latency.
- **Native GPU Acceleration**: High-performance training and inference powered by `goffi` + WebGPU/Vulkan.
- **Stabilized MoE Training**: 
  - **Router Noise & Jitter**: Prevents expert collapse and forces specialization.
  - **Expert Health Monitoring**: Real-time tracking of expert utilization and saturation.
  - **Context Multiplier**: Enhanced signal strength for maintaining long-range conversational threads.
- **Sequence Awareness**: Integrated **Positional Encoding** in the MoE encoder to solve bag-of-words limitations.
- **Advanced Training Pipeline**:
  - **Social Curriculum**: Specialized training for natural, diverse human-like interaction.
  - **Scheduled Sampling**: Smooth transition from teacher-forcing to self-generated sequence learning.
  - **Robust Model Persistence**: Gzip-compressed checkpoints with legacy fallback support and atomic saving.
- **Pure Go Core**: No Python/C++ dependencies; leverages Go's concurrency for maximum data throughput.

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
We provide a simplified `Makefile` to handle the curriculum training process.

```bash
# Start the Social Curriculum Training (Reset state & begin)
make train
```

### 4. Running the LLM
Chat with your trained model using the interactive shell.

```bash
# Launch the interactive assistant
make llm
```

---

## 🛠️ Advanced Usage

### Makefile Commands
| Command | Description |
|---|---|
| `make train` | Cleans old state and starts the Social model training cycle. |
| `make llm` | Launches the interactive chat shell with the current model. |
| `make clean` | Safely removes current model checkpoints and vocabularies. |

### Configuration Tuning
Model stability is controlled via parameters in the training configuration:

- **`context_multiplier`**: Adjusts how much previous tokens dictate routing (1.5 - 2.0 recommended for deep context).
- **`router_noise`**: Stochastic jitter (0.8+) to ensure all experts are utilized.
- **`k=1`**: Forces each token to a single specialized expert, improving specialization.
- **`max_grad_norm`**: Hard cap (1.0) on updates to prevent mathematical instability.

---

## 🏗️ Project Structure

```text
.
├── cmd/                # Entry points (train_moe, tools)
├── internal/           # Core Logic
│   ├── ai/             # MoE, RNN, Positional Encoding, and Optimizers
│   └── training/       # Social and Intent curriculum pipelines
├── data/               # Assets
│   ├── models/         # Gzip-compressed .gob model weights
│   └── training/       # Conversational datasets
├── Makefile            # Simplified workflow entry points
├── build_gpu.sh        # GPU dependency setup script
└── GOB_DECODING_FIX.md # Documentation on recent serialization improvements
```

---

## 🚪 Exit
- **Interactive Shell**: Type `exit` to shut down.
- **Interrupt**: `Ctrl+C` terminates training or inference loops safely.
```