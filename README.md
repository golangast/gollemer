# Gollemer

Gollemer is a high-performance **Mixture of Experts (MoE)** neural network framework and training pipeline written entirely in **Go**. It is designed for maximum performance with **zero external dependencies**, featuring a native core optimized for SIMD-accelerated CPU training.

---

## 🚀 Key Features

- **Mixture of Experts Architecture**: Efficient multi-expert routing (K=1) for increased model capacity with low inference latency.
- **High Performance**: Native SIMD-vectorized operations for AVX2/SSE/Neon.
- **Stabilized MoE Training**: 
  - **Router Noise & Jitter**: Prevents expert collapse and forces specialization.
  - **Expert Health Monitoring**: Real-time tracking of expert utilization and saturation.
  - **Context Multiplier**: Enhanced signal strength for maintaining long-range conversational threads.
- **Sequence Awareness**: Integrated **Positional Encoding** in the MoE encoder to solve bag-of-words limitations.
- **Advanced Training Pipeline**:
  - **Social Curriculum**: Specialized training for natural, diverse human-like interaction.
  - **Scheduled Sampling**: Smooth transition from teacher-forcing to self-generated sequence learning.
  - **Robust Model Persistence**: Gzip-compressed checkpoints with legacy fallback support and atomic saving.
- **Dependency-Free Core**: No Rust or Python toolchains required. The entire neural engine is native Go.
- **Hardware-Native Performance**:
  - **CPU**: Optimized Go assembly and SIMD intrinsics via `archsimd`.
  - **CPU**: Native **SIMD acceleration** using Go's `archsimd` (AVX2/SSE), matching BLAS performance without external libraries like Gonum.
- **Ultra-Portable**: Compiles to a single static binary. No shared libraries or complex environments needed.

---

## 🏗️ Why Gollemer?

By eliminating heavy external libraries like **Rust-based backends** and **Gonum**, Gollemer achieves a state of "Mechanical Sympathy" with the Go runtime:
- **Faster Setup**: No need for Rust toolchains, LLVM, or complex CGO configurations for basic training.
- **Superior Portability**: The native SIMD fallbacks ensure that the model remains fast even on systems without a dedicated GPU.
- **Lean Footprint**: A single binary deployment with zero runtime dependencies (except for optional GPU drivers).

---

## ⚡ Quick Start

### 1. Installation
Gollemer requires **Go 1.26+** to leverage native SIMD acceleration.

```bash
git clone https://github.com/golangast/gollemer
cd gollemer
# Enable SIMD experiments for maximum CPU performance
export GOEXPERIMENT=simd 
go mod tidy
```

### 2. GPU Acceleration
GPU acceleration is powered by **Goffi** and requires no manual compilation of Rust or C++ backends. It automatically attempts to leverage **WebGPU/Vulkan** if drivers are present.

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

## ⚙️ Social Training Configuration

The `data/config/social_train.json` file controls the behavior of the social curriculum training. Tuning these parameters is key to achieving stable and coherent conversational output.

### Core Architecture
| Property | Description | Recommendation |
|---|---|---|
| `"num_experts"` | Total number of specialized feed-forward sub-networks. | 8-12 for balanced performance. |
| `"model_dim"` | Vector size for embeddings and hidden layers. | 256 for efficiency on CPU/GPU. |
| `"k"` | Number of experts used per token. | Keep at `1` for maximum specialization. |

### Training Dynamics
| Property | Description | Rationale |
|---|---|---|
| `"epochs"` | Total passes through the dataset. | High values (1000+) help MoE stability. |
| `"learning_rate"` | Magnitude of weight updates. | 0.001 - 0.0005 to avoid NaN loss. |
| `"batch_size"` | Samples per iteration. | 1-4 depending on available VRAM. |
| `"accumulate_steps"` | Virtual batching (Effective Batch = batch * steps). | Increase for smoother gradients. |
| `"overfit_mode"` | Forces verbatim memorization of small datasets. | `true` for small conversational anchor sets. |

### MoE Routing & Stability
| Property | Description | Usage |
|---|---|---|
| `"context_multiplier"` | Strength of the signal from previous tokens. | 2.0 - 5.0 for deep conversational context. |
| `"router_noise"` | Stochastic jitter added to expert selection. | 0.1 - 0.8 to force expert exploration. |
| `"router_temperature"`| Sharpness of expert selection probability. | `1.0` for balanced; lower for "hard" routing. |
| `"load_balancing_weight"` | Penalty for experts that monopolize tokens. | 0.01 - 0.1 to keep experts active. |
| `"collapse_threshold"`| Usage rate below which an expert is reset. | 0.15; triggers `auto_heal` for dead experts. |
| `"auto_heal"` | Automatically resets/re-randomizes dead experts. | `true` to maintain model health during training. |
| `"capacity_factor"` | Limits token load per expert during routing. | 1.0 - 1.5; prevents expert saturation. |

### Generation & Sampling
| Property | Description | Usage |
|---|---|---|
| `"repetition_penalty"`| Penalizes tokens already present in the output. | 1.2 - 2.0 to prevent "word salad" loops. |
| `"label_smoothing"` | Softens targets to prevent overconfidence. | 0.1 for generalization; 0 for strict recall. |
| `"sampling_start_epoch"`| When to start using model output for training. | Epoch 50-100; lets model learn basics first. |
| `"sampling_max_prob"`| Max probability of model self-sampling. | 0.3 - 0.5 to balance ground-truth vs feedback. |
| `"verbose_thinking"` | Displays internal expert selection in logs. | `true` for debugging router behavior. |

---

## 🏗️ Project Structure

Gollemer is organized into a clean, modular structure designed for ease of use and high performance:

```text
.
├── cmd/tools/          # Entry Points
│   └── train_moe/      # The primary training engine and interactive LLM shell.
│
├── internal/ai/        # Core Intelligence Engine
│   ├── moe/            # Mixture of Experts (MoE) core architecture and routing logic.
│   ├── neural/         # Native neural math:
│   │   ├── tensor/     # SIMD-accelerated (AVX2/SSE) tensor operations.
│   │   ├── nn/         # Neural network layers (Linear, Embedding, etc.).
│   │   └── tokenizer/  # Advanced sub-word and sentence tokenization.
│   ├── llm/            # High-level assistant runner, client, and intent logic.
│   └── training/       # Specialized curriculum pipelines (Social, Intent).
│
├── data/               # Assets & Persistence
│   ├── config/         # Hot-reloadable training and model configurations.
│   ├── models/         # Serialized (.gob) model checkpoints and vocabularies.
│   ├── training/       # Curated datasets (JSON/CSV/TXT) for social learning.
│   ├── db/             # SQLite-backed long-term memory and knowledge bases.
│   └── knowledge.json  # Static world-knowledge and retrieval facts.
│
├── scripts/            # Support Tools
│   └── *.go            # Standalone tools for vocab analysis and model inspection.
│
├── Makefile            # Central workflow: train, chat, and clean.
└── go.mod              # Minimal dependency management (Goffi only).
```

---

## 🚪 Exit
- **Interactive Shell**: Type `exit` to shut down.
- **Interrupt**: `Ctrl+C` terminates training or inference loops safely.
```