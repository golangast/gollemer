# Gollemer

Gollemer is a high-performance **Mixture of Experts (MoE)** neural network framework and training pipeline written entirely in **Go**. It is designed for maximum performance with **zero external dependencies**, featuring a native core optimized for SIMD-accelerated CPU training and an autonomous adaptive supervisor.

---

## 🚀 Key Features

- **Mixture of Experts Architecture**: Efficient multi-expert routing ($K=1$) for increased model capacity with low inference latency.
- **High Performance**: Native SIMD-vectorized operations for AVX2/SSE/Neon.
- **Autonomous Adaptive Supervisor**: Real-time hyperparameter reflection, self-directed expert spawning/eviction, and on-the-fly training data evolution.
- **Chromebook & Raspberry Pi Ready**: Lightweight enough to train models entirely on Chromebooks, Raspberry Pi 3B, and other low-power commodity devices.
- **Stabilized MoE Training**: 
  - **Router Noise & Jitter**: Prevents expert collapse and forces specialization.
  - **Expert Health Monitoring**: Real-time tracking of expert utilization and saturation.
  - **Context Multiplier**: Enhanced signal strength for maintaining long-range conversational threads.
- **Sequence Awareness**: Integrated **Positional Encoding** in the MoE encoder to solve bag-of-words limitations.
- **Advanced Training Pipeline**:
  - **Social Curriculum**: Specialized training for natural, diverse human-like interaction.
  - **Scheduled Sampling**: Smooth transition from teacher-forcing to self-generated sequence learning.
  - **Robust Model Persistence**: Gzip-compressed checkpoints with legacy fallback support and atomic saving.
- **Dependency-Free Core**: No Python, Rust, PyTorch, C++, or external BLAS (like Gonum/OpenBLAS) required. The entire neural engine is native Go.
- **Hardware-Native Performance**:
  - **CPU**: Native **SIMD acceleration** using Go's `archsimd` (AVX2/SSE/Neon), matching BLAS performance without external C libraries.
- **Ultra-Portable**: Compiles to a single static binary. No shared libraries, virtual environments, or complex environments needed.

---

## 🏗️ Why Gollemer?

By eliminating heavy external libraries like **Rust-based backends**, **Python runtimes**, and **Gonum**, Gollemer achieves a state of absolute "Mechanical Sympathy" with the Go compiler and runtime:
- **Zero External Dependencies**: Built entirely from scratch. No need for Python, CGO, Rust toolchains, LLVM, or complex shared libraries. Just clean, native Go.
- **Chromebook-Ready Performance**: Due to its ultra-lean memory footprint and optimized Go-assembly/SIMD vector math, you can run and train Gollemer models directly on a **Chromebook**, basic laptops, or entry-level edge servers. No massive GPU clusters or gigabytes of VRAM required.
- **Instant Deployment**: Compile once, run anywhere. The entire training suite, tokenizers, world knowledge database, and interactive shell deploy as a single static binary.

---

## 🧠 The Adaptive Supervisor (Self-Evolving MoE)

The most advanced capability of Gollemer is its **Adaptive Supervisor** (`internal/ai/moe/supervisor.go`), an autonomous orchestrator that manages a real-time feedback loop during training. Instead of static hyperparameters and a fixed layout, the supervisor actively monitors, heals, and expands the network architecture dynamically.

### 1. Dynamic Config Tuning & Real-Time Reflection
The supervisor continuously reflects on training statistics (`Reflect` / `ReflectSparse`):
*   **Anti-Monopoly Jitter**: If a single expert monopolizes traffic (dominance >85%), the supervisor automatically increases router noise and temperature to force expert exploration and restore healthy specialization.
*   **Plateau Decay**: If training plateaus (perplexity or loss fails to improve for 500–1000 steps), it autonomously decays the global learning rate (`opt.SetLearningRate`) to safely guide weights into a stable minimum.
*   **Confidence Restoration**: If routing confidence falls below a safe threshold (e.g., 18%), it nudges router temperature upward to prevent repetitive "word salad" patterns.
*   **Numerical Safety (Emergency Brake)**: Validates all tensor operations via `SanitizeTensors` to detect and intercept numerical instabilities like `NaN` or `Inf` from failing hardware bridges.

### 2. Autonomous Training Data Evolution
When quality gates or linguistic validations fail on weak or shorthand structures, the supervisor dynamically updates the corpus:
*   **Syntactic Augmentation**: It reads raw training files and mutates weak token fragments (e.g., "hello" or "thanks") on disk into rich grammatical trees containing full Subject-Verb-Object structures (e.g., "i welcome you with hello" or "i offer you my thanks").
*   **Dynamic Sample Weighting**: It reduces the sample weight of grammatically ambiguous training pairs (weight *= 0.5) to prevent them from poisoning the model's loss gradient.
*   **Hot-Injection**: If average similarity falls too low, the supervisor hot-injects highly structured, diverse synthetic conversational patterns directly into the active training loop.

### 3. Self-Directed Expert Spawning, Eviction & Surgery
Gollemer models can dynamically scale their brain capacity during curriculum learning:
*   **Dynamic Spawning**: If an expert combination repeatedly fails (3+ times) under a target intent, the supervisor dynamically spawns a brand new, specialized expert into the MoE layers, automatically extending the gating and noise network weights to handle the new dimensions.
*   **Staggered Triage (Expert Surgery)**: It monitors expert health based on an L2 norm and utilization score. Collapsed or dead experts are triaged and refreshed by cloning the "alpha" (strongest) expert with subtle mutation, while maintaining a ceiling (healing ≤50% per layer) to protect learned sequences.
*   **Active Capacity Eviction**: To prevent memory explosion and out-of-memory (OOM) failures, layers are capped (e.g., 64 experts max). The supervisor evicts the lowest-performing dynamic experts using LRU/health tracking (`EvictLeastActive` or `ForceEvictLowestUtility`), while keeping foundational system experts (E0–E7 representing locked structural grammatical roles like `PRON`, `VERB`, `AUX`) permanently pinned and protected.

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

#### Desktop / x86 (SIMD-accelerated)
| Command | Description |
|---|---|
| `make train` | Cleans old state and starts a fresh Social model training cycle. |
| `make train-social` | Resumes an existing Social model training cycle. |
| `make llm` | Launches the interactive chat shell with the current model. |
| `make clean` | Safely removes current model checkpoints and vocabularies. |

#### Raspberry Pi 3B — Cross-compilation
| Command | Description |
|---|---|
| `make build-pi` | Cross-compiles a 32-bit ARMv7 binary (`gollemer-pi`) for Raspberry Pi OS 32-bit. |
| `make build-pi64` | Cross-compiles a 64-bit ARM64 binary (`gollemer-pi64`) for a 64-bit Pi OS. |

#### Raspberry Pi 3B — On-device Training & Inference
| Command | Description |
|---|---|
| `make pi` | Alias for `make pi-social` — recommended first command on the Pi. |
| `make pi-social` | Resumes social-only curriculum training in Pi 3B safe mode (900 MB cap). |
| `make pi-social-fresh` | Clears existing model and starts a fresh social training run on the Pi. |
| `make pi-chat` | Resumes chat curriculum training in Pi 3B safe mode. |
| `make pi-llm` | Launches the interactive LLM in inference-only mode on the Pi. |

### Configuration Tuning
Model stability is controlled via parameters in the training configuration:

- **`context_multiplier`**: Adjusts how much previous tokens dictate routing (1.5 - 2.0 recommended for deep context).
- **`router_noise`**: Stochastic jitter (0.8+) to ensure all experts are utilized.
- **`k=1`**: Forces each token to a single specialized expert, improving specialization.
- **`max_grad_norm`**: Hard cap (1.0) on updates to prevent mathematical instability.

---

## 🥧 Raspberry Pi 3B Deployment

Gollemer includes first-class support for training and running on a **Raspberry Pi 3B** (~900 MB RAM, ARM Cortex-A53). Pi mode applies automatic constraints — 600 MB memory cap, single-threaded GC, `batch=1`, `accumulate=16`, and 4 experts — to keep the heap safely within the Pi's limits.

> **Note**: `GOEXPERIMENT=simd` and `CGO_ENABLED` are intentionally disabled for Pi targets. The Pi has no x86 SIMD, and cross-compilation with CGO requires a separate toolchain.

### Step 1 — Cross-Compile on Your Workstation

```bash
# 32-bit binary for Raspberry Pi OS (32-bit) — most common
make build-pi

# 64-bit binary for a 64-bit Pi OS (Pi 3B / 4 running arm64)
make build-pi64
```

This produces a `gollemer-pi` (or `gollemer-pi64`) static binary with zero shared-library dependencies.

### Step 2 — Transfer to the Pi

```bash
# Copy binary and required data directory to the Pi
scp gollemer-pi pi@<pi-ip>:~/gollemer/
scp -r data/           pi@<pi-ip>:~/gollemer/
```

### Step 3 — Run on the Pi

```bash
# SSH into the Pi first
ssh pi@<pi-ip>
cd ~/gollemer

# Start a FRESH social training run (clears old model)
make pi-social-fresh

# OR resume an interrupted training run
make pi-social

# Launch the interactive chat (inference only — no -pi flag needed)
make pi-llm
```

### Pi Runtime Limits
| Setting | Value | Reason |
|---|---|---|
| `GOMEMLIMIT` | `700MiB` | Leaves ~200 MB headroom for the OS out of 900 MB total. |
| `GOGC` | `10` | Fires GC very aggressively to keep heap below the limit. |
| `GOMAXPROCS` | `1` (training) / `2` (inference) | Matches the Pi's single high-performance core for training. |
| Memory cap (`-pi`) | `600 MB` | Enforced in code via the `-pi` flag. |
| Batch size | `1` | Prevents OOM during forward/backward passes. |
| Accumulation steps | `16` | Provides an effective batch of 16 without extra memory. |
| Experts per layer | `4` | Reduces model size to fit within Pi constraints. |

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
└── go.mod              # Minimal dependency management (github.com/hegedustibor/htgo-tts for text to voice).
```

---

## 🚪 Exit
- **Interactive Shell**: Type `exit` to shut down.
- **Interrupt**: `Ctrl+C` terminates training or inference loops safely.
```