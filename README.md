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