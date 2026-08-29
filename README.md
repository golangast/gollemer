# Gollemer

Gollemer is a high-performance **Mixture of Experts (MoE)** neural network framework and training pipeline written entirely in **Go**. It is designed for maximum performance with **zero external dependencies**, featuring a native core optimized for SIMD-accelerated CPU training and an autonomous adaptive supervisor.

---

## ⚡ Quick Start

### 1. Installation
Gollemer requires **Go 1.26+** to leverage native SIMD acceleration.

```bash
git clone https://github.com/golangast/gollemer
cd gollemer
export GOEXPERIMENT=simd
go mod tidy
```

### 2. Training the Model
We provide a simplified `Makefile` to handle the curriculum training process.

```bash
make train
make train ARGS='-cartridges="data/models/intents/computer.cartridge" -data "data/training/trainingdata/computer/computer.csv"'
```

### 3. Running the LLM
Chat with your trained model using the interactive shell.

```bash
make chat
# or use 'make llm'
```

---

## 🛠️ Makefile Commands

| Command | Description |
|---|---|
| `make train` | Start fresh curriculum training (clears MoE models, preserves word2vec) |
| `make train-fresh` | Full cold start — clears ALL models including word2vec, then trains |
| `make train-small` | Run small social dataset, print loss + memory, and test the model |
| `make train-small-seq2seq` | Run strict pure Q→A seq2seq tiny demo |
| `make test-small-seq2seq` | Load tiny seq2seq model and probe a few prompts |
| `make seq2seq-prompt PROMPT="hello"` | Send a custom prompt to the saved tiny seq2seq model |
| `make seq2seq-chat` | Start an interactive tiny seq2seq chat loop |
| `make metrics` | Run metrics aggregation and CSV export for edit logs |
| `make export-labels` | Export training examples to CSV for manual labeling |
| `make install-hooks` | Install Gollemer Git pre-commit validation hook |
| `make clean` | Remove MoE model checkpoints (preserves word2vec) |
| `make clean-all` | Remove ALL model files including word2vec |
| `make help` | Display available commands |
| `make conversing-pb` | Convert `conversing.yaml` → `conversing.pb` (protobuf dataset) |

**Usage example with custom data:**
```bash
make train ARGS='-cartridges="data/models/intents/computer.cartridge" -data "data/training/trainingdata/computer/computer.csv"'
```

---

### Small Training (`make train-small`)

`make train-small` runs a compact social-curriculum training loop on the tiny demo dataset (`small_social_demo.pb` or `.csv`). It:

1. Trains a small MoE model using a direct answer-only objective to force rapid loss descent.
2. Prints heap and system memory stats before and after training.
3. Probes the trained model with four fixed prompts (`hello`, `what is your name`, `how are you`, `can you help me`) and prints the generated responses plus latency.

This is useful for quickly verifying that training, inference, and the model loader work end-to-end without running the full curriculum.

---

### Protobuf Datasets

Training data can now be loaded from **protobuf** (`*.pb`) files in addition to CSV. The small-training and seq2seq pipelines prefer `.pb` datasets when available:

- `data/training/trainingdata/conversing.pb` — multi-turn conversations (converted from YAML via `make conversing-pb`)
- `data/training/trainingdata/small_social_demo.pb` — tiny social demo dataset

The proto definitions live under `internal/ai/training/proto/` and `internal/ai/training/proto/dataset/`.

---