# Gollemer

Gollemer is an intelligent coding assistant and project orchestrator designed to help you build Go applications, specifically focusing on web servers and WASM frontends. It understands natural language and can hold a **real conversation** — powered by a custom-built, end-to-end trainable neural network written entirely in Go with no external ML dependencies.

---

## 💬 Conversational AI — How It Works

Gollemer communicates with you through a dual-mode conversational system. It can:
1. **Understand developer commands** — "create a handler called Login with URL /login"
2. **Hold general conversation** — "how are you?", "what do you think about Go?"

Both modes run through the same neural pipeline: Word2Vec embeddings → MoE Encoder → Intent Classifier → (if chat) RNN Decoder for response generation.

---

## 🛠️ Essential Commands

Gollemer is built for action. Here are the primary commands you can use in the LLM shell (`go run cmd/gollemer/main.go -llm`).

### 📁 Project Scaffolding
| Command | Example | Description |
|---|---|---|
| `create webserver` | `create webserver myapp` | Scaffolds a complete Go project folder with SQLite and HTTP setup. |
| `create handler` | `add handler Login at /login` | Generates a new handler function and registers the route in `main.go`. |
| `create page` | `create page index` | Generates a WASM-ready Go frontend page in the assets folder. |
| `create database` | `make db users` | Initializes an SQLite database file. |
| `create folder` | `mkdir utils` | Creates a new directory. |
| `create file` | `touch config.go` | Creates an empty file. |

### 🚀 Life-Cycle Management
| Command | Example | Description |
|---|---|---|
| `run webserver` | `run webserver jj` | Compiles and executes the specified webserver. |
| `stop webserver` | `stop webserver jj` | Safely terminates a running webserver using its PID file. |
| `watch` | `watch current folder` | Starts a real-time monitor that reacts to file changes and suggests commits. |
| `tutorial` | `tutorial` | Starts an interactive, step-by-step guide (switches to `examples/tutorial`). |

### 🔍 Discovery & Quality
| Command | Example | Description |
|---|---|---|
| `audit` | `audit project` | Performs a deep structural scan for unused code, security leaks, and architecture health. |
| `doctor` | `doctor` | Diagnoses and repairs project structure (misplaced files, missing registrations). |
| `profile` | `show profile` | Displays detailed project health, size, last activity, and structural overview. |
| `quests` | `quest log` | Scans all `.go` files for `TODO:` and `FIXME:` and presents them as a mission log. |
| `list` | `ls cmd/` | Lists files and directories in the specified or current path. |
| `grep` | `search for "func"` | Performs a recursive text search within the project. |

---

## 🎓 Interactive Tutorial

Gollemer features a built-in interactive tutorial that guides you through the entire workflow of building a web application.

- **To Start:** Type `tutorial` at the prompt.
- **Workflow:** Gollemer will guide you through:
    1. Creating a folder.
    2. Scaffolding a webserver.
    3. Adding logic.
    4. Running and testing your server.
- **Persistence:** Your tutorial progress is saved in `data/db/gollemer.db`, so you can resume even after restarting the app.

---

## 🏗️ Project Structure

Gollemer follows a clean, industry-standard directory layout:

```text
.
├── cmd/                # Entry points for the application and tools
│   ├── gollemer/       # Main assistant entry point
│   └── tools/          # Neural training and visualization utilities
├── internal/           # Private application logic
│   ├── ai/             # Neural network (MoE, RNN, Word2Vec)
│   ├── platform/       # Infrastructure (UI, DB, Watcher, Discovery)
│   └── util/           # Common utilities (Colors, File IO)
├── data/               # Persistent data and models
│   ├── models/         # Trained .gob and checkpoint files
│   ├── training/       # CSV and TXT training datasets
│   └── db/             # Project profiles and tutorial state (SQLite)
├── examples/           # Tutorial content and sample projects
└── logs/               # Training logs and profiling data
```

---

## 🧠 Neural Architecture

Gollemer's conversational brain is an **encoder-decoder Seq2Seq architecture** with a Mixture-of-Experts encoder.

### 🔀 MoE Encoder — Sparse Mixture of Experts
The encoder uses a **Sparse MoE Layer** with 8 specialized experts. Each token is routed to the **top-2** most relevant experts using a noisy gating network, ensuring both specialization and discovery.

### 📡 RNN Decoder — Seq2Seq Response Generation
For conversational replies, a 2-layer LSTM decoder with **Cross-Attention** generates tokens by attending to the encoder's context vector.

---

## 🗣️ Training & Models

You can train the AI components using the following commands:

- **Train Chat:** `go run cmd/gollemer/main.go -train-chat` (Seq2Seq conversation)
- **Train MoE:** `go run cmd/gollemer/main.go -train-moe` (Intent classification)
- **Train Word2Vec:** `go run cmd/gollemer/main.go -train-word2vec` (Word embeddings)

### ⚙️ Performance
Gollemer supports **SIMD acceleration** for neural operations. To enable it:
```bash
GOEXPERIMENT=simd go run cmd/gollemer/main.go -llm
```

---

## 🚪 Exit & Cleanup
- **Exit:** Type `exit` to shut down the mascot and save your session.
- **Halt:** `Ctrl+C` will terminate all background processes and the main loop.
