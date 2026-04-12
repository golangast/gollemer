# Gollemer

Gollemer is an intelligent coding assistant and project orchestrator designed to help you build Go applications, specifically focusing on web servers and WASM frontends. It understands natural language and can hold a **real conversation** — powered by a custom-built, end-to-end trainable neural network written entirely in Go with no external ML dependencies.

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and ensure you have Go 1.22+ installed:
```bash
git clone github.com/golangast/gollemer
cd gollemer
go mod tidy
```

### 2. Launching the Assistant
To start your journey with the AI mascot, run the following command:
```bash
CGO_ENABLED=0 go run cmd/tools/train_moe/main.go -llm

```

### 3. Training the AI
To train the Mixture-of-Experts (MoE) classifier or the conversational chat model:
```bash
# On Linux/WSL
# Recommended stable run command
./scripts/linuxtrain.sh --gpu --batch-size 4 --acc-steps 16 --epochs 50
```

### 4. GPU Acceleration & Windows Execution
Gollemer now supports **native GPU acceleration** via Gogpu (Gogpu V3). 

On Windows, use the provided helper script to automatically configure the CGO environment and the local compiler:
```powershell
# Run the chat training on GPU
.\run.ps1 -train-chat -gpu

```

> [!IMPORTANT]
> **Windows Requirements**: GPU support requires `CGO_ENABLED=1` and a GCC compiler (provided in `.\tools\mingw64\`). Always use `run.ps1` to ensure the environment is correctly initialized.

---

## 💬 Essential Commands

Gollemer is built for action. Here are the primary commands you can use in the LLM shell.

### 📁 Project Scaffolding
| Command | Example | Description |
|---|---|---|
| `menu` | `menu` | Opens the interactive command menu for all features. |
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

## 🛠️ Performance & Advanced Usage

### ⚙️ SIMD Acceleration
To enable experimental **SIMD acceleration** for neural operations:
```bash
GOEXPERIMENT=simd go run cmd/gollemer/main.go -llm
```

### 🎓 Interactive Tutorial
Gollemer features a built-in interactive tutorial that guides you through the entire workflow of building a web application.
- **To Start:** Type `tutorial` at the prompt.
- **Workflow:** Guides you from folder creation to running a full webserver.
- **Persistence:** Progress is saved in `data/db/gollemer.db`.

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

## 🚪 Exit & Cleanup
- **Exit:** Type `exit` to shut down the mascot and save your session.
- **Halt:** `Ctrl+C` will terminate all background processes and the main loop.
