# gollemer: LLM-focused Toolkit

[![Go Report Card](https://goreportcard.com/badge/github.com/golangast/gollemer)](https://goreportcard.com/report/github.com/golangast/gollemer)
[![GoDoc](https://img.shields.io/badge/godoc-reference-blue.svg)](https://pkg.go.dev/github.com/golangast/gollemer)
[![Go Version](https://img.shields.io/github/go-mod/go-version/golangast/gollemer)](https://github.com/golangast/gollemer)
![GitHub top language](https://img.shields.io/github/languages/top/golangast/gollemer)
[![GitHub license](https://img.shields.io/github/license/golangast/gollemer)](https://github.com/golangast/gollemer/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/golangast/gollemer)](#)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/golangast/gollemer)](#)
![GitHub repo size](https://img.shields.io/github/repo-size/golangast/gollemer)
![Status](https://img.shields.io/badge/Status-Beta-red)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/golangast/gollemer/graphs/commit-activity)

`gollemer` is a versatile toolkit primarily focused on integrating and utilizing Large Language Model (LLM) concepts within a high-performance Go environment. This project explores advanced NLP architectures like Mixture of Experts (MoE) and Intent Classification, laying the groundwork for sophisticated semantic understanding and command generation.

> **Note:** This project is currently in a beta stage and is under active development. The API and functionality are subject to change.

## Table of Contents

- [✨ Key LLM-focused Features](#-key-llm-focused-features)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Building from Source](#building-from-source)
- [🛠️ Usage: LLM-related Commands](#️-usage-llm-related-commands)
  - [Training LLM Components](#1-training-llm-components)
  - [Running MoE Inference](#2-running-moe-inference)
  - [LLM-driven Workflow Generation and Execution](#3-llm-driven-workflow-generation-and-execution)
- [🧩 Integrating `gollemer` LLM Components into Your Projects](#-integrating-gollemer-llm-components-into-your-projects)
- [🗺️ Roadmap for LLM Capabilities](#️-roadmap-for-llm-capabilities)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [Why Go for LLMs?](#why-go-for-llms)

## ✨ Key LLM-focused Features

`gollemer` provides core components and utilities for building and experimenting with LLM-like functionalities:

*   **Mixture of Experts (MoE) Architecture**: Implementation of an MoE model, designed for improved performance, scalability, and handling of complex sequential or structural data, a common technique in large language models.
*   **Intent Classification**: Develop a model for accurately categorizing user queries into predefined semantic intents, crucial for understanding user commands in LLM applications.
*   **Semantic Parsing Foundation**: The project is designed with a future direction towards advanced semantic parsing, aiming to translate natural language into structured, executable workflows.
*   **Efficient Go Implementation**: Leveraging Go's performance and concurrency features for fast training and inference of LLM components, suitable for production-grade applications.

## 🚀 Getting Started

### Prerequisites

You need a working **Go environment** (version 1.25 or higher is recommended) installed on your system.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/golangast/gollemer.git
    cd gollemer
    ```

### Building from Source

You can build the executable from the root of the project directory:

```bash
go build .
```

This will create an `gollemer` executable in the current directory.

## 🛠️ Usage: LLM-related Commands

The main executable (`gollemer` or `main.go`) controls all operations using specific command-line flags. All commands should be run from the root directory of the project.

### 1. Training LLM Components

Use the respective flags to initiate the training process for LLM-related components. Each flag executes a separate module located in the `cmd/` directory.

| LLM Component            | Flag                      | Command                                      |
| :----------------------- | :------------------------ | :------------------------------------------- |
| **Mixture of Experts (MoE)** | `--train-moe`             | `go run main.go --train-moe`                 |
| **Intent Classifier**    | `--train-intent-classifier` | `go run main.go --train-intent-classifier`   |

### 2. Running MoE Inference

To run predictions using a previously trained MoE model, use the `--moe_inference` flag and pass the input query string. This demonstrates a practical application of a foundational LLM technique.

| Action          | Flag              | Command Example                                                              |
| :---------------- | :---------------- | :--------------------------------------------------------------------------- |
| **MoE Inference** | `--moe_inference` | `go run main.go --moe_inference "schedule a meeting with John for tomorrow at 2pm"` |

### 3. LLM-driven Workflow Generation and Execution

The `example/main.go` program demonstrates how `gollemer` can parse a natural language query using its internal semantic understanding components (drawing inspiration from LLM principles), generate a workflow, and execute it. This showcases the core capabilities for understanding and acting upon user commands.

To run the example, use the following command with a query:

```bash
go run ./example/main.go -query "create folder jack with a go webserver jill"
```

You can also run it interactively:

```bash
go run ./example/main.go
```
Then, enter queries at the prompt.

Expected Output (for the query "create folder jack with a go webserver jill"):

```
Processing query: "create folder jack with a go webserver jill"

--- Generated Workflow (after inference and validation) ---
Node ID: Filesystem::Folder-jack-0, Operation: CREATE, Resource Type: Filesystem::Folder, Resource Name: jack, Properties: map[permissions:493], Command: , Dependencies: []
Node ID: Filesystem::File-jill-0, Operation: CREATE, Resource Type: Filesystem::File, Resource Name: jill, Properties: map[permissions:493], Command: , Dependencies: [Filesystem::Folder-jack-0]
Node ID: file-createfile-0, Operation: WRITE_FILE, Resource Type: , Resource Name: , Properties: map[], Command: , Dependencies: [Filesystem::File-jill-0]
```

## 🧩 Integrating `gollemer` LLM Components into Your Projects

`gollemer` is designed to be a collection of reusable Go packages. You can integrate its LLM-related components into your own Go projects for tasks requiring semantic understanding, intent recognition, or MoE-based predictions.

Example usage is in the `/example` folder, showcasing how to leverage the parser and workflow executor. The `neural/moe` and `neural/nnu/intent` packages are key entry points for LLM-focused integration.

## 🗺️ Roadmap for LLM Capabilities

This project is under active development with a strong focus on enhancing its LLM capabilities:

-   [ ] **Advanced Semantic Parsing**: Deepening the ability to translate complex natural language into Abstract Semantic Graphs (ASG) or Structured Objects, moving beyond keyword matching to true contextual understanding.
-   [ ] **Integration with External LLMs**: Exploring interfaces for `gollemer` to interact with and leverage external large language models (e.g., via APIs) for enhanced reasoning and generation tasks.
-   [ ] **Generative Capabilities**: Implementing modules for text generation, summarization, or code generation based on learned patterns and semantic understanding.
-   [ ] **Improved Contextual Understanding**: Enhancing models to maintain and utilize conversational context for more natural and accurate interactions.
-   [ ] **Expanded Training Data**: Curating and integrating larger, more diverse datasets specifically tailored for LLM training within the Go ecosystem.
-   [ ] **Modular LLM Architecture**: Further modularizing the LLM components to allow for easier experimentation with different architectures (e.g., transformers, attention mechanisms).
-   [ ] **Comprehensive Benchmarking**: Establishing benchmarks to measure the performance and accuracy of `gollemer`'s LLM components against industry standards.

## 🤝 Contributing

We welcome contributions! Please feel free to open issues for bug reports or feature requests, or submit pull requests for any enhancements, especially those related to LLM capabilities.

## 📜 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

## Why Go for LLMs?

Go is an excellent choice for developing LLM-focused toolkits due to:

*   **Performance**: Go's compilation to native code and efficient garbage collection deliver the speed critical for demanding NLP and LLM workloads.
*   **Concurrency**: Go's goroutines and channels simplify the implementation of concurrent data processing and parallel model training/inference, essential for scaling LLM applications.
*   **Memory Efficiency**: Go provides fine-grained control over memory, which is vital when working with the large models and datasets characteristic of LLMs.
*   **Reliability**: Go's strong typing and robust standard library contribute to building stable and maintainable LLM systems.
*   **Deployment**: Go binaries are statically linked, making deployment of LLM services straightforward and portable across different environments.
*   **Developer Experience**: Go's simplicity, fast compilation times, and excellent tooling enhance developer productivity when building complex LLM systems.