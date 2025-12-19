# Gollemer: LLM Command-line Utility

[![Status](https://img.shields.io/badge/Status-Beta-red)](https://github.com/golangast/gollemer)
[![GitHub License](https://img.shields.io/github/license/golangast/gollemer)](https://github.com/golangast/gollemer/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/golangast/gollemer)](https://github.com/golangast/gollemer)

> An LLM-powered command-line utility to streamline your development workflow.

This project is in beta and changes daily. I once in a while upload youtube [videos](https://youtu.be/KMI8-UXmfi4) talking about its changes.

## Table of Contents

- [Gollemer: LLM Command-line Utility](#gollemer-llm-command-line-utility)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
    - [Architecture Overview](#architecture-overview)
  - [Built With](#built-with)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [LLM Interactive Mode](#llm-interactive-mode)
      - [LLM Interactive Mode Commands](#llm-interactive-mode-commands)
    - [Individual Command Modules](#individual-command-modules)
      - [Training Commands](#training-commands)
      - [Data Generation \& Handling](#data-generation--handling)
      - [Inference \& Prediction](#inference--prediction)
      - [Demos \& Examples](#demos--examples)
      - [Utilities](#utilities)
  - [Demos](#demos)
  - [Roadmap](#roadmap)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## About The Project

`gollemer` is a command-line utility that leverages the power of Large Language Models (LLMs) to provide a natural language interface for various development tasks. From scaffolding projects to managing files and training machine learning models, `gollemer` aims to make your workflow more intuitive and efficient.

### Architecture Overview

The `gollemer` CLI processes natural language user input through an LLM interface, which then dispatches commands to a robust command processor. This processor leverages a sophisticated Natural Language Processing (NLP) pipeline for understanding user intent and entities. Based on this understanding, it orchestrates either code generation tasks or direct file system operations.

```
+-------------------+       +--------------------+       +---------------------+
|   User Input      | ----> |  LLM Interface     | ----> |  Command Processor  |
| (Natural Language)|       | (Text/Voice)       |       |                     |
+-------------------+       +--------------------+       +----------+----------+
                                                                     |
                                                                     v
+---------------------+    +--------------------+    +-----------------------+
|  NLP Pipeline       | <--| Intent Classifier  |    |  Code Generation      |
| (Intent, NER,        |    | (MoE Model)        |    |  (Templates, DB, etc.)|
| Semantic Parsing)   |    +--------------------+    +-----------------------+
+----------+----------+
           |
           v
+-------------------------+
|  File System Operations |
| (Create, Delete, List)  |
+-------------------------+
```

## Built With

* [Go](https://golang.org/)
* [SQLite](https://www.sqlite.org/index.html)
* [ModernC SQLite Driver](https://modernc.org/sqlite/)

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* Go 1.18 or later
* `goimports` tool
  ```sh
  go install golang.org/x/tools/cmd/goimports@latest
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/golangast/gollemer.git
   ```
2. Change into the project directory
    ```sh
    cd gollemer
    ```
3. Tidy the modules
    ```sh
    go mod tidy
    ```

## Usage

The main executable (`main.go`) provides an interactive LLM mode. However, the project also includes numerous individual modules for specific tasks like model training, data processing, and running demos. These modules are located in the `cmd/` directory and can be run directly.

All commands should be run from the root directory of the project.

### LLM Interactive Mode

The LLM Interactive Mode allows you to interact with the `gollemer` CLI using natural language.

To run the LLM utility in interactive mode, use the following command from the project root:

```bash
go run . -llm
```

Once in interactive mode, you can type natural language commands directly. The CLI will attempt to understand your intent and execute the corresponding action.

#### LLM Interactive Mode Commands

Here are the commands you can use in the interactive LLM mode:

| Command | Description | Example Prompt |
| :--- | :--- | :--- |
| **Change Directory** | Navigates to a specified directory. Aliases: `go to`, `cd`. The last visited directory is remembered. | `go to myproject`<br>`cd src` |
| **List Contents** | Lists files, folders, or both in the current or a specified directory. Aliases: `ls`, `list`. | `list all files`<br>`list folders in /tmp`<br>`ls` |
| **Create Folder** | Creates a new directory. Aliases: `create directory`. | `create folder mydata`<br>`create directory temp in /tmp` |
| **Create File** | Creates a new empty file. | `create file README.md`<br>`create file test.txt in src` |
| **Create Handler**| Generates a new Go HTTP handler function. Can be registered to a webserver's `main.go` file with a specific URL. | `create handler MyHandler`<br>`create handler Auth with url /auth in main.go`|
| **Create Webserver**| Scaffolds a new Go web server project in `cmd/<name>/main.go`. | `create webserver MyServer` |
| **Create Database**| Creates a SQLite database file and optionally a table with specified fields. | `create database myappdb`<br>`create database users with the fields name string and age int`|
| **Create Data Structure**| Creates a Go struct, a corresponding SQLite database and table, and `Create`/`Update`/`Delete` HTTP handlers. | `create data structure User with field Name string and Age int` |
| **Delete Folder** | Deletes a specified directory and its contents. Aliases: `delete directory`. | `delete folder temp` |
| **Delete File** | Deletes a specified file. | `delete file old.txt` |
| **Run Webserver** | Builds and runs a specified webserver from the `cmd/` directory. | `run webserver MyServer` |
| **Stop Webserver**| (Currently disabled) Stops a running webserver. This is handled by the webserver itself. | `stop webserver MyServer` |
| **Print Directory**| Prints the current working directory. Aliases: `pwd`. | `pwd` |
| **Clear Screen** | Clears the terminal screen. Aliases: `clear`. | `clear` |
| **Exit** | Exits the interactive LLM mode. Aliases: `exit`. | `exit` |

### Individual Command Modules

The following modules in the `cmd/` directory can be run individually.

#### Training Commands

| Command | Description |
| :--- | :--- |
| `go run ./cmd/train_intent_classifier` | Trains the intent classification model. |
| `go run ./cmd/train_intent_model` | Trains the intent model. |
| `go run ./cmd/train_moe` | Trains the Mixture of Experts (MoE) model. |
| `go run ./cmd/train_ner` | Trains the Named Entity Recognition (NER) model. |
| `go run ./cmd/train_seq2seq` | Trains a sequence-to-sequence model. |
| `go run ./cmd/train_simple_intent_classifier` | Trains a simpler intent classification model. |
| `go run ./cmd/train_tagger` | Trains a generic tagger model. |
| `go run ./cmd/train_word2vec` | Trains the Word2Vec model. |

#### Data Generation & Handling

| Command | Description |
| :--- | :--- |
| `go run ./cmd/convert_qa_to_semantic` | Converts question-answering data to a semantic format. |
| `go run ./cmd/create_vocab` | Creates a vocabulary file. |
| `go run ./cmd/csv_feed_generator` | Generates a CSV feed from a data source. |
| `go run ./cmd/generate_training_data` | Generates training data. |
| `go run ./cmd/generate_wikiqa_intents` | Generates intents from the WikiQA dataset. |
| `go run ./cmd/inspect_vocab` | Inspects a vocabulary file. |
| `go run ./cmd/prepare_tagging_data` | Prepares data for tagging models. |
| `go run ./cmd/transform_intents_to_seq2seq`| Transforms intent data into a sequence-to-sequence format. |

#### Inference & Prediction

| Command | Description |
| :--- | :--- |
| `go run ./cmd/debug_inference` | Runs inference in debug mode. |
| `go run ./cmd/moe_inference` | Runs inference with the MoE model. |
| `go run ./cmd/predict_seq2seq` | Makes predictions with a sequence-to-sequence model. |

#### Demos & Examples

| Command | Description |
| :--- | :--- |
| `go run ./cmd/advanced_demo` | Runs an advanced demonstration. |
| `go run ./cmd/command_structure_demo` | Demonstrates the command structure. |
| `go run ./cmd/hierarchical_demo` | Runs a demonstration of hierarchical intents. |
| `go run ./cmd/moe_example` | Shows an example of the MoE model. |
| `go run ./cmd/vfs_demo` | A demo of the virtual file system. |

#### Utilities

| Command | Description |
| :--- | :--- |
| `go run ./cmd/check_token_length` | Checks the token length of inputs. |
| `go run ./cmd/create_handler` | Creates a new request handler. |
| `go run ./cmd/delete_item` | Deletes an item. |
| `go run ./cmd/interactive_scaffolder` | Starts an interactive scaffolding tool. |
| `go run ./cmd/multi_orchestrator` | Runs the multi-command orchestrator. |

## Demos

Check out these videos to see `gollemer` in action:

*   [YouTube: gollemer LLM Command-line Utility](https://www.youtube.com/watch?v=KMI8-UXmfi4)
*   [Local Demo Video](mov/mov.webm) (Note: This is a local file and will not play on GitHub)

## Roadmap

See the [open issues](https://github.com/golangast/gollemer/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Contact

Discord: [https://discord.gg/3CHFDCvG](https://discord.gg/3CHFDCvG)

Project Link: [https://github.com/golangast/gollemer](https://github.com/golangast/gollemer)

## Acknowledgements

* [Go Team](https://github.com/golang/go/graphs/contributors)
