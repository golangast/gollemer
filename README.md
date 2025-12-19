# gollemer: LLM Command-line Utility

![status beta](https://img.shields.io/badge/Status-Beta-red)
<img src="https://img.shields.io/github/license/golangast/gollemer.svg"><img src="https://img.shields.io/github/stars/golangast/gollemer.svg">

### Beta
This project is in beta and changes daily. I once in a while upload youtube [videos](https://youtu.be/KMI8-UXmfi4) talking about its changes.

This project provides a command-line utility for interacting with Large Language Model (LLM) functionalities, enabling natural language interaction for various tasks.

## 🏛️ Architecture Overview

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

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/golangast/gollemer.git
    
    cd gollemer

    go mod tidy
    ```

## 🛠️ Usage

The main executable (`main.go`) provides an interactive LLM mode. However, the project also includes numerous individual modules for specific tasks like model training, data processing, and running demos. These modules are located in the `cmd/` directory and can be run directly.

All commands should be run from the root directory of the project.

### 1. LLM Interactive Mode

The LLM Interactive Mode allows you to interact with the `gollemer` CLI using natural language.

To run the LLM utility in interactive mode, use the following command from the project root:

```bash
go run . -llm
```

Once in interactive mode, you can type natural language commands directly. The CLI will attempt to understand your intent and execute the corresponding action.

#### LLM Interactive Mode Commands

Here are the commands you can use in the interactive LLM mode:

| Command | Description | Example Prompt |
| :------ | :---------- | :------------- |
| **Change Directory** | Navigates to a specified directory. The `gollemer` remembers the last directory you navigated to. | `go to myproject` <br> `cd src` |
| **List Directory Contents** | Lists files, folders, or both in the current directory or a specified path. | `list all files` <br> `list folders` <br> `ls` |
| **Create Folder** | Creates a new directory. | `create folder mydata` <br> `create folder temp in /tmp` |
| **Create File** | Creates a new empty file. | `create file README.md` <br> `create file test.txt in src` |
| **Create Handler** | Generates a new Go HTTP handler function and registers it in `main.go`. | `create handler MyHandler with url /myapi` |
| **Create Webserver** | Scaffolds a new Go web server project in `cmd/<name>/main.go`. | `create webserver MyServer` |
| **Create Database** | Creates a SQLite database file and optionally a table with specified fields. | `create database myappdb` <br> `create database myappdb with the fields name string and age int` |
| **Create Data Structure** | Creates a Go struct file and a corresponding table in a SQLite database (`jim.db`). | `create data structure User with field Name string and Age int` |
| **Delete Folder** | Deletes a specified directory. | `delete folder temp` |
| **Delete File** | Deletes a specified file. | `delete file old.txt` |
| **Print Working Directory** | Prints the current working directory. | `pwd` |

### 2. Individual Command Modules

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

### 3. Help / No Action

If no flags are provided, the application will prompt the user to specify an action:

```
$ go run main.go
2025/10/05 07:35:00 No action specified. Use -train-word2vec, -train-moe, -train-intent-classifier, or -llm.
```

-----

## Special thanks
* [Go Team because they are gods](https://github.com/golang/go/graphs/contributors)

## Why Go?
* The language is done since 1.0.https://youtu.be/rFejpH_tAHM there are little features that get added after 10 years but whatever you learn now will forever be useful.
* It also has a compatibility promise https://go.dev/doc/go1compat
* It was also built by great people. https://hackernoon.com/why-go-ef8850dc5f3c
* 14th used language https://insights.stackoverflow.com/survey/2021
* Highest starred language https://github.com/golang/go
* It is also number 1 language to go to and not from https://www.jetbrains.com/lp/devecosystem-2021/#Do-you-plan-to-adopt--migrate-to-other-languages-in-the-next--months-If-so-to-which-ones
* Go is growing in all measures https://madnight.github.io/githut/#/stars/2023/3
* Jobs are almost doubling every year. https://stacktrends.dev/technologies/programming-languages/golang/
* Companies that use go. https://go.dev/wiki/GoUsers
* Why I picked Go https://youtu.be/fD005g07cU4