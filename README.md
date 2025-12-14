# gollemer: LLM Command-line Utility

![status beta](https://img.shields.io/badge/Status-Beta-red)
<img src="https://img.shields.io/github/license/golangast/gollemer.svg"><img src="https://img.shields.io/github/stars/golangast/gollemer.svg">[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://endrulats.com)[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)[![GitHub go.mod Go version of a Go module](https://img.shields.io/github/go-mod/go-version/gomods/athens.svg)](https://github.com/golangast/gollemer)[![GoDoc reference example](https://img.shields.io/badge/godoc-reference-blue.svg)](https://pkg.go.dev/github.com/golangast/gollemer/gollemerer)[![GoReportCard example](https://goreportcard.com/badge/github.com/golangast/gollemer)](https://goreportcard.com/report/github.com/golangast/gollemer)[![saythanks](https://img.shields.io/badge/say-thanks-ff69b4.svg)](https://saythanks.io/to/golangast)


### Beta
This project is in beta and changes daily.  I once in a while upload youtube [videos](https://www.youtube.com/watch?v=8paxWwPt4-A&list=PL_sE11fwtBT-0GqVHEX-tYTBzAIGHelQ6) talking about it's changes.





This project provides a command-line utility for interacting with Large Language Model (LLM) functionalities, enabling natural language interaction for various tasks.
### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/golangast/gollemer.git
    
    cd gollemer

    go mod tidy


    ```

-----

## 🛠️ Usage

You may need to retrain the model before using the llm mode.

The main executable (`main.go`) controls all operations using specific command-line flags. All commands should be run from the root directory of the project.

### 1\. Training Models

Use the respective flags to initiate the training process. Each flag executes a separate module located in the `cmd/` directory.

| Model | Flag | Command |
| :--- | :--- | :--- |
| **Word2Vec** | `--train-word2vec` | `go run main.go --train-word2vec` |
| **Mixture of Experts (MoE)** | `--train-moe` | `go run main.go --train-moe` |

<p align="center">
<img src="[URL_OR_PATH_TO_GIF.gif](https://github.com/golangast/gollemer/blob/main/mov/mov.gif)" alt="Description" width="700" height="auto" />

</p>



### 2\. Help / No Action

If no flags are provided, the application will prompt the user to specify an action:

```
$ go run main.go
2025/10/05 07:35:00 No action specified. Use -train-word2vec, -train-moe, -train-intent-classifier, or -moe_inference <query>.
```

-----

## Usage

To run the LLM utility, use the following command from the project root:

```bash
//run the command
go run . -llm 

then type in your command (create file named jim)
"Your natural language command or query here"
```

Replace `"Your natural language command or query here"` with the specific input you want the LLM to process.

## Examples

Here are some examples of commands you can use with the LLM utility:

*   **Information Query:**
    ```bash
    //run commands
    go run . -llm 
    
    "create a webserver jim"
    ```





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