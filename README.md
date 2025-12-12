# gollemer: LLM Command-line Utility

This project provides a command-line utility for interacting with Large Language Model (LLM) functionalities, enabling natural language interaction for various tasks.

## Usage

To run the LLM utility, use the following command from the project root:

```bash
go run . -llm "Your natural language command or query here"
```

Replace `"Your natural language command or query here"` with the specific input you want the LLM to process.

## Examples

Here are some examples of commands you can use with the LLM utility:

*   **Information Query:**
    ```bash
    go run . -llm "What is a webserver?"
    ```

*   **File System Operation:**
    ```bash
    go run . -llm "create a file named main.go"
    ```

This utility allows you to interact with the system using natural language, leveraging the power of Large Language Models.