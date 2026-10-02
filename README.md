# Gollemer 🤖

One chat in the terminal. It reads Go codebases, explains commands, and runs them for you.

Start it:

```sh
make chat
```

Six brains behind one prompt: social chatter, Go concepts, Go code, Go commands, make commands, and the codebase reader (goanalyze). Every reply is tagged with the brain that answered it, e.g. `gollemer [goanalyze]>`.

## 💬 Chat commands

Say any of these inside `make chat`. When the chat answers with a `go`/`gofmt` command or a `make` command, it asks `[run it here? ...] [y/n]` — answer `y` and it runs right there in the terminal.

| Say this | Example | What it does |
|---|---|---|
| Analyze a repo | `analyze this project` (or `analyze ~/path/to/foo`, `analyze github.com/owner/repo`) | Reads the Go code and explains it in plain English — entry points, most-called functions, reading order — plus visuals: package sizes, coupling, biggest files, dependency graph |
| Ask about the code | `what does routeDomain do`, `what's in package chat` | Signature, docs, and who calls it — or a package's purpose and key pieces |
| Read the source | `show me routeDomain` | Prints the actual function |
| Find where to edit | `where would I add a retry helper` | Suggests the right file and why |
| Run one idea end-to-end | `/flow count with a mutex` | Prompt → generated Go → safety check → tuned code → plain-words trace |
| Ask about a Go command | `what does go build ./... do` | Explains the exact command |
| Ask for a Go command | `build with the race detector` | Answers `go build -race ./...`, offers to run it |
| Ask about a project command | `how do i retrain the model` | Answers `run make train-go`, offers to run it |
| Run a project command | `run make eval` or just `eval` | Validates the target and offers to run it right there |
| Explain a project command | `explain make eval` | Says what the target does and shows its recipe, then offers to run it |
| Ask about a Go concept | `what is a goroutine` | Plain-English explanation |
| Ask for Go code | `write a function that reverses a string` | Generates the Go code |
| Review the session | `/history` | Shows what it remembers from this session |
| Start over | `/forget` | Wipes the session memory |
| Peek at its reasoning | `/thoughts` | Toggles the per-token thought process display |
| Leave | `/quit` | Ends the chat |

## ⌨️ Make commands

Every target, with an example and what it does. The chat knows these too — say `run make X` to run one directly, or `explain make X` to hear what it does first; either way it offers to run it.

| Command | Example | What it does |
|---|---|---|
| `make start` | `make start` | Prints the start-here guide with example prompts |
| `make chat` | `make chat` | Talk to Gollemer — one session, six brains, clean replies only |
| `make debug-chat` | `make debug-chat` | Same chat, with the thought process + debug prints shown |
| `make classic ARGS="..."` | `make classic ARGS="-flow -prompt 'count with a mutex'"` | The flag-driven pipeline CLI (`-flow`, `-impact`, `-style`, …) |
| `make flow PROMPT="..."` | `make flow PROMPT="count with a mutex"` | One prompt through the pipeline, in plain words |
| `make sel` | `make sel` | Pick a command from a columnar fuzzy finder |
| `make explain` | `make explain` | Project overview + what each command does |
| `make help` | `make help` | List the commands |
| `make smarter` | `make smarter` | The one-command upgrade: more data → retrained brains → evals → report |
| `make eval` | `make eval` | Score every brain on its fixed eval suite |
| `make train-social` | `make train-social` | Retrain the social conversation brain |
| `make train-go` | `make train-go` | Retrain the Go concept brain |
| `make train-gocode` | `make train-gocode` | Retrain the Go code-generation brain |
| `make train-gocli` | `make train-gocli` | Retrain the Go CLI command brain |
| `make train-makefile` | `make train-makefile` | Retrain the makefile command brain |
| `make import FILE=...` | `make import FILE=my_pairs.jsonl` | Import new training pairs through the quality gate |
