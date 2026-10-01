# Gollemer 🤖

A tiny LLM written in **pure Go** — zero external dependencies.
One chat, six brains: every message is routed to the right one.

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                                make chat                                 │
 │                                                                          │
 │ you> analyze this project                                                │
 │            │                                                             │
 │            ▼                                                             │
 │      ┌─────────────┐                                                     │
 │      │   router    │─── which brain should answer?                       │
 │      └──────┬──────┘                                                     │
 │             │                                                            │
 │       ┌─────────┬─────────┬─────────┬──────────┬──────────┐                │
 │       ▼         ▼         ▼         ▼          ▼          ▼                │
 │   ┌──────┐ ┌────────┐ ┌───────┐ ┌───────┐ ┌────────┐ ┌──────────┐            │
 │   │social│ │  go    │ │gocode │ │ gocli │ │makefile│ │goanalyze │            │
 │   │ chat │ │concepts│ │writes │ │English│ │"how do │ │reads Go  │            │
 │   │      │ │        │ │Go code│ │→go cmd│ │ i..."  │ │codebases │            │
 │   └──────┘ └────────┘ └───────┘ └───────┘ └────────┘ └──────────┘            │
 │                          │                                               │
 │                          ▼                                               │
 │   gollemer [goanalyze]> Project: github.com/golangast/gollemer           │
 └──────────────────────────────────────────────────────────────────────────┘
```

The tag (`[go]`, `[social]`, …) always shows which brain answered.

---

## 🚀 Start here

```sh
make start     # the start-here guide (what to do, what it can do, how to expand it)
make chat      # talk to it — clean replies only
make debug-chat # talk to it with the thought process + debug prints shown
```

Try one prompt from each brain:

| Try saying…                              | Brain      | What happens                              |
|------------------------------------------|------------|-------------------------------------------|
| `have you ever played soccer`            | social     | chit-chat, remembers the session          |
| `what is a goroutine`                    | go         | explains the Go concept                   |
| `write a function that sums a slice`     | gocode     | writes Go code (never chats)              |
| `what command formats my code`           | gocli      | gives the exact `gofmt` command, asks [y/n] before running |
| `how do i chat with gollemer`            | makefile   | suggests `run make chat`                  |
| `analyze this project`                   | goanalyze  | maps the codebase — packages, engine room, reading order (see below) |

```sh
make sel       # browse every command in a fuzzy finder (see below)
make explain   # full project overview + command reference
```

The gocli brain covers the full `go` toolchain (per
[pkg.go.dev/cmd/go](https://pkg.go.dev/cmd/go)): `build`, `run`, `test`,
`vet`, `fmt`/`gofmt`, `mod`, `get`, `install`, `list`, `clean`, `doc`,
`env`, `version`, `generate`, `fix`, `work`, `tool`, `bug`, and
`telemetry` — ask in plain English (`build with the race detector`,
`turn off go telemetry`, `run just the TestLogin tests`) and it answers
with the exact command. It also knows the
[Go by Example](https://gobyexample.com/) catalog: ask `what is a
waitgroup` for the concept or `show me a waitgroup example` for the
code.

## 💬 Chat commands — say this, get that

Every command below works inside `make chat`. When the chat answers
with a `go`/`gofmt` command or a `make` command, it asks
`[run it here? ...] [y/n]` — answer `y` and it runs right there in
the terminal, no shell involved.

| Say this | Example | What it does |
|---|---|---|
| Analyze a repo | `analyze this project` | Reads the repo's Go code and explains it in plain English: entry points, most-called functions, suggested reading order |
| Analyze another repo | `analyze ~/projects/foo` or `analyze github.com/owner/repo` | Clones (if a URL) and explains that codebase instead |
| See the structure | `show me a visual` | Draws the package dependency graph as ASCII in the terminal |
| Ask about the code | `what does routeDomain do` | Signature, docs, and who calls it |
| Read the source | `show me routeDomain` | Prints the actual function |
| Find where to edit | `where would I add a retry helper` | Suggests the right file and why |
| Tour a package | `what's in package chat` | What the package is for, its key pieces |
| Run one idea end-to-end | `/flow count with a mutex` | Prompt → generated Go → safety check → tuned code → plain-words trace |
| Ask about a Go command | `what does go build ./... do` | Explains the exact command |
| Ask for a Go command | `build with the race detector` | Answers `go build -race ./...`, offers to run it |
| Ask about a project command | `how do i retrain the model` | Answers `run make train-go`, offers to run it |
| Ask about a Go concept | `what is a goroutine` | Plain-English explanation |
| Ask for Go code | `write a function that reverses a string` | Generates the Go code |
| Review the session | `/history` | Shows what it remembers from this session |
| Start over | `/forget` | Wipes the session memory |
| Peek at its reasoning | `/thoughts` | Toggles the per-token thought process display |
| Leave | `/quit` | Ends the chat |

The [goanalyze](#-goanalyze--point-it-at-any-go-codebase) and
[flow](#-flow--one-command-through-the-whole-pipeline-in-plain-words)
sections below show longer worked examples of the same commands.

---

## 🔍 goanalyze — point it at any Go codebase

The sixth brain doesn't guess — it **reads**. It parses real Go source
with the standard library's `go/ast`, builds a call graph, and tells you
how a project works and exactly where to change it. It's deterministic:
no neural net, so it can't hallucinate structure that isn't there.

```
 you say                          →  gollemer does
 ─────────────────────────────────────────────────────────────────
 analyze this project               maps the repo you're chatting in
 /analyze ~/path/to/project         maps a folder on disk (~ works)
 analyze ~/path/to/project          same, without the slash
 analyze github.com/owner/repo      shallow-clones it (cached) and maps it
 where would I add a retry helper   ranked file:line hits, using the last
                                    analyzed project
 analyze this project and           everything above, plus the dependency
   show me a visual                 graph drawn right in the terminal
 show me a visual                   same graph, after any analysis above
                                    (uses the last analyzed project)

Then just **ask questions** about the code — it answers from what it
actually parsed, with file:line pointers:

 you> what does routeDomain do      signature, doc, callers, callees
 you> how does NewTensor work       step-by-step through what it calls
 you> what calls NewTensor          every caller, with file:line
 you> where is Project defined      definition site
 you> show me routeDomain           the actual source code
 you> what's in package chat        what the package is for + key functions
 you> where is saving handled       keyword search → ranked file:line hits
```

Questions only trigger when they name something really in the project,
so `what does a goroutine do` still goes to the Go brain and
`what does make chat do` still goes to the makefile brain.

Point it at itself and you get this (real output):

```
you> analyze this project
gollemer [goanalyze]> Project: github.com/golangast/gollemer
10 packages, 85 Go files, ~20278 lines
Entry points:
  - cmd/tools/goz/main.go
  - main.go
  - scripts/stats_viewer.go
22 test files

IN PLAIN ENGLISH
gollemer is a runnable Go program: 10 packages, about 20,278 lines of
Go. It starts in main.go. The hardest-working function is
tensor.NewTensor, called from 62 places — that's the engine room, the
best place to look when something breaks. Read the code in this order:
goz → analyze → vocab → tensor → nn → tokenizer, because each layer
builds on the one before it.

ENGINE ROOM (most-called functions)
████████████████████ tensor.NewTensor (62 callers)
████ analyze.Func.Display (13 callers)
████ analyze.Project.Answer (11 callers)
███ vocab.Vocabulary.AddToken (9 callers)
███ nn.NewLinear (8 callers)
...

WHERE TO START
1. Start at the entry point:  main.go:13 — func main
2. Then read the engine room:  tensor.NewTensor (62 callers)
3. Learn the key types:        tensor.Operation — interface, 42 implementers
4. Package reading order (dependencies first):
   1. cmd/tools/goz
   2. internal/ai/analyze
      Package analyze reads a Go project off disk and builds a
      structural map of it: packages, exported API, a heuristic call
      graph, and the import graph.
   3. internal/ai/neural/nnu/vocab
   4. internal/ai/neural/tensor
   ...
```

And the map it drew of itself — bigger node = more packages depend on it,
red = entry point, arrows run importer → imported:

![gollemer package dependency map](docs/codebase-map.png)

Ask for a visual and it draws the dependency graph right in the
terminal — no browser needed:

```
you> show me a visual
gollemer [goanalyze]> Dependency graph: github.com/golangast/gollemer
A → B means A imports B. ★ = entry point (func main).

★ 📦 cmd/gollemer-classic
  ├── → pkg/analysis
  ├── → pkg/ast
  ├── → pkg/engine
  ...
```

A → B means A imports B; ★ marks entry points (func main).

Follow-ups reuse the last analyzed project, so you can drill in:

```
you> where would I add a new brain
gollemer [goanalyze]> WHERE TO CHANGE for "add a new brain"
1. struct chat.domainTrainData  internal/ai/training/chat/real_seq2seq_train.go:59
2. func chat.routeDomain         internal/ai/training/chat/dataset.go:541  (5 callers)
3. func chat.loadRealModel       internal/ai/training/chat/real_seq2seq_chat.go:120
...
```

And once you've found a name, interrogate it — this is the part that
makes a new codebase editable instead of just mappable:

```
you> what does routeDomain do
gollemer [goanalyze]> func routeDomain(input string) string
  internal/ai/training/chat/dataset.go:541
  routeDomain classifies a chat INPUT (not the output) into the domain
  model that should handle it.
  Calls (2): chat.isGoCliRequest, chat.isGoCommandQuestion
  Called by (5): chat.runUnifiedChat, ...

you> how does NewTensor work
gollemer [goanalyze]> func NewTensor(shape []int, data []float32,
                                     requiresGrad bool) *Tensor
  internal/ai/neural/tensor/tensor.go:293
  NewTensor creates a new Tensor with the given shape and optional
  data on the CPU.
  Called by (62): seq2seq.Encoder.Forward, seq2seq.TrainBatch, ...

you> show me routeDomain
gollemer [goanalyze]> chat.routeDomain  (internal/ai/training/chat/dataset.go:541)
  ```go
  func routeDomain(input string) string {
      // Definition questions about Gollemer itself ("what is gollemer",
      // "who made you") are conversational, not make-command requests,
      // even though they name Gollemer. Checked before makefileIntent.
      if gollemerDefinition.MatchString(input) {
          return SocialDomain
      }
      ...
  ```
```

Notes:

- `show me a visual` draws the dependency graph as ASCII, right in the
  terminal — nothing to open, nothing to serve.
- GitHub repos are cloned once into `~/workspace/codebase-maps/` and
  reused on later runs.
- The call graph is name-based (heuristic) — great for finding hot spots
  and navigation paths, not compiler-grade call resolution.

---

## 🌊 flow — one command through the whole pipeline, in plain words

`flow` ties Go's small language and AST tooling to the ease of an LLM:
one plain-English prompt in, and out comes the Go program, a one-line
analogy, a step-by-step walk of what it does, and the safety and speed
verdicts — no jargon. It runs the reactive engine (`pkg/engine`):
synthesis → symbolic safety proving → auto-tuning → visual trace.

Three ways to run it:

```sh
make flow PROMPT="count with a mutex"        # terminal
go run ./cmd/gollemer-classic -flow -prompt "count with a mutex"
```

```
/flow count with a mutex                      # inside make chat
```

Real output:

````
Here's your Go program:

```go
package main

import (
	"fmt"
	"sync"
)

func main() {
	var mu sync.Mutex
	count := 0
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			mu.Lock()
			count++
			mu.Unlock()
		}()
	}
	wg.Wait()
	fmt.Println("count:", count)
}
```

Think of it like this: An independent worker at a factory line:
lightweight and managed by Go's runtime scheduler.

What it does, step by step:
1. Main

✅ Safety: checked — no unchecked nil dereferences, no leaked resources.
⚡ Speed: already lean — the tuner found nothing to trim.
````

It can build worker pools,
mutex counters, file I/O, HTTP servers, JSON, tickers, …) — anything
else gets a plain-English "I don't know how to build that yet" listing
what it can do. `/flow` runs in-process in the chat; `make flow` goes
through the classic CLI.

---

## 🧪 The pkg/ engine room — how to run it all

The heavy lifting lives in `pkg/` as plain library packages. Two ways
to run them: their test suites, or the classic CLI that wires them
together.

### 1. The test suites (runs everything)

```sh
go test ./pkg/...          # all nine packages
go test ./pkg/engine/ -v   # one package, verbose
```

All nine pass. This is also the fastest way to exercise `pkg/engine`
(the reactive NL → AST → prove → tune → trace pipeline), which has no
CLI of its own yet — see `TestExecutePipelineWorkerPool`.

### 2. The classic CLI (local only)

`cmd/gollemer-classic` stitches the packages into runnable commands.
It isn't pushed yet — these run from your working tree:

```sh
# Full pipeline demo with a scripted mock LLM (no API key needed):
# graph → symbolic proving → MCTS sandbox eval → self-heal → auto-tune
go run ./cmd/gollemer-classic -mock
```

Real output (trimmed):

```
demo: full pipeline — graph → prove → MCTS → self-heal → auto-tune → commit
🔍 Resolving Graph
   [graph] 3 nodes, vector search + depth-2 expansion -> 3 context chunks
🧠 Drafting 3 candidates
🛡️ Symbolic Proving
   candidate-0: clean ✅
   candidate-1: clean ✅
   candidate-2: clean ✅
   3/3 candidates survived proving
🧪 MCTS Parallel Sandboxes [3/3]
   winner: candidate-0  fitness=20.0  allocs/op=0  ns/op=0
🔧 Self-Healing
   healed on attempt 1 ✅
⚡ Auto-Tuning Memory [iter 1/3]: trying prealloc-append...
   ...
```

```sh
# pkg/ast: which files would need updating if you changed a symbol?
go run ./cmd/gollemer-classic -impact -symbol "BuildGraph" -dir ./pkg/memory
# [impact] Symbol "BuildGraph" modified -> 2 dependent files identified
#   cmd/gollemer-classic/pipeline.go:222: BuildGraph
#   pkg/memory/indexer.go:97: BuildGraph
```

```sh
# pkg/ast: infer a repo's coding conventions (for generation prompts)
go run ./cmd/gollemer-classic -style -dir ./pkg/memory
# error handling: fmt_wrap
# struct tags:    [json gob]
# uses context:   false
# doc density:    0.60
```

### 3. Per-package cheat sheet

| Package | What it does | How to run it |
|---------|--------------|---------------|
| `pkg/ast` | Load a Go module into a `CodebaseContext`; chunk files; impact radius; repo style | `-impact`, `-style` above; `go test ./pkg/ast/` |
| `pkg/analysis` | Symbolic safety proving: nil derefs, resource leaks, slice bounds (soundy, not sound) | `-mock` proving stage; `go test ./pkg/analysis/` |
| `pkg/memory` | Knowledge graph: chunk → embed → index → query a codebase | `-mock` graph stage; `go test ./pkg/memory/` |
| `pkg/runner` | Sandboxed `go build`/`go test`/benchmark of generated code, transactional patch apply | `-mock` MCTS stage; `go test ./pkg/runner/` |
| `pkg/synthesis` | MCTS candidate ranking + fitness; allocation optimizer | `-mock` stages; `go test ./pkg/synthesis/` |
| `pkg/engine` | Reactive pipeline: NL → AST → prove → tune → trace | tests only: `go test ./pkg/engine/ -v` |
| `pkg/xray` | Execution-path tracing + concept analogies | used by `pkg/engine`; `go test ./pkg/xray/` |
| `pkg/beginner` | Plain-English code generation templates | powers `/flow`; `go test ./pkg/beginner/` |

## 🎯 `make sel` — the command picker

Don't remember the command names? `make sel` lists every target in
columns. Start typing to fuzzy-filter, pick with the arrow keys, and
it runs:

```
 ┌─ make sel ─────────────────────────────────────┐
 │ > cha                                          │
 │   chat            Talk to gollemer…            │
 │   train-chat      …                            │
 └────────────────────────────────────────────────┘
         │ you pick "chat"
         ▼
   runs: go run . -real-chat -domain unified
```

It reads the `## target: description` comments straight out of the
`Makefile`, so the list is never out of date.

---

## ⌨️ Make commands

Every target, with an example and what it does. The chat knows these
too — ask `how do i X` inside `make chat` and it answers with the
command and offers to run it.

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

---

## 🧠 How it learns

```
 ┌──────────────┐
 │ you write    │
 │ Q&A pairs    │   {"input": "what is a slice",
 │ (JSONL)      │    "output": "a dynamic view into an array...",
 │              │    "domain": "go"}
 └──────┬───────┘
        │ make import FILE=pairs.jsonl
        ▼
 ┌──────────────┐
 │ quality gate │─── bad pairs quarantined (never trained on)
 └──────┬───────┘
        │ good pairs admitted
        ▼
 ┌──────────────────────────────┐
 │ data/training/chat_pairs.jsonl │  ← the one dataset (1,800+ pairs)
 └──────┬───────────────────────┘
        │ make train-go / make smarter
        ▼
 ┌──────────────┐     ┌──────────────┐
 │  retrained   │────▶│  make eval   │─── did the new knowledge stick?
 │  brain (.gob)│     │  fixed suite │
 └──────────────┘     └──────────────┘
```

Four safety nets keep answers reliable:

- **Recall layers** — exact training questions get their trained answer
  verbatim (social + makefile). The tiny net doesn't have to memorize.
- **Go knowledge base** — ~50 curated Go facts answer before the neural
  net is even asked.
- **goanalyze is deterministic** — codebase questions are answered by
  parsing real ASTs, never by a model. Structure can't be hallucinated.
- **Mode separation** — the code brain can never chat and the chat brain
  can never emit code. No `I am func doing (x)` mush, by construction.

---

## 📁 Project layout

```
gollemer/
├── main.go                     # entry point: -real-chat, -train-*, -import
├── Makefile                    # all commands (make help / make sel / make start)
├── README.md                   # this file
├── docs/codebase-map.png       # dependency map shown in this README
├── go.mod                      # stdlib-only core (+ x/tools, x/sync, x/mod — approved)
│
├── scripts/
│   ├── start.sh                # make start — the start-here guide
│   ├── explain.sh              # make explain — project + command reference
│   ├── smarter.sh              # make smarter — the one-command upgrade
│   ├── *_eval_run.py           # fixed eval suites per brain
│   └── *_eval_cases.json       # eval prompts + expected answers
│
├── cmd/
│   ├── gollemer-classic/       # flag-driven pipeline CLI (-flow, -impact, -style, …)
│   └── tools/
│       ├── goz/                # the fuzzy finder behind make sel
│       ├── moe_inference/      # run a model from the CLI
│       └── ...                 # one-off data + debug utilities
│
├── pkg/
│   ├── beginner/               # plain-English code generation templates (powers /flow)
│   ├── ast/                    # codebase contexts: load, chunk, impact, style
│   ├── analysis/               # symbolic safety proving (nil, leaks, bounds)
│   ├── memory/                 # knowledge graph: index + query the codebase
│   ├── runner/                 # sandboxed build/test/benchmark of code
│   ├── synthesis/              # MCTS candidate ranking, allocation optimizer
│   ├── engine/                 # reactive pipeline: NL → AST → prove → tune → trace
│   ├── xray/                   # execution tracing + concept analogies
│
├── internal/ai/
│   ├── analyze/                # 🔍 goanalyze: AST codebase reader (deterministic)
│   │   ├── analyze.go          # parses Go source into packages/functions/types
│   │   ├── deep.go             # plain-English summaries, signatures, overviews
│   │   ├── qa.go               # answers "what does X do" about the code
│   │   ├── graph.go            # heuristic call graph + engine-room ranking
│   │   ├── report.go           # summary / reading guide / where-to-change
│   │   ├── ascii.go            # terminal dependency-graph renderer
│   ├── moe/                    # mixture-of-experts layers (pure Go)
│   ├── neural/
│   │   ├── nn/                 # tensors, autograd, optimizers
│   │   ├── nnu/seq2seq/        # the seq2seq model
│   │   └── nnu/vocab/          # tokenizer + vocabulary
│   ├── training/chat/          # ⭐ the brains live here
│   │   ├── real_seq2seq_chat.go  # chat loops (unified + single-domain)
│   │   ├── real_seq2seq_train.go # training (dims, epochs, checkpoints)
│   │   ├── dataset.go            # router: which brain gets this message?
│   │   ├── goknowledge.go        # curated Go facts (answered first)
│   │   ├── socialrecall.go       # exact-match answers: social
│   │   ├── makefilerecall.go     # exact-match answers: makefile
│   │   ├── conversation.go       # session memory (/history, /forget)
│   │   ├── goanalyze.go          # goanalyze brain: analyze paths/URLs, /analyze, terminal visuals
│   │   ├── flow.go               # /flow command: engine pipeline in-process, plain words
│   │   └── *_test.go             # router, recall, knowledge, routing tests
│   └── ...
│
└── data/
    ├── training/
    │   ├── chat_pairs.jsonl      # ⭐ the one dataset — every brain's Q&A
    │   └── *_20260929.jsonl      # expansion batches (imported via quality gate)
    └── models/
        └── gob_models/
            ├── real_tiny_seq2seq_social.gob     # chat brain
            ├── real_tiny_seq2seq_go.gob         # Go concepts brain
            ├── real_tiny_seq2seq_gocode.gob     # Go code brain
            ├── real_tiny_seq2seq_gocli.gob      # Go CLI brain
            └── real_tiny_seq2seq_makefile.gob   # makefile brain
```

### Where to look for what

| You want to…                        | Look at                                          |
|-------------------------------------|--------------------------------------------------|
| Change how messages get routed      | `internal/ai/training/chat/dataset.go` (`routeDomain`) |
| Change what a brain knows           | `data/training/chat_pairs.jsonl` + `make import` |
| Fix a wrong deterministic answer    | `goknowledge.go`, `socialrecall.go`, `makefilerecall.go` |
| Change how codebases get analyzed    | `internal/ai/analyze/` (`report.go` renders the answers) |
| Tune training (dims, epochs)        | `real_seq2seq_train.go` (`dimsForDomain`)        |
| Change the chat UI (`/history`, …)  | `real_seq2seq_chat.go`, `conversation.go`        |
| Change `/flow` rendering        | `pkg/engine/render.go` (shared by chat + CLI) |
| Add a command                       | `Makefile` (`## name: description` — `sel` picks it up automatically) |

---

## 🔧 Teach it something new

```sh
# 1. write pairs (one JSON object per line)
cat > my_pairs.jsonl << 'EOF'
{"input": "what is a map in go", "output": "a map is go's hash table: keys to values...", "domain": "go"}
EOF

# 2. import through the quality gate
make import FILE=my_pairs.jsonl

# 3. retrain the brain
make train-go

# 4. check it stuck
make eval
```

Or do it all at once: `make smarter`.

---

## 📊 Current state

- **Pure Go** — stdlib-only core, plus `golang.org/x/tools`, `x/sync`,
  `x/mod` (approved exceptions)
- **6 brains**, 2,000+ training pairs, trained to answer in your words
- Social: multiturn 9/10 · Go concepts: curated KB + neural · Gocode: 16/17 ·
  Gocli: 16/16 · Makefile: deterministic recall + neural fallback ·
  Goanalyze: deterministic AST analysis, zero training needed
