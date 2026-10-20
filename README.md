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
| Analyze a repo | `analyze this project` (or `analyze ~/path/to/foo`, `analyze github.com/owner/repo`) | Explains the code in plain English for beginners: the story of what happens when you run it, the pipeline visual (main calls X calls Y), where to start reading — then visuals: package sizes, coupling, biggest files, dependency graph |
| Clone a repo to look at | `clone https://github.com/owner/repo in folder example` | Clones it into `./example` (right where you are) and immediately explains the codebase — file:line links click straight through to the files |
| Ask about the code | `what does routeDomain do`, `what's in package chat` | Signature, docs, and who calls it — or a package's purpose and key pieces |
| Understand a function simply | `walk me through routeDomain` | The xray intelligence engine explains it like you're a beginner: a real-world analogy, a step-by-step execution walk, a safety badge |
| Read the source | `show me routeDomain` | Prints the actual function |
| Find where to edit | `where would I add a retry helper`, or paste a GitHub issue or PR URL | One guided answer, not a hit list: for a feature it plans the whole change — config struct, flags, the flow to extend, similar helpers, the new file — and the call chain it sits in. A PR URL lists the files the PR touched, each with a plain-words one-liner. Paste the issues page with no number and it lists the open issues so you can pick one by number |
| Analyze a repo in plain words | `analyze https://github.com/owner/repo` or `take a look at this repo: <url>` | No command syntax needed — just say what you want in a sentence. Detects the repo URL and your intent, clones it, and explains the codebase |
| Build an API from pieces | `/build-api /users /health` | Composes a working HTTP API from learned piece schemas — not copied, not predicted. Each piece explains why it's where it is (route before server because the server reads routes at startup, etc.). Saves to `./compose/main.go` |
| Learn pieces from docs | `/learn-pieces` | Reads the doc comments in the analyzed project and extracts composable piece schemas. The library grows from any Go code's natural-language docs |
| Find code by concept | `/find auth` | Natural-language code search: finds functions and types matching your words, with file:line and a one-line description |
| Fix a Go error | `/fix undefined: foo` | Parses the compiler error, suggests similar names from your code (typo check), and explains it in plain words. Handles "cannot use X as Y", "already declared", "imported and not used" |
| Explain a concept via real code | `/whatis mutex` | Explains using the actual usage in your codebase ("Here, `Names()` uses `RLock` because it only reads") — not a textbook definition. Works for types and functions too |
| Summarize the architecture | `/arch` | Walks from `main()` through the call graph and describes what the project does in plain words |
| Scaffold a test | `/test Names` | Generates a table-driven test skeleton from the function's signature, with the doc-comment intent included |
| Explain a dependency | `/why-import json` | Traces which functions use the imported package and shows exactly how |
| Review a function | `/review Freeze` | Checks for ignored errors, overlong functions, and missing docs — each with a plain-words comment |
| See a function used for real | `/example Names` | Real call sites with the surrounding code, plus a test that exercises it. Beginners learn from examples, not definitions — and when nothing calls it, it says why |
| Take the guided tour | `/tour`, then `next` | Walks the project one stop at a time: where it starts, what to read first, the cast of types, the engine room, the shape, the tests |
| Learn the project's words | `/glossary` | The dictionary: the types a newcomer must know, one plain line each, most important first |
| Watch it work | `/watch` | Runs the project (its Makefile run target, its entry point, or — for a library — its tests) after the usual confirmation, then narrates what happened in plain words |
| Know what to learn first | `/learn-path` | The project's concepts in dependency order — each step only uses ideas from earlier steps — with the Go features each concept leans on linked to gobyexample |
| Generate a doc comment | `/comment Names` | Writes a Go doc comment from the function's structure (verb from the name, locking/iteration/error details from the analyzer) and offers to insert it |
| See what it's learned about you | `/profile` | Shows your frequent commands, projects, learned patterns, directives, and code style — all stored locally, never uploaded |
| Learn your code style | `/learn-style` | Scans your Go files and learns your conventions: handler naming, receiver style, test patterns, error handling. Generated code will match your style |
| Check the action predictor | `/predictor` | Shows how many actions are logged and whether the neural next-action predictor is ready to train (Layer 4) |
| Export structure→language pairs | `/export-understanding` | Generates training pairs from the analyzer's structural understanding: (facts → explanation). This is how you train a model on *how the program knows code* — not character-by-character like a transformer |
| Test the understanding model | `/uexplain <Func>` | The trained model explains a function from its structural facts. It never writes the function's name itself — it emits placeholders the chat fills in with the real names — and its answer is verified against the facts; a rejected answer falls back to the verified template explanation |
| Run one idea end-to-end | `/flow count with a mutex` | Box-style pipeline visual (prompt → 4 stages → renderer), then generated Go → safety check → tuned code → plain-words trace |
| Ask about a Go command | `what does go build ./... do` | Explains the exact command |
| Ask for a Go command | `build with the race detector` | Answers `go build -race ./...`, offers to run it |
| Ask about a project command | `how do i retrain the model` | Answers `run make train-go`, offers to run it |
| Run a project command | `run make eval` or just `eval` | Validates the target and offers to run it right there |
| Explain a project command | `explain make eval` | Says what the target does and shows its recipe, then offers to run it |
| Ask about a Go concept | `what is a goroutine` | Plain-English explanation |
| Ask for Go code | `write a function that reverses a string` | Generates the Go code |
| Paste a Go error | `./prog.go:12:5: undefined: foo` (or a panic) | Translates it into plain words: what it means and how to fix it — no command needed, just paste it |
| Review the session | `/history` | Shows what it remembers from this session |
| Start over | `/forget` | Wipes the session memory |
| Peek at its reasoning | `/thoughts` | Toggles the per-token thought process display |
| Leave | `/quit` (or just `exit`, `quit`, `stop`) | Ends the chat |

The prompt edits like a shell: **up/down** recalls history (kept in `~/.gollemer_history`), **left/right** moves the cursor, **tab** completes commands, make targets, file paths, and code symbols (tab twice lists), **Ctrl-C** cancels the line, **Ctrl-D** exits.

## 🗺️ New to a codebase? Start here

Five commands that teach you a project the way people actually learn one — by examples, in the right order, and by watching it run. They all work on the **last analyzed project**, so analyze (or clone) first; if you haven't, each one reminds you.

**1. Take the tour.** One stop at a time, at your pace:

```
you> /tour
--- Stop 1 of 8 ---
Welcome to github.com/fgm/izidic — 5 files, 515 lines, 3 packages...
(say 'next' for the next stop)
you> next
--- Stop 2 of 8 ---
Start where the program starts: main (examples/demo.go:10)...
```

The stops: where the program starts → what to read first → the cast of types → the engine room (most-called functions) → the shape → the tests → what to try next. Type `next` between stops; start over any time with `/tour` again.

**2. Learn the words.** `/glossary` prints the project's dictionary — the types a newcomer must know, one plain line each, most important first. If two packages define the same name, it's labeled: `Container (in examples/di)`.

**3. Know what to learn first.** `/learn-path` lists the project's concepts in dependency order — each step only uses ideas from the steps before it, says what it *Builds on*, and links the Go features it leans on to [gobyexample](https://gobyexample.com) (`mutexes (https://gobyexample.com/mutexes)`).

**4. See any function used for real.** `/example Names` — names can be bare (`Names`), receiver-qualified (`Client.MakeRequest`), or package-qualified (`matrix.New`). You get up to two real call sites with the calling line marked `→`, a line on what each caller does, and a test that exercises the function. If nothing in the project calls it, it tells you why: entry point, public surface for other projects, or possibly dead code.

**5. Watch it work.** `/watch` figures out how the project runs, in this order: a `run`/`demo`/`example`/`start`/`serve` target in the project's own Makefile → the entry point (`go run`) → for a library with no program, its tests (`go test ./...`). It asks `[y/n]` first (your "do all" directive auto-yeses, as usual), caps the run at 60 seconds, shows the program's real output, then narrates in plain words what happened — including the honest cases: it printed nothing, it ran long because it's probably a server, or it stopped with an error (paste that error into `/fix` next).

### Reading the dependency graph

The graph in the analyze report is deliberately simplified. Only packages that *connect* to something get a block — entry points (★) first, then the most-imported hubs. Packages that import nothing internal fold into one summary line (`… 10 package(s) import nothing internal: …`), build tooling folds into its own line (`… plus tooling …`), a deep cluster of packages appears as a single node (`internal/ai/neural (5 packages)`), and arrows that are already implied by a path through another package are dropped. Nothing is hidden — it's grouped, not removed.

## 🧠 Why: pieces, not memorization

Everything above is built on one idea: **understand Go code structurally, then compose — never memorize character sequences.**

- **Go's simplicity is the lever.** Go is explicit: few keywords, clear structure, doc comments on everything. The analyzer reads types, functions, call graphs, and data flow directly from the AST — no guessing.
- **Lightweight NLP bridges words to code.** Stemming maps "service definition" to the `serviceDefs` field. Doc comments become piece purposes. Issue verbs ("expose", "show") pick composition templates. No giant transformer, no character prediction.
- **Pieces carry placement constraints.** Each learned piece knows where it goes structurally (route before server because the server reads routes at startup) and why. Composition is constraint solving, not pattern completion.
- **Beginners get the map, not just the answer.** Every placement cites its reason. Every generated stub lists the available fields. The system shows *where*, explains *why*, and lets the human supply the *insight*.

This is why a composed API works the first time: the pieces were learned as structures, the composition was derived from constraints, and `go build` verified it — not because the system saw a similar API before.

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
