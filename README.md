# Gollemer 🤖

One chat in the terminal. It reads Go codebases, explains commands, and runs them for you.

Start it:

```sh
make chat
```

Six brains behind one prompt: social chatter, Go concepts, Go code, Go commands, make commands, and the codebase reader (goanalyze). Every reply is tagged with the brain that answered it, e.g. `gollemer [goanalyze]>`.

## 🎯 What Gollemer is for

**Catching up on code you haven't touched in a while — including your own.** Open a project and ask `what does this do`, `where does the routing happen`, `where does counts get its data`. You get the story, a reading order, and exact file:line answers in plain words, instead of an evening of re-reading files cold. (Run the Watcher beside your editor and the Friend will open your next session with "while you were away", covering the saves that mattered.)

**The first hour in somebody else's Go repo.** Clone a project you're curious about and chat with it: the structure, the main flow, where a feature lives, which words the project uses. Reading unfamiliar code fast — and being able to explain it back — is the core junior-dev skill, and it's what this chat practices with you.

**Small changes, made safely.** Ask `where would I add a retry helper` and it points at the spot. Then ask for the edit itself in a plain sentence — `add a rice case to Timer` — and it assembles the change from the project's own sibling code, shows the exact diff, and waits for your `[y/n]`. If the project doesn't build afterwards, it puts the file back the way it was and tells you. Nothing in your project is ever edited without your yes.

**Learning Go from real code instead of tutorials.** The tour, the project dictionary, the learning path, and plain-English translations of compiler errors all work on the project in front of you — your code, not textbook examples.

**What it is not.** Gollemer is not a general AI assistant, and it won't write arbitrary new programs for you — its neural brains are small on purpose, and their measured limits are printed in the cast list below, failures included. It speaks Go, runs entirely on your machine (offline, free, nothing uploaded), and serves one reader at a time. What it gives up in breadth it returns in exactness: because a real AST analyzer reads your actual files, its answers cite code that is really there — never a plausible-sounding file that doesn't exist. It teaches instead of doing it for you, which is the point if you're the one trying to get better.

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
| Trace where data comes from | `where does counts get its data`, `what changes Items` | Follows one variable, struct field, or map through the whole project: where it's declared, what first gives it data (the initializer, the `make(...)`, the assignment), and every place it changes — each with file:line. If several things share the name, it lists the candidates instead of guessing |
| Make a small edit | `add a rice case to Timer`, `add a SugarKg field of type float64 to Pantry`, `register tea in Recipes`, `clone Brew as Steep` | Assembles the change from the project's own sibling code — a switch case cloned from a sibling case, a struct field (positional literals elsewhere in the project are extended in the same diff), a map/slice table entry, a function copied from a sibling. Shows the exact diff and waits for `[y/n]`. Declining leaves the file byte-identical; if the build fails after applying, the original file is restored and it says so. Generated and vendored files are always refused |
| Add a declaration in your own words | `add a struct called Point with X as an int and Y as a string to point.go` | When the concept book's exact grammar doesn't match, a small trained model proposes the declaration instead — marked as a model proposal in the plan, behind the same `[y/n]` gate and build check. See "The model proposer" below |
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
| See a function used for real | `example Names`, `show me an example of Names` (or `/example Names`) | Real call sites with the surrounding code, plus a test that exercises it. Beginners learn from examples, not definitions — and when nothing calls it, it says why |
| Take the guided tour | `tour https://github.com/fgm/izidic`, `give me a tour` (or `/tour`), then `next` | Walks the project one stop at a time: where it starts, what to read first, the cast of types, the engine room, the shape, the tests |
| Take the training tour | `teach me` (or `training`, `beginner tour`) | Six tiny lessons that train YOU on what gollemer can do — the story, one function, why/compare, running it, getting unstuck, the team. Each lesson: it shows you, then you say the phrase yourself and watch it work. The Friend remembers your progress, so it resumes where you left off — and a fresh install offers it at startup, because nobody reads the README |
| Control the reading speed | `/speed` (or `speed off`, `type slower`) | Long answers stream in line by line instead of dumping as one giant block. Short answers print instantly; `/speed` toggles. Output is also colored on a real terminal (prompt, reply tags, section bands, accepted/rejected verdicts) — `NO_COLOR=1` turns color off, piping turns both off automatically |
| Learn the project's words | `what are the main types` (or `/glossary`) | The dictionary: the types a newcomer must know, one plain line each, most important first |
| Watch it work | `run it` (or `/watch`) | Runs the project (its Makefile run target, its entry point, or — for a library — its tests) after the usual confirmation, then narrates what happened in plain words |
| Know what to learn first | `what should I learn first` (or `/learn-path`) | The project's concepts in dependency order — each step only uses ideas from earlier steps — with the Go features each concept leans on linked to gobyexample |
| Generate a doc comment | `/comment Names` | Writes a Go doc comment from the function's structure (verb from the name, locking/iteration/error details from the analyzer) and offers to insert it |
| See what it's learned about you | `/profile` | Shows your frequent commands, projects, learned patterns, directives, and code style — all stored locally, never uploaded |
| Learn your code style | `/learn-style` | Scans your Go files and learns your conventions: handler naming, receiver style, test patterns, error handling. Generated code will match your style |
| Check the action predictor | `/predictor` | Shows how many actions are logged and whether the neural next-action predictor is ready to train (Layer 4) |
| Export structure→language pairs | `/export-understanding` | Generates training pairs from the analyzer's structural understanding: (facts → explanation), written to `understanding_pairs.jsonl` in the analyzed project's folder. This is how you train a model on *how the program knows code* — not character-by-character like a transformer |
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

### The model proposer: when the exact words don't match

Almost every write above is assembled deterministically — the concept book knows a fixed set of sentences, like `add a struct named Config to store.go`. Spell the members out in your own words instead — `add a struct called Point with X as an int and Y as a string to point.go` — and the book's grammar misses. That miss is the one place a trained model gets a turn in the write family:

- **When it fires.** Only on an `add`/`put`/`create` ask that names a struct, interface, function or method *and* a file (or folder) to put it in — and only after every exact shape, the concept book included, has passed on the sentence. Anything the book serves is served exactly as before; the model is never consulted.
- **What it does.** The model (a small pointer-generator trained to turn a sentence into a declaration) reads your words and proposes one declaration. If its answer can't be turned into a real declaration — names or types it invented that nothing can restore — nothing changes at all: you get the same honest "I can't add that yet" answer you get today.
- **How it stays safe.** A usable proposal arrives as a plan like any other write, with a note in the plan saying a model proposed it and that the model is often wrong. The exact diff, the `[y/n]` gate, the on-disk backup, and the build check all work the same as for an assembled edit — approve a proposal that doesn't build and your files are restored, with the build error explained.
- **Its measured record, so you can calibrate trust.** On declaration combinations it never saw in training, it gets the whole declaration exactly right 44.4% of the time with familiar wording and 34.7% with unfamiliar wording too — against a 90% bar it did not meet. That is exactly why it *proposes* behind the gate instead of writing.

**Setup:** the proposer needs its model file at `data/models/gob_models/real_tiny_seq2seq_speccopy.gob` in the checkout you run `make chat` from. That file is **not in git** — trained models are local files — so if you run Gollemer on another machine (a Chromebook checkout, say), copy the file over by hand. Without it the proposer is simply off: no error, no mention, and chat behaves exactly as it did before.

## 🗺️ New to a codebase? Start here

Five commands that teach you a project the way people actually learn one — by examples, in the right order, and by watching it run. **Just say them as sentences** — the `/slash` forms in the table above are shortcuts that do the same thing. The chat prints a few of them every time it starts, and `help` (or `what can you do`) shows everything you can say, grouped by what you're trying to do.

No project open yet? No dead end: these commands look at the folder the chat started in, and if there's no Go code there either, the tour runs on **pocketcounter** — a tiny example project bundled in `examples/pocketcounter` (a counter with a lock, ~85 lines) — so the first tour works in the first minute, with nothing to clone. And a mistyped command gets a "did you mean…?" instead of an error.

**You don't need the exact words.** A small intent model — a classifier trained on 22,221 phrasings, fragments and typos included (`go run ./scripts/gen_intent_data && go run ./scripts/train_intent`) — works out what you mean across the analyzer's whole command set: *"show me Freeze being used somewhere real"*, *"is Freeze written ok"*, *"whats the big picture structure here"*. It also knows the questions every stuck beginner asks: **"tell me more"** (goes one step deeper into whatever you just saw — an example becomes a walkthrough, a review becomes a test), **"what should I do now"** (looks at where you are and names one concrete next step), and **"what does this program do"** (the plain-English summary). It is judged the honest way, on a handwritten set of phrasings it never trained on (208 of 212 at last training — the generator now filters out anything that would leak a held-out line into training), and a regression test keeps it above 90%. Lines it isn't confident about fall through to the brains exactly as before — and a near miss now gets a "did you mean…?" question with its best one or two guesses, instead of silence. It only ever adds understanding, never takes it away.

**1. Take the tour.** Say `tour https://github.com/fgm/izidic` (the link carries the project along), or `give me a tour` for the project already analyzed. One stop at a time, at your pace:

```
you> tour https://github.com/fgm/izidic
--- Stop 1 of 8 ---
Welcome to github.com/fgm/izidic — 5 files, 515 lines, 3 packages...
(say 'next' for the next stop)
you> next
--- Stop 2 of 8 ---
Start where the program starts: main (examples/demo.go:10)...
```

The stops: where the program starts → what to read first → the cast of types → the engine room (most-called functions) → the shape → the tests → what to try next. Type `next` between stops; start over any time with `/tour` again.

**2. Learn the words.** Say `what are the main types` (or `/glossary`). It prints the project's dictionary — the types a newcomer must know, one plain line each, most important first. If two packages define the same name, it's labeled: `Container (in examples/di)`.

**3. Know what to learn first.** Say `what should I learn first` (or `/learn-path`). It lists the project's concepts in dependency order — each step only uses ideas from the steps before it, says what it *Builds on*, and links the Go features it leans on to [gobyexample](https://gobyexample.com) (`mutexes (https://gobyexample.com/mutexes)`).

**4. See any function used for real.** Say `example Names` or `show me an example of Names` (or `/example Names`) — names can be bare (`Names`), receiver-qualified (`Client.MakeRequest`), or package-qualified (`matrix.New`). You get up to two real call sites with the calling line marked `→`, a line on what each caller does, and a test that exercises the function. If nothing in the project calls it, it tells you why: entry point, public surface for other projects, or possibly dead code.

**5. Watch it work.** Say `run it` (or `/watch`). It figures out how the project runs, in this order: a `run`/`demo`/`example`/`start`/`serve` target in the project's own Makefile → the entry point (`go run`) → for a library with no program, its tests (`go test ./...`). It asks `[y/n]` first (your "do all" directive auto-yeses, as usual), caps the run at 60 seconds, shows the program's real output, then narrates in plain words what happened — including the honest cases: it printed nothing, it ran long because it's probably a server, or it stopped with an error (paste that error into `/fix` next).

### New here? Say `start`

The guided walk: Gollemer drives — six short stops over the open
project (the story, its hardest-working function, its key words, a look
at it running, the team and the Friend, your first concept to learn).
You just say `next`; questions interrupt freely; your place is kept in
the Friend's notebook per project, so `start` resumes across sessions.
`help` answers with three situations, not a catalog; the catalog lives
behind `help all`.

## The cast: many small brains, one coordinator

Gollemer is not one model. A **coordinator** routes each sentence to a
specialist brain, records whether it delivered, and answers for the
routing. Ask it directly:

| Say this | What happens |
|---|---|
| `brains` | the cast: every brain, its job, and its honest limits |
| `how are your brains doing` | the scoreboard — per-brain ok/failed counts, this session and all time (logged locally in `~/.gollemer_brains.jsonl`) |
| `why did you do that` | the coordinator names the brain that answered, the recorded outcome, and that brain's known limit |
| `why does Add lock` | the Reasoner: why a function is shaped as it is — what the author stated vs what the structure implies |
| `any suggestions` | the Friend: a personal brain that watches how you work across all the brains, remembers topics and habits between sessions (`~/.gollemer_friend.json`), and speaks up rarely — a stuck brain, a function you keep circling — under a long cooldown. Silence is its default |
| `compare Add and Get` | the Comparer: two functions side by side, ending with the one biggest difference |

**The Watcher** (`gollemer -watch <folder>`, or `make watch DIR=<folder>`):
an opt-in mode that sits beside your editor. On every save it re-analyzes
the project and diffs the *structure* — only meaningful changes become
events (a signature change with callers at stake, a function that started
locking, a new exported function no test touches, a function edited 4+
times). Events print in its terminal and are logged locally; the Friend
reads the log and mentions what mattered, under its usual restraint.

The trained models are cast members too — with their measured records
printed in the cast list, failures included. When the Storyteller model
invents facts, the Verifier rejects it and the Analyst's verified
wording speaks instead; the rejection is recorded on the scoreboard.

## Reading the dependency graph

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
