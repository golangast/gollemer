#!/bin/bash
# make explain — what Gollemer is and what to say to it.
cat << 'EOF'
Gollemer — a tiny LLM in pure Go, no external dependencies.
Six brains, one chat: every message is routed to the right one.
Just talk to it — `make chat` — and say what you want.

WHAT TO SAY TO THE CHAT
-----------------------

Understand a Go codebase (the goanalyze brain — it reads real code,
no guessing, no hallucinations):

  you> analyze this project
  gollemer [goanalyze]> Project: github.com/golangast/gollemer
      40 packages, 300 Go files, ~45000 lines
      Entry points: ...
      IN PLAIN ENGLISH: ...

      STORY: what happens when you run it, in plain words
      Everything starts at main() in cmd/gollemer-classic/main.go.
      main() calls runClassic(): it parses the flags and picks a command.

      PIPELINE: what happens when you run it
      main  (cmd/gollemer-classic/main.go:16)
      └─▶ main.runClassic — parses flags, runs the command
          ├─▶ main.parseFlags — ...
          ...

      ... then the reading guide and the rest of the visuals:
      package-size bars, coupling, most-called functions,
      biggest files, dependency graph

  you> analyze ~/path/to/project        (any folder on disk)
  you> analyze github.com/owner/repo    (cloned once, cached)

Then ask questions about the code — it answers from what it parsed:

  you> what does routeDomain do     signature, docs, callers, callees
  you> what's in package chat       what the package is for, key pieces
  you> show me routeDomain          the actual source code
  you> where would I add a retry    ranked file:line hits
       helper

Questions only fire on names really in the project, so
"what does a goroutine do" still goes to the Go brain.

Explain commands (the gocli + makefile brains):

  you> what does go build ./... do
  you> build with the race detector
  you> what does make chat do
  you> run make eval              validated against the Makefile, then run
  you> explain make eval          what it does + its recipe, then run

The gocli brain knows the full go toolchain (build, run, test, vet,
fmt, mod, get, install, list, clean, doc, env, version, generate,
fix, work, tool, bug, telemetry). The makefile brain knows every
make command in this repo. Each prints the exact command and asks
[run it here? ...] [y/n] — answer y and it runs right there in the
terminal, no shell involved.

Write a small Go program in plain words (the /flow command):

  you> /flow count with a mutex
  gollemer [flow]> Here's your Go program:
      ```go
      ...
      ```
      💡 Think of it like ...
      1. Main — calls worker.
      ✅ Safety: verified — ...
      ⚡ Speed: already lean — ...

COMMANDS
--------
  make start         the start-here guide with example prompts
  make chat          talk to gollemer — one session, six brains
  make debug-chat    same, with the thought process shown
  make flow          one prompt through the pipeline, plain words:
                     PROMPT="create a worker pool"
  /flow <prompt>     (inside make chat) same, in plain words
  make classic       the flag-driven pipeline CLI (ARGS="-help"):
                     -flow, -impact, -style, -mock, ...
  make sel           pick a command from a columnar fuzzy finder
  make explain       this overview
  make smarter       the one-command upgrade — more data, retrained
                     brains, evals, and a report
  make eval          score every brain on its fixed eval suite
  make train-social  retrain the social conversation brain
  make train-go      retrain the Go concept brain
  make train-gocode  retrain the Go code-generation brain
  make train-gocli   retrain the Go CLI command brain
  make train-makefile
                     retrain the makefile command brain
  make import        import new training pairs through the quality gate
                     (FILE=path.jsonl) — bad pairs are quarantined
  make help          list the commands

Ask inside the chat and it will offer to run any of these for you:
  you> list the commands
  gollemer [makefile]> run make help
  [run it here? 'make help'] [y/n]: y

THE PKG/ ENGINE ROOM (the libraries behind the chat)
---------------------------------------------------
  go test ./pkg/...   # eight packages: ast, analysis, memory,
                      # runner, synthesis, engine, xray, beginner

  # the classic CLI wires them into runnable commands:
  go run ./cmd/gollemer-classic -impact -symbol "BuildGraph" -dir ./pkg/memory
      # which files would need updating if you changed that symbol
  go run ./cmd/gollemer-classic -style -dir ./pkg/memory
      # infer the repo's coding conventions (error style, tags, docs)
  go run ./cmd/gollemer-classic -mock
      # full pipeline demo: graph → proving → MCTS → self-heal → auto-tune

Where the pieces live:
  internal/ai/analyze/                 goanalyze: reads Go code with go/ast
    ascii.go                           terminal dependency-graph renderer
    visuals.go                         package sizes, coupling, biggest files,
                                       the combined visual report
    report.go                          summaries, reading guide, where-to-change
    qa.go                              answers "what does X do" about the code
  internal/ai/training/chat/
    goanalyze.go                       /analyze command, "show me a visual"
    makeexplain.go                     "run make X" / "explain make X" in chat
    flow.go                            /flow command (engine, in-process)
  pkg/engine/reactive.go               the reactive pipeline (synthesis → prove
                                       → tune → trace)
  pkg/engine/render.go                 RenderBeginner: the plain-words output
  pkg/beginner/assistant.go            plain-English code generation templates

How it learns: new Q&A pairs pass a coherence quality gate, merge into
data/training/chat_pairs.jsonl, and the brains retrain on old + new.
Ask it "what is gollemer" or "what does make smarter do" in the chat.
EOF
