#!/bin/bash
# make explain — what Gollemer is and what each make command does.
cat << 'EOF'
Gollemer — a tiny LLM in pure Go, no external dependencies.
Seven brains, one chat: every message is routed to the right one.

  social    conversation — chats, answers questions, remembers the session
  go        Go concepts — goroutines, slices, errors, modules, gotchas
  gocode    writes Go code from your description (never chats)
  gocli     turns plain English into exact go/gofmt terminal commands
  makefile  maps your requests to make commands
  goanalyze reads a Go codebase with go/ast — entry points, hot spots,
            package dependencies, reading order, where to change things
            (deterministic, no neural model)
  beginner  explains Go simply — analogies, tiny snippets, plain-English
            "why" for generated code (deterministic, no neural model)

  $ # beginner brain, inside make chat:
  you> beginner: explain goroutines
  gollemer [beginner]> **Goroutines** — Hiring an extra cook — the kitchen
  keeps working while they chop vegetables in the background.
  ```go
  func main() {
      go chop("carrots")   // runs concurrently
      go chop("onions")
      time.Sleep(100 * time.Millisecond)
  }
  ```
  Key rules:
  - When main returns, the program exits — even if goroutines are
    still running.
  - Use channels or a sync.WaitGroup to wait for goroutines to finish.

Visuals: the goanalyze brain draws the codebase, two ways —

  $ # ASCII visuals, right in the chat:
  you> analyze this project
  ENGINE ROOM (most-called functions)
  ████████████████████ tensor.NewTensor (62 callers)
  ████ analyze.Func.Display (13 callers)
  ...

  $ # ...or a standalone HTML page you open in a browser:
  you> show me a visual
  gollemer [goanalyze]> Interactive visual:
    ~/workspace/your_files/codebase-maps/github-com-golangast-gollemer.html
  (SVG dependency graph — blue nodes, red = entry point — plus stats
  and reading-order cards; no server needed)

The shell is a separate single-file REPL (cmd/gollemer/main.go):
every prompt runs synthesis → format/parse → safety inspection →
genetic auto-tuning → visual execution trace, rendered in the
terminal with ANSI color.

  $ make shell-once PROMPT="build a list of squares"
  === GENERATED GO SOURCE ===
  func squares(n int) []int {
      out := make([]int, 0, n)   // preallocated by the tuner
      for i := 0; i < n; i++ {
          out = append(out, i*i)
      }
      return out
  }
  💡 BEGINNER CONCEPT ANALOGY
  A factory assembly line: the tray is sized for the whole order up
  front, so workers never stop to fetch a bigger one.
  ─── VISUAL EXECUTION TRACE ───
  [1] Main — calls squares, entry point
  [2] Squares — leaf, runs 1 loop, 3 statements
  STATUS BADGES
  ⚡ PERF — 1 → 0 allocs/op (saved 1)

  $ make shell-once PROMPT="create a worker pool"   # goroutines + channels
  /shell <prompt>   # same pipeline, inline, inside make chat

The pkg/ engine room — the libraries behind it all:

  go test ./pkg/...   # all nine packages: ast, analysis, memory,
                      # runner, synthesis, engine, xray, beginner, chat

  # the classic CLI (local only) wires them into runnable commands:
  go run ./cmd/gollemer-classic -mock
      # full pipeline demo: graph → proving → MCTS → self-heal → auto-tune
  go run ./cmd/gollemer-classic -impact -symbol "BuildGraph" -dir ./pkg/memory
      # which files would need updating if you changed that symbol
  go run ./cmd/gollemer-classic -style -dir ./pkg/memory
      # infer the repo's coding conventions (error style, tags, docs)

  # pkg/chat serves a web UI: 20-line main → http://localhost:8080
  # (see README "The pkg/ engine room" for the recipe)
  # pkg/engine has no CLI yet — run it via go test ./pkg/engine/ -v

Commands:
  make chat          talk to gollemer — one session, seven brains
  make shell         interactive natural-language Go shell (REPL)
  make shell-once    run one shell prompt: PROMPT="build a list of squares"
  /shell <prompt>    (inside make chat) run the shell pipeline inline
  make flow          one prompt through the beginner pipeline, plain words:
                     PROMPT="create a worker pool"
  /flow <prompt>     (inside make chat) same pipeline, in plain words
  (local only: make shell-classic runs the flag-driven pipeline CLI)
  make sel           pick a command from a columnar fuzzy finder
  make explain       this overview
  make smarter       the one-command upgrade — more data, retrained
                     brains, evals, and a report
  make eval          score every brain on its fixed eval suite
  make train-social  retrain the social brain (128/256 dims)
  make train-go      retrain the Go concept brain (128/256 dims)
  make train-gocode  retrain the Go code brain (256/512 + copy gate)
  make train-gocli   retrain the Go CLI command brain (128/256 dims)
  make train-makefile
                     retrain the makefile command brain (64/128 dims)
  make import        import new training pairs through the quality gate
                     (FILE=path.jsonl) — bad pairs are quarantined
  make help          list the commands

Where the new pieces live:
  cmd/gollemer/main.go                 the shell — one self-contained file
                                       (go run cmd/gollemer/main.go)
  cmd/gollemer-classic/                older flag-driven pipeline CLI (local only)
                                       (-flow: one prompt, plain words)
  pkg/engine/reactive.go               the reactive pipeline (synthesis → prove
                                       → tune → trace)
  pkg/engine/render.go                 RenderBeginner: the plain-words output
  pkg/beginner/assistant.go            beginner brain engine: explanations + code
  internal/ai/training/chat/beginner.go
                                       beginner brain wiring (beginner: markers)
  internal/ai/training/chat/shell.go   /shell chat command (runs the shell once)
  internal/ai/training/chat/flow.go    /flow chat command (engine, in-process)

How it learns: new Q&A pairs pass a coherence quality gate, merge into
data/training/chat_pairs.jsonl, and the brains retrain on old + new.
Ask it "what is gollemer" or "what does make smarter do" in the chat.
EOF
