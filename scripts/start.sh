#!/bin/bash
# make start — open the chat. On a real terminal (stdin and stdout
# both TTYs) it execs the unified chat, exactly the invocation
# `make chat` runs, and nothing else: the chat's own start screen
# is the greeting, exiting the chat returns you to your shell, and
# NO_COLOR never blocks the launch (it only governs color, which
# the chat decides for itself). Off a terminal it prints one line
# saying a terminal is needed and exits 0 — no text is ever dumped
# at a pipe.
#
# make guide — the start-here guide, on demand only. It prints
# through this script (`scripts/start.sh guide`). The guide text
# lives in guide_text() and is the ONLY source. When stdout is a
# pipe or NO_COLOR is set, it prints verbatim. On a real terminal
# the same text passes through a bash-only colorizer speaking the
# termui vocabulary: gold ◆ title, cyan ▸ section headers, quoted
# example prompts in cyan, one hue per brain (the per-brain
# palette), make commands in cyan.

guide_text() {
cat << 'EOF'
GOLLEMER — START HERE
==================

What this is:
  A tiny LLM written in pure Go (no external dependencies).
  One chat, 17 brains: small trained models for talk, Go, code,
  commands and makefiles, plus a deterministic cast that reads
  code, makes the edits, and remembers. Each message is routed
  to the right brain.

Where to start:
  1. make start — talk to it. That is the whole product: one
     chat. (This guide is make guide, there when you want it.)
     Try one from each classic brain:
       "have you ever played soccer"         social: conversation
       "what is a goroutine"                 go: Go concepts
       "write a function that sums a slice"  gocode: writes Go code
       "what command formats my code"        gocli: go/gofmt commands
       "how do i chat with gollemer"         makefile: make commands
       "analyze this project"                goanalyze: reads a Go codebase
     Then two from the wider cast, same chat:
       "learn https://github.com/golangast/gollemer"  learn a project
       "i want to build a webserver"                  resources to build one
  2. make sel — browse every command in a columnar fuzzy finder.
  3. make explain — the full project overview and command reference.
  4. make debug-chat — same as make chat, but shows the model's thought
     process and debug prints (which experts fired, routing, etc.).

What it can do:
  social    chats, answers questions, remembers the session
            (/history to see it, /forget to wipe it)
  go        explains Go: goroutines, slices, errors, modules, gotchas
  gocode    writes Go code from your description — never chats
  gocli     turns plain English into exact go/gofmt commands,
            and asks [y/n] before running one
  makefile  maps "how do i ..." to the right Makefile command
  goanalyze reads a Go project (local path or github.com URL) with go/ast
            and maps it: entry points, most-called functions, package
            dependencies, a reading order, where the data flows, and
            "where for <github issue link>" — where an edit belongs.
            Deterministic — no neural model, so it can't hallucinate
            structure.
  composer  writes code with you, never behind your back: edits
            assembled from the project's own shapes,
            "add a webserver to main.go", renames, fixes for pasted
            Go errors, and new folders and starter files. Every
            write is planned, shown, and gated behind [y/n]; each
            is build-checked after, and a failure restores the
            original bytes.
  teacher   "learn <github link>" — or just "learn" in a project —
            teaches it in one paced flow: the story, the map, a
            walkthrough, the concepts it uses, then check questions
            drawn from the code itself.
  friend    remembers how you work between sessions: the chat opens
            with personal starting lines and what changed while you
            were away, and your learn-session misses are kept for
            later.
  watcher   gollemer -watch <folder> (make watch DIR=...) diffs
            your code as you save and notes what mattered.

How to expand it (teach it new things):
  1. Write Q&A pairs as JSONL, one per line:
       {"input": "how do i reverse a string", "output": "use a rune slice and swap ...", "domain": "go"}
     Domains: social, go, gocode, gocli, makefile.
  2. make import FILE=pairs.jsonl
     Every pair passes a quality gate: coherent ones are admitted,
     bad ones are quarantined and never trained on.
  3. Retrain a brain:
       make train-social | make train-go | make train-gocode |
       make train-gocli | make train-makefile
     Or run the whole upgrade at once:
       make smarter — more data, retrained brains, evals, and a report.
  4. make eval — score every brain on its fixed eval suite
     to confirm the new knowledge stuck.

  Data lives in data/training/chat_pairs.jsonl.
  Checkpoints live in data/models/gob_models/ (backed up before retrains).
  Ask the chat itself: "how do i teach you new things".
EOF
}

# brain_color echoes the ANSI code for a brain name (termui palette,
# BrainColorName: the six classic hues, friend purple with social,
# and the rest of the cast on termui's default purple).
brain_color() {
  case "$1" in
    social)   printf '\033[35m' ;;
    go)       printf '\033[34m' ;;
    gocode)   printf '\033[32m' ;;
    gocli)    printf '\033[36m' ;;
    makefile) printf '\033[33m' ;;
    goanalyze) printf '\033[1;37m' ;;
    friend|composer|teacher|watcher) printf '\033[35m' ;;
    *)        printf '' ;;
  esac
}

colorize_guide() {
  local RST=$'\033[0m' GOLD=$'\033[1;33m' CYAN=$'\033[36m' DIM=$'\033[2m'
  local line first rest pre quote after label
  while IFS= read -r line; do
    case "$line" in
      "GOLLEMER — START HERE")
        printf '%s◆ %s%s\n' "$GOLD" "$line" "$RST" ;;
      "==================")
        printf '%s──────────────────%s\n' "$DIM" "$RST" ;;
      "What this is:"|"Where to start:"|"What it can do:"|"How to expand it (teach it new things):")
        printf '%s▸ %s%s\n' "$CYAN" "$line" "$RST" ;;
      *)
        # Brain names opening a "What it can do" entry.
        local trimmed=${line#"${line%%[![:space:]]*}"}
        first=${trimmed%%[[:space:]]*}
        case " social go gocode gocli makefile goanalyze composer teacher friend watcher " in
          *" $first "*)
            if [[ "$line" == "  $first "* ]]; then
              printf '  %s%s%s%s\n' "$(brain_color "$first")" "$first" "$RST" "${line#  $first}"
              continue
            fi
            ;;
        esac
        # Quoted example prompts: light the quote, color the brain label.
        # (The JSONL example line is data, not a prompt — skip it.)
        if [[ "$line" == *'"'*'"'* && "$line" != *'{"input"'* ]]; then
          pre=${line%%\"*}
          rest=${line#*\"}
          quote=${rest%%\"*}
          after=${rest#*\"}
          label=${after%%:*}
          label=${label//[[:space:]]/}
          if [ -n "$(brain_color "$label")" ]; then
            printf '%s%s"%s"%s%s%s%s\n' "$pre" "$CYAN" "$quote" "$RST" \
              "${after%%"$label:"*}" "$(brain_color "$label")" "$label:$RST${after#*"$label:"}"
            continue
          fi
          printf '%s%s"%s"%s%s\n' "$pre" "$CYAN" "$quote" "$RST" "$after"
          continue
        fi
        # make commands, wherever they appear. The scan consumes
        # the line left to right so colored output is never
        # re-matched.
        local out="" rest="$line" head tail cmd
        while [[ "$rest" == *"make "* ]]; do
          head=${rest%%make *}
          tail=${rest#*make }
          cmd=${tail%%[[:space:]|,]*}
          out+="$head${CYAN}make $cmd${RST}"
          rest=${tail#"$cmd"}
        done
        printf '%s\n' "$out$rest" ;;
    esac
  done
}

print_guide() {
  if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
    guide_text | colorize_guide
  else
    guide_text
  fi
}

if [ "${1:-}" = "guide" ]; then
  print_guide
  exit 0
fi

# make start: the chat, on a real terminal only.
if [ -t 0 ] && [ -t 1 ]; then
  if ! command -v go >/dev/null 2>&1; then
    echo "gollemer's chat needs Go on your PATH — install Go, then run: make start"
    exit 0
  fi
  cd "$(dirname "$0")/.." || exit 1
  # The same environment the Makefile exports for `make chat`.
  export GOEXPERIMENT="${GOEXPERIMENT:-simd}" CGO_ENABLED="${CGO_ENABLED:-1}"
  exec go run . -real-chat -domain unified
fi

echo "make start opens the chat and needs a real terminal — for the guide, run: make guide"
exit 0
