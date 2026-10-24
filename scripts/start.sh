#!/bin/bash
# make start — the start-here guide: what to do, where to begin,
# what Gollemer can do, and how to expand it.
#
# The guide text lives in guide_text() and is the ONLY source. When
# stdout is a pipe or NO_COLOR is set, it prints verbatim (byte-
# identical to the original plain guide). On a real terminal the
# same text passes through a bash-only colorizer speaking the
# termui vocabulary: gold ◆ title, cyan ▸ section headers, quoted
# example prompts in cyan, one hue per brain (the per-brain
# palette), make commands in cyan.

guide_text() {
cat << 'EOF'
GOLLEMER — START HERE
==================

What this is:
  A tiny LLM written in pure Go (no external dependencies).
  One chat, six brains: each message is routed to the right one.

Where to start:
  1. make chat — talk to it (clean replies only). Try one from each brain:
       "have you ever played soccer"         social: conversation
       "what is a goroutine"                 go: Go concepts
       "write a function that sums a slice"  gocode: writes Go code
       "what command formats my code"        gocli: go/gofmt commands
       "how do i chat with gollemer"         makefile: make commands
       "analyze this project"                goanalyze: reads a Go codebase
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
  makefile  maps "how do i ..." to the right make command
  goanalyze reads a Go project (local path or github.com URL) with go/ast
            and maps it: entry points, most-called functions, package
            dependencies, a reading order, and where to change things
            ("where would I add a retry helper"). Deterministic — no
            neural model, so it can't hallucinate structure.

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

# brain_color echoes the ANSI code for a brain name (termui palette).
brain_color() {
  case "$1" in
    social)   printf '\033[35m' ;;
    go)       printf '\033[34m' ;;
    gocode)   printf '\033[32m' ;;
    gocli)    printf '\033[36m' ;;
    makefile) printf '\033[33m' ;;
    goanalyze) printf '\033[1;37m' ;;
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
        case " social go gocode gocli makefile goanalyze " in
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

if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
  guide_text | colorize_guide
else
  guide_text
fi
