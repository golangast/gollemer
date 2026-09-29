#!/bin/bash
# make start — the start-here guide: what to do, where to begin,
# what Gollemer can do, and how to expand it.
cat << 'EOF'
GOLLEMER — START HERE
==================

What this is:
  A tiny LLM written in pure Go (no external dependencies).
  One chat, five brains: each message is routed to the right one.

Where to start:
  1. make chat — talk to it (clean replies only). Try one from each brain:
       "have you ever played soccer"         social: conversation
       "what is a goroutine"                 go: Go concepts
       "write a function that sums a slice"  gocode: writes Go code
       "what command formats my code"        gocli: go/gofmt commands
       "how do i chat with gollemer"         makefile: make commands
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
