#!/bin/bash
# make explain — what Gollemer is and what each make command does.
cat << 'EOF'
Gollemer — a tiny LLM in pure Go, no external dependencies.
Five brains, one chat: every message is routed to the right one.

  social    conversation — chats, answers questions, remembers the session
  go        Go concepts — goroutines, slices, errors, modules, gotchas
  gocode    writes Go code from your description (never chats)
  gocli     turns plain English into exact go/gofmt terminal commands
  makefile  maps your requests to make commands

Commands:
  make chat          talk to gollemer — one session, five brains
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

How it learns: new Q&A pairs pass a coherence quality gate, merge into
data/training/chat_pairs.jsonl, and the brains retrain on old + new.
Ask it "what is gollemer" or "what does make smarter do" in the chat.
EOF
