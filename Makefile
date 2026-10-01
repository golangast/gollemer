# Gollemer — one chat, seven brains.
#
#   make start      the start-here guide: what to do, what it can do, how to expand it
#   make chat       talk to gollemer (unified chat: social, go, code, makefile, gocli)
#   make debug-chat chat with the thought process and debug prints shown
#   make shell      interactive natural-language Go shell (REPL)
#   make shell-once run one shell prompt non-interactively (PROMPT="...")
#   make shell-classic
#   make flow       one prompt through the beginner pipeline (PROMPT="...")
#                   the full flag-driven pipeline CLI (local only)
#   make sel        pick a command from a columnar fuzzy finder
#   make explain    what the project is and what each command does
#   make smarter    the one-command upgrade: expands training data, retrains the
#                   social + go brains, runs the evals, prints a report
#   make eval       score every brain on its fixed eval suite
#   make train-*    retrain one brain: social, go, gocode, gocli, makefile
#   make import     import new training pairs through the quality gate (FILE=path.jsonl)
#   make help       this list

export GOEXPERIMENT=simd
export CGO_ENABLED=1

MEM_LIMIT  = 2500MiB
GOGC       = 100
GOMAXPROCS = 8
MAIN_CMD   = go run main.go

.PHONY: start chat debug-chat shell shell-once shell-classic flow sel explain smarter eval help import \
        train-social train-go train-gocode train-gocli train-makefile

## start: The start-here guide — what to do, what it can do, how to expand it
start:
	bash scripts/start.sh

## chat: Talk to gollemer — one session, five brains, routed per message
chat:
	$(MAIN_CMD) -real-chat -domain unified

## debug-chat: Chat with the thought process and debug prints shown
debug-chat:
	$(MAIN_CMD) -real-chat -domain unified -debug-chat

## shell: Interactive natural-language Go shell — synthesis, safety, tuning, trace
shell:
	go run ./cmd/gollemer

## shell-once: Run one shell prompt non-interactively: make shell-once PROMPT="build a list of squares"
shell-once:
	go run ./cmd/gollemer -once "$(PROMPT)"

## shell-classic: The full flag-driven pipeline CLI (pass ARGS, e.g. ARGS="-help")
shell-classic:
	go run ./cmd/gollemer-classic $(ARGS)

## flow: One prompt through the beginner pipeline — make flow PROMPT="create a worker pool"
flow:
	go run ./cmd/gollemer-classic -flow -prompt "$(PROMPT)"

## sel: Pick a make command from a columnar fuzzy finder
sel:
	@target=$$(awk '/^## [a-zA-Z0-9_-]+:/ { \
		cmd=$$2; sub(":", "", cmd); \
		$$1=$$2=""; \
		printf "%-14s %s\n", cmd, $$0; \
	}' $(MAKEFILE_LIST) | go run ./cmd/tools/goz/main.go -h 25); \
	if [ -n "$$target" ]; then \
		$(MAKE) $$target; \
	fi

## explain: What the project is and what each command does
explain:
	bash scripts/explain.sh

## smarter: The one-command upgrade — more data, retrained brains, evals, report
smarter:
	bash scripts/smarter.sh

## eval: Score every brain on its fixed eval suite
eval:
	python3 scripts/gocode_eval_run.py
	python3 scripts/goconcept_eval_run.py
	python3 scripts/gocli_eval_run.py
	python3 scripts/social_multiturn_eval.py

## train-social: Retrain the social brain (128/256 dims)
train-social:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-real-seq2seq -domain social

## train-go: Retrain the Go concept brain (128/256 dims)
train-go:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-real-seq2seq -domain go

## train-gocode: Retrain the Go code brain (256/512 dims + copy gate)
train-gocode:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-real-seq2seq -domain gocode

## train-gocli: Retrain the Go CLI command brain (128/256 dims)
train-gocli:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-real-seq2seq -domain gocli

## train-makefile: Retrain the makefile command brain (128/256 dims)
train-makefile:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-real-seq2seq -domain makefile

## import: Import new training pairs through the quality gate (FILE=path.jsonl)
import:
	$(MAIN_CMD) -import-pairs="$(FILE)"

## help: Show this list
help:
	@awk 'BEGIN { print "Gollemer commands:\n" } \
		/^## [a-z-]+:/ { cmd=$$2; sub(":", "", cmd); $$1=$$2=""; \
			printf "  make %-14s %s\n", cmd, $$0 }' $(MAKEFILE_LIST)
