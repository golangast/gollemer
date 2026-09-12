# Gollemer Makefile
# -----------------------------------------------------------------------------

# Configuration
export GOEXPERIMENT=simd
export CGO_ENABLED=1

# Runtime Tuning
MEM_LIMIT    = 2500MiB
GOGC         = 50
GOMAXPROCS   = 8
MAIN_CMD     = go run main.go

.PHONY: train train-resume train-fresh train-small train-small-seq2seq \
       test-small-seq2seq seq2seq-prompt seq2seq-chat chat metrics export-labels \
       clean clean-all conversing-pb social-replies-pb tech-multiturn-pb all-pb \
       makefile-pb makefile-train chat-makefile install-hooks help sel

## install-hooks: Install Gollemer Git pre-commit validation hook
install-hooks:
	@bash scripts/install_git_hook.sh

# --- Training ---

## train: Start a fresh curriculum training (clears MoE models, preserves word2vec)
train: clean
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-multiphase $(ARGS)

## train-resume: Start training without cleaning existing model checkpoints
train-resume:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-multiphase $(ARGS)

## train-fresh: Full fresh start — clears ALL models including word2vec, then trains
train-fresh: clean-all
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-multiphase $(ARGS)

## train-small: Run the small social dataset, print loss + memory, and test the model
train-small:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-small

## train-small-seq2seq: Run a strict pure Q->A seq2seq tiny demo
train-small-seq2seq:
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-small-seq2seq

## test-small-seq2seq: Load the tiny seq2seq model and probe a few prompts
test-small-seq2seq:
	$(MAIN_CMD) -test-small-seq2seq

PROMPT ?= "hello"
## seq2seq-prompt: Send a custom prompt to the saved tiny seq2seq model
seq2seq-prompt:
	$(MAIN_CMD) -seq2seq-prompt="$(PROMPT)"

## seq2seq-chat: Start an interactive tiny seq2seq chat loop with the saved model
seq2seq-chat:
	$(MAIN_CMD) -seq2seq-chat

## chat: Start an interactive full MoE chat loop with conversation history
chat:
	$(MAIN_CMD) -chat

# --- Analytics ---

## metrics: Run metrics aggregation and CSV export for edit logs
metrics:
	@echo " Generating edit metrics and CSV..."
	@go run scripts/compute_edit_metrics.go || true
	@go run scripts/edits_to_csv.go || true
	@echo " metrics written to logs/edits/"

## export-labels: Export training examples to CSV for manual labeling
export-labels:
	@echo " Exporting edits_failed.jsonl -> data/training/edits_for_labeling.csv"
	@go run scripts/export_for_labeling.go || true

# --- Maintenance ---

## clean: Remove MoE model checkpoints (preserves word2vec)
clean:
	rm -f data/models/gob_models/*.gob
	@if [ -f data/models/gob_models/word2vec_model.gob.bak ]; then \
		cp data/models/gob_models/word2vec_model.gob.bak data/models/gob_models/word2vec_model.gob 2>/dev/null || true; \
	fi

## clean-all: Remove ALL model files including word2vec
clean-all:
	rm -f data/models/gob_models/*.gob

## conversing-pb: Convert conversing.yaml to conversing.pb
conversing-pb:
	go run ./cmd/tools/gen_conversing_yaml_pb -in data/training/trainingdata/conversing.yaml

## social-replies-pb: Convert social_replies.yaml to social_replies.pb
social-replies-pb:
	go run ./cmd/tools/gen_conversing_yaml_pb -in data/training/trainingdata/social_replies.yaml

## tech-multiturn-pb: Convert tech_multiturn.yaml to tech_multiturn.pb
tech-multiturn-pb:
	go run ./cmd/tools/gen_conversing_yaml_pb -in data/training/trainingdata/tech_multiturn.yaml

## all-pb: Convert all YAML training files to protobuf
all-pb: conversing-pb social-replies-pb tech-multiturn-pb makefile-pb

## makefile-pb: Convert Makefile targets to protobuf training data
makefile-pb:
	go run ./cmd/tools/gen_makefile_pb -makefile Makefile -out data/training/trainingdata/makefile.pb

## makefile-train: Train on social + makefile-generated data only
makefile-train: all-pb
	GOMEMLIMIT=$(MEM_LIMIT) GOGC=$(GOGC) GOMAXPROCS=$(GOMAXPROCS) $(MAIN_CMD) -train-multiphase -makefile-only $(ARGS)

## chat-makefile: Start makefile chat with top command predictions
chat-makefile:
	$(MAIN_CMD) -chat-makefile

## sel: Interactive fuzzy finder target selector
sel:
	@target=$$(awk '/^## [a-zA-Z0-9_-]+:/ { \
		cmd=$$2; sub(":", "", cmd); \
		$$1=$$2=""; \
		printf "%-22s %s\n", cmd, $$0; \
	}' $(MAKEFILE_LIST) | go run ./cmd/tools/goz/main.go -h 25); \
	if [ -n "$$target" ]; then \
		$(MAKE) $$target; \
	fi