# Gollemer Makefile

# Set environment variables for all commands
export CGO_ENABLED=0

.PHONY: train llm clean

# Social Curriculum Training
train: clean
	@echo "🚀 Starting Social Curriculum Training..."
	go run cmd/tools/train_moe/main.go -train-social

# Interactive LLM Chat
llm:
	@echo "💬 Starting Interactive Chat..."
	go run -mod=mod cmd/tools/train_moe/main.go -llm

# Cleanup model state
clean:
	@echo "🧹 Cleaning old model state..."
	rm -f data/models/gob_models/moe_social_model.gob
	rm -f data/models/gob_models/social_vocabulary.gob
	rm -f data/models/gob_models/moe_social_model_vocab.gob
	rm -f data/models/gob_models/seq2seq_output_vocab.gob
	rm -f data/models/gob_models/moe_classification_model.gob
	rm -f data/models/gob_models/classification_vocabulary.gob
	rm -f data/models/gob_models/moe_social_model_epoch_*.gob
	rm -f data/models/gob_models/moe_social_model_step_*.gob
