#!/bin/bash
set -e

echo "🚀 Gollemer NLP Structure Initialization"
echo "========================================="

# 1. Clean up old models to ensure consistency
echo "🧹 Cleaning up old GOB models..."
rm -f data/models/gob_models/*.gob

# 2. Re-create vocabularies
echo "🏋️ Training Word2Vec model..."
go run cmd/train_word2vec/main.go

echo "📝 Creating Query Vocabulary..."
go run cmd/create_vocab/main.go

echo "📝 Creating Semantic Output Vocabulary..."
go run cmd/init_nlp_structure/main.go

# 3. Initialize MoE Model with 1 epoch of training
echo "🏋️ Initializing MoE Model weights (Dry Run)..."
go run cmd/train_moe/main.go -epochs 1 -dry-run

echo ""
echo "✅ NLP Structure Initialized Successfully!"
echo "Checkpoints saved in data/models/gob_models/"
