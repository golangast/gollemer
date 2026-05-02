#!/bin/bash
# Gollemer Training Helper

export CGO_ENABLED=0

echo "🧹 Cleaning old social model state..."
rm -f data/models/gob_models/moe_social_model.gob
rm -f data/models/gob_models/social_vocabulary.gob

echo "🚀 Starting Curriculum Training..."
go run cmd/tools/train_moe/main.go -train-social
