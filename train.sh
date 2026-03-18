#!/bin/bash

# 1. Clean up old fragmented vocab/models
rm -f gob_models/seq2seq_output_vocab.gob
rm -f gob_models/moe_classification_model.gob
rm -f gob_models/moe_classification_model_best.gob

# 2. Setup Environment
export GOGC=20
export GOMAXPROCS=2

# Create necessary directories so Go doesn't panic on file creation
mkdir -p logs
mkdir -p checkpoints
mkdir -p gob_models

# 3. Handle the CSV for the viewer
# Touch the file so the stats_viewer doesn't crash on an empty/missing file
touch logs/training.csv

# 4. Start the run
# Using GOEXPERIMENT=simd for that extra performance boost you wanted
echo "🚀 Starting Gollemer Training..."
GOEXPERIMENT=simd go run main.go -train-chat -rebalance -overfit

# 4. Notify when done
echo -e "\a" # Terminal Bell
echo "--------------------------------------------------"
echo "✅ Training Complete! Check your new MoE stats."
echo "--------------------------------------------------"