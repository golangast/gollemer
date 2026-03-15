#!/bin/bash
# 1. Clean up old fragmented vocab/models (force fresh start with new compact architecture)
rm -f gob_models/seq2seq_output_vocab.gob
rm -f gob_models/moe_classification_model.gob
rm -f gob_models/moe_classification_model_best.gob

# 2. Very aggressive GC for low-RAM systems (triggers GC when heap grows by 20%)
export GOGC=20
# Limit goroutine parallelism to reduce peak memory from parallel expert execution
export GOMAXPROCS=2

# 3. Start the run with Rebalance and Curriculum
# go run main.go -train-chat -rebalance
GOEXPERIMENT=simd go run main.go -train-chat -rebalance