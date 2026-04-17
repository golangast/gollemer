# Training Guide for Gollemer

## Two Different Training Modes

### 1. Performance Benchmark (Synthetic Data)
**File**: `cmd/train/main.go`  
**Purpose**: GPU performance testing with random synthetic data  
**Output**: Throughput metrics, NOT trained LLM

```bash
# Build GPU-accelerated benchmark
CGO_ENABLED=1 go build -mod=mod -o bin/train_gpu ./cmd/train/main.go

# Run benchmark (fastest throughput)
./bin/train_gpu -batch 256 -experts 8 -dim 256 -prefetch 4
# Output: Epoch 0 | Loss: 0.030476 | Throughput: 2000 samples/sec

# Runs indefinitely on synthetic data x*0.5 = y
# Press Ctrl+C to stop
```

**⚠️ WARNING**: This does NOT train the LLM. It trains on fake data. The model will NOT respond intelligently after this.

---

### 2. Real LLM Training (Actual Intent Data)
**File**: `cmd/tools/train_moe/main.go -train-chat`  
**Purpose**: Train conversational AI on real chat/intent data  
**Output**: Trained model that responds intelligently

```bash
# Train real LLM with GPU acceleration
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-chat \
  -gpu \
  -batch-size 4 \
  -acc-steps 8 \
  -lr 0.0001 \
  -epochs 50

# Then test the trained model
CGO_ENABLED=0 go run -mod=mod cmd/tools/train_moe/main.go -llm
```

**Data Loaded**: 
- Chat pairs from `data/training/`
- Intent classification from `data/training/trainingdata/`
- ~30K training samples

---

### 3. Social-Only Training (Pure Conversational)
**File**: `cmd/tools/train_moe/main.go -train-social`  
**Purpose**: Train a specialized model ONLY on social/conversational data  
**Output**: Separate model for natural social interactions without technical confusion  
**Data Source**: `data/training/trainingdata/human_chat.txt` ONLY

```bash
# Train social-only model (takes 5-15 minutes)
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-social \
  -gpu \
  -batch-size 4 \
  -epochs 30

# Creates two models:
# - moe_social_model.gob (conversational only)
# - social_vocabulary.gob (social vocab)
```

**Key Differences from `-train-chat`**:
- ❌ Does NOT load `conversing.csv` (no technical data)
- ✅ ONLY trains on natural human conversations from `human_chat.txt`
- ✅ Smaller, focused model (256d, 4 experts vs 512d, 8 experts)
- ✅ Pure conversational responses without word salad

**When This Activates**:
During inference, social queries automatically route to this model:
- "How are you?" → Uses social model ✅
- "What's your favorite holiday?" → Uses social model ✅
- "Create a Go handler" → Uses general model ❌
- "Build a project" → Uses general model ❌

**Example Responses**:
```
User: "How are you doing?"
Social Model: "I'm doing well! I enjoy learning about what people think matters in their lives."

User: "Do you ever feel lonely?"
Social Model: "I find it interesting how fleeting human connections can be, yet how meaningful."

User: "Create a handler file"
General Model: <creates handler.go file>
```

---

## Training Strategy: Combining Both Models

For best results, train **both** general and social models:

```bash
# Step 1: Train general model (handles all data)
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-chat -gpu -batch-size 4 -epochs 20

# Step 2: Train social model (handles social queries)
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-social -gpu -batch-size 4 -epochs 30

# Step 3: Test - queries automatically route to correct model
CGO_ENABLED=0 go run -mod=mod cmd/tools/train_moe/main.go -llm
```

**Result**: 
- Social queries get pure conversational training
- Technical queries get mixed training data for better understanding
- No mixing → No word salad

---

## Quick Start Workflows

### I Just Want to Benchmark GPU Performance
```bash
CGO_ENABLED=1 go build -mod=mod -o bin/train_gpu ./cmd/train/main.go
./bin/train_gpu -batch 256 -experts 8 -dim 256 -prefetch 4

# Measure throughput, press Ctrl+C after a few seconds
# Expected: ~2000 samples/sec
```

### I Want to Train a Smart LLM
```bash
# Step 1: Train the model (takes 10-30 minutes depending on data)
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-chat -gpu -batch-size 4 -acc-steps 8 -epochs 20

# Step 2: Test the trained model interactively
CGO_ENABLED=0 go run -mod=mod cmd/tools/train_moe/main.go -llm

# Now type commands and get intelligent responses
# > how are you
# Model responds with learned patterns
```

### I Want GPU-Optimized Real Training
```bash
# Recommended for best throughput: train on real data with goroutine pipelining
# Run in background with logging
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-chat -gpu -batch-size 8 -acc-steps 16 -epochs 50 \
  > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Stop when satisfied
pkill -f train_moe
```

### I Want Natural Social Conversations (No Word Salad)
```bash
# Train ONLY on human_chat.txt for pure social responses
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-social -gpu -batch-size 4 -epochs 30

# Creates: moe_social_model.gob + social_vocabulary.gob
# These are automatically used when you ask social questions

# Test it:
# > how are you?
# 🎭 Social Model: "I'm doing well! I enjoy learning about what fascinates people."
# 
# > create a handler
# ➜ General Model: <creates handler.go>
```

### I Want Both Models (Recommended for Production)
```bash
# Step 1: Train general model first (all data combined)
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-chat -gpu -batch-size 4 -epochs 20

# Step 2: Train social model (pure conversational)
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-social -gpu -batch-size 4 -epochs 30

# Step 3: Test with automatic routing
CGO_ENABLED=0 go run -mod=mod cmd/tools/train_moe/main.go -llm

# Result:
# - Social queries → social_model.gob (clean conversational responses)
# - Technical queries → moe_classification_model.gob (command execution)
# - No mixing → No confusion for either type of query
```

---

## Understanding the Difference

| Aspect | Benchmark (cmd/train) | General Training (-train-chat) | Social Training (-train-social) |
|--------|----------------------|--------------------------------|--------------------------------|
| Data Source | Random `x * 0.5` | conversing.csv + human_chat.txt | **human_chat.txt ONLY** |
| Purpose | Performance testing | Balanced LLM capability | Pure conversational responses |
| Model Quality | ❌ Gibberish | ✅ Mixed capability | ✅ Social focus |
| Output File | (benchmark only) | moe_classification_model.gob | **moe_social_model.gob** |
| Use Case | GPU benchmarking | Production LLM | Social query handling |
| Response Style | Random | Technical + Social | Natural conversation |

---

## Model Files After Training

After running both training modes:

```
data/models/gob_models/
├── moe_classification_model.gob      ← General model (from -train-chat)
├── moe_social_model.gob              ← Social model (from -train-social) ✨ NEW
├── semantic_output_vocabulary.gob
├── query_vocabulary.gob
├── social_vocabulary.gob             ← Social vocab (from -train-social) ✨ NEW
└── word2vec_model.gob
```

**Inference automatically uses both**:
- Social queries → `moe_social_model.gob`
- Technical queries → `moe_classification_model.gob`

---

## The Problem You Encountered

You ran the benchmark on synthetic data:
```bash
./bin/train_gpu -batch 256 -experts 8 -dim 256 -prefetch 4
```

This trained on fake data `x * 0.5`, so when you ran:
```bash
go run cmd/tools/train_moe/main.go -llm
```

The model had learned only random patterns, producing gibberish like:
```
"paper? trading recognize monophyletic predator?"
```

---

## Solution: Train on Real Data

```bash
# Step 1: Train on actual chat data
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-chat -gpu -batch-size 4 -acc-steps 16 -epochs 30

# Step 2: Run LLM (now with learned patterns)
CGO_ENABLED=0 go run -mod=mod cmd/tools/train_moe/main.go -llm

# Step 3: Test
# > how are you
# /ʕ●‿●ʔ/ > I'm doing well! What can I help you with?
```

---

## Training Flags for Real LLM

```bash
# Flags for both -train-chat and -train-social:
-train-chat          # Enable general chat training (all data)
-train-social        # Enable social-only training (human_chat.txt only) ✨ NEW
-gpu                 # Use GPU acceleration (requires CGO_ENABLED=1)
-batch-size INT      # Batch size (4-16 recommended)
-acc-steps INT       # Gradient accumulation steps (8-32) [only for -train-chat]
-epochs INT          # Number of training epochs (20-50 for general, 20-30 for social)
-lr FLOAT            # Learning rate (0.0001-0.001 recommended)
-weight-decay FLOAT  # L2 regularization (0.01 typical)
-max_grad_norm FLOAT # Gradient clipping (1.0 typical)
```

### Tuning Guide

**General Training - Fast (20 minutes)**:
```bash
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-chat -gpu -batch-size 8 -acc-steps 8 -epochs 10 -lr 0.0005
```

**General Training - Best Quality (2 hours)**:
```bash
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-chat -gpu -batch-size 4 -acc-steps 16 -epochs 50 -lr 0.0001 -weight-decay 0.01
```

**Social Training - Fast (10 minutes)**:
```bash
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-social -gpu -batch-size 8 -epochs 15
```

**Social Training - Best Quality (15-20 minutes)**:
```bash
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go \
  -train-social -gpu -batch-size 4 -epochs 30 -lr 0.0005
```

**For GPU Benchmarking Only**:
```bash
# Synthetic data only - does NOT train the LLM
./bin/train_gpu -batch 256 -experts 8 -dim 256 -prefetch 4
```

---

## Saving & Loading Models

Models are automatically saved to `data/models/gob_models/` after training:

**From `-train-chat`**:
- `moe_classification_model.gob` - General model (technical + social mixed)
- `query_vocabulary.gob` - General vocabulary

**From `-train-social`** (✨ NEW):
- `moe_social_model.gob` - Social model (conversational only)
- `social_vocabulary.gob` - Social vocabulary

**Automatic Routing**:
When you run `-llm` for inference:
- Social queries automatically load `moe_social_model.gob`
- Technical queries use `moe_classification_model.gob`
- No manual model switching needed

To start fresh:
```bash
rm data/models/*.gob
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go -train-chat -gpu -epochs 20
```

---

## Summary

### The Problem: Mixed Training Data Confusion
When training on both technical and social data together, the model learns both but can confuse them:
- Social queries might produce technical word salad
- Technical queries might produce conversational responses

### The Solution: Separate Models

**Option 1: General Model Only**
```bash
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go -train-chat -gpu -epochs 20
# Result: Mixed capability (okay for both, not great at either)
```

**Option 2: Separate Specialized Models** (✨ RECOMMENDED)
```bash
# Step 1: Train general model
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go -train-chat -gpu -epochs 20

# Step 2: Train social model  
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go -train-social -gpu -epochs 30

# Result: Each model excels at its domain
```

### Key Takeaways

| Use Case | Command | Result |
|----------|---------|--------|
| Benchmark GPU only | `./bin/train_gpu` | Performance metrics, NOT real training |
| General LLM (mixed) | `-train-chat` | Handles all queries but not specialized |
| Social conversations | `-train-social` | Pure conversational responses ✨ |
| Production (best) | Both `-train-chat` + `-train-social` | Automatic routing, no confusion ✨ |

### To Get Started

```bash
# Train both models for best results
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go -train-chat -gpu -epochs 20
CGO_ENABLED=1 go run -mod=mod cmd/tools/train_moe/main.go -train-social -gpu -epochs 30

# Test with automatic routing
CGO_ENABLED=0 go run -mod=mod cmd/tools/train_moe/main.go -llm
```

The system now automatically detects social vs technical queries and routes to the appropriate model!
