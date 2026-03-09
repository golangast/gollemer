# Gollemer

Gollemer is an intelligent coding assistant and project orchestrator designed to help you build Go applications, specifically focusing on web servers and WASM frontends. It understands natural language and can hold a **real conversation** — powered by a custom-built, end-to-end trainable neural network written entirely in Go with no external ML dependencies.

---

## 💬 Conversational AI — How It Works

Gollemer communicates with you through a dual-mode conversational system. It can:
1. **Understand developer commands** — "create a handler called Login with URL /login"
2. **Hold general conversation** — "how are you?", "what do you think about Go?"

Both modes run through the same neural pipeline: Word2Vec embeddings → MoE Encoder → Intent Classifier → (if chat) RNN Decoder for response generation.

### 🔀 Intent Resolution Flow

When you type a message, Gollemer runs it through a **Hybrid Intent Resolver** with the following stages:

```
User Input
    │
    ▼
cleanTokenize()  ← splits on whitespace + punctuation, lowercases
    │
    ▼
SentenceVocab.GetTokenID()  ← maps each word to a token ID
    │
    ▼
nn.Embedding.Forward()  ← looks up learned 256-dim word vectors
    │
    ▼
HybridLLMGNNEncoder  ← MoE sparse encoding (8 experts, top-2)
    │
    ▼
Intent Classification
    ├── "create_webserver", "create_handler", "create_page", etc.  → Command Execution
    └── "chat_response"  → RNN Decoder generates reply
```

### 🎯 Intent Categories

| Intent | Example Input | Action |
|---|---|---|
| `create_webserver` | "make a webserver called myapp" | Scaffolds a new Go webserver project |
| `create_handler` | "add handler Login at /login" | Creates a handler file + registers route |
| `create_page` | "create a page called dashboard" | Generates a WASM page file |
| `create_database` | "make a database named users" | Creates an SQLite DB |
| `create_file` | "create file config.go" | Creates a blank file |
| `create_folder` | "make folder utils" | Creates a directory |
| `move_file` | "move main.go to cmd/" | Moves a file to a new path |
| `chat_response` | "how are you?" | Generates a conversational reply |
| `help_command` | "help with handlers" | Shows relevant help text |

### 🧩 Entity Extraction

After classifying the intent, Gollemer extracts **named entities** from your input:

- **`name`** — the target object name (e.g., "Login" in "create handler Login")
- **`url`** — HTTP route (e.g., "/login")  
- **`path`** — destination directory (e.g., "in cmd/" or "to src/")
- **`tables`** — database fields (e.g., "fields name string age int")
- **`command`** — for help requests

It uses a layered approach: NER tags → POS tags → heuristic keyword proximity scanning → fallback patterns.

---

## 🧠 Neural Architecture

Gollemer's conversational brain is an **encoder-decoder Seq2Seq architecture** with a Mixture-of-Experts encoder, all implemented from scratch in Go.

### Word2Vec Embeddings

All words are first converted to dense **256-dimensional vectors** using a pre-trained Word2Vec model (~61,700 words). During chat training, the vocabulary is expanded with new words from the training data. Any word not found in Word2Vec gets a random Xavier-initialized vector so it can still be learned.

- **Vocabulary Size:** ~61,700+ words (expandable)
- **Embedding Dim:** 256
- **Coverage:** 100% of known training data tokens

### 🔀 MoE Encoder — Sparse Mixture of Experts

The encoder is a **Sparse MoE Layer** that processes the embedded token sequence:

```
Input Sequence [Batch, SeqLen, 256]
        │
        ▼
GatingNetwork (Linear 256 → 8)  ← decides which experts to use
        │  ← Noisy Top-K: adds Gaussian noise during training
        │  ← Temperature Scaling: controls routing sharpness
        │  ← Expert Dropout: randomly disables experts to prevent collapse
        ▼
Top-2 Expert Selection (per token)
        │
        ├── Expert 0 (FFN: 256→512→256)
        ├── Expert 1 (FFN: 256→512→256)
        ├── ...
        └── Expert 7 (FFN: 256→512→256)
        │
        ▼
Weighted Sum of Expert Outputs  ← gating probabilities as weights
        │
        ▼
Context Vector [Batch, SeqLen, 256]
```

**Key MoE properties:**
- **8 Experts** — each specializes in different linguistic patterns (greeting tones, command verbs, technical terms, etc.)
- **Top-2 routing** — each token is processed by its 2 most relevant experts
- **Capacity Factor (1.25)** — limits how many tokens each expert handles per batch to prevent overload
- **Load Balancing Loss** — auxiliary loss added to training to ensure all experts get used, not just the popular ones
- **Noisy Gating** — Gaussian noise (σ = 2/numExperts) added to router logits during training to force exploration
- **GRPO Support** — optional Group Relative Policy Optimization for advanced expert selection

### 🔁 HybridLLMGNNEncoder

The MoE layer is wrapped in a **Hybrid LLM-GNN Encoder**:
- The MoE layer acts as the LLM-style contextual encoder
- A Graph Neural Network (GNN) layer adds relational reasoning between tokens
- The two are combined and projected back to the model dimension

### 📡 RNN Decoder — Seq2Seq Response Generation

For `chat_response` intents, the decoder generates a reply token by token:

```
Context Vector (from Encoder)
        │
        ▼
hiddenState = mean(ContextVector)  ← initialize LSTM hidden state
cellState   = zeros
        │
        ▼
[For each output token]:
        │
        ├── Embedding.Forward(previousToken)
        │
        ├── LSTM Step (2 layers, hiddenSize=512)
        │       ├── Input gate
        │       ├── Forget gate
        │       ├── Cell gate
        │       └── Output gate
        │
        ├── MultiHeadCrossAttention (8 heads)
        │       └── Query=LSTM_hidden, Key/Value=ContextVector
        │
        ├── LayerNorm([LSTM_hidden + Attention_output])
        │
        ├── Linear(hiddenSize → VocabSize=5189)
        │
        ├── ApplyRepetitionPenalty(generatedIDs)
        ├── ApplyFrequencyPenalty(generatedIDs)
        ├── StuckDetector → force change if same token 3x in a row
        │
        └── SampleFromLogits(temperature, topK, topP)
                │
                ▼
        Next Token ID  → stop if EOS, else loop
```

**Key decoder properties:**
- **2-Layer LSTM** with hidden size 512
- **8-Head Cross-Attention:** attends to the full encoder context vector
- **Integrated LayerNorm:** stabilizes the combined hidden + attention vector
- **Output Vocabulary:** 5,189 tokens (all training words + special tokens)

---

## 🗣️ Chat Training (`go run main.go -train-chat`)

The chat model is trained on `trainingdata/human_chat.txt` — a curated dataset of human conversation pairs.

### Training Data Format

The training data consists of question-answer pairs:
```
Q: how are you
A: i'm doing well, thanks for asking! how about you?

Q: what do you do for fun
A: i enjoy reading, hiking, and talking with friends!
```

### Training Pipeline

```
1. Load Word2Vec (~61K words)
2. Load or initialize MoE model
3. Expand Word2Vec vocab with new chat words (Xavier-init for new vectors)
4. Build SentenceVocab from all training Q+A tokens
5. Resize model output layer to match SentenceVocab size
6. Compute class weights for imbalanced vocab (penalize UNK/PAD)
7. Mute UNK token weight in output layer (bias = -100.0)
8. For each epoch:
   a. Shuffle training pairs
   b. Tokenize Q/A → tensor IDs
   c. Encoder.Forward(Q_embeddings)
   d. Decoder.Forward(context, A_tokens, samplingProb)
   e. WeightedCrossEntropy(logits, targets, mask)
   f. Add LoadBalancingLoss from MoE gating
   g. Backward pass (BPTT through decoder, encoder, embeddings)
   h. Gradient clip at norm=5.0
   i. Adam optimizer step
   j. Validate on hold-out set → compute Perplexity
   k. Save best model checkpoint
```

### Training Techniques

| Technique | Details |
|---|---|
| **Teacher Forcing** | Uses ground-truth tokens as decoder input for first epochs — fast convergence |
| **Scheduled Sampling** | From epoch 2+: probability ramps up to 25% of using model's own predictions as input |
| **Weighted CrossEntropy** | UNK token weighted 0.01×, pad tokens masked, regular words 1.0× |
| **Label Smoothing** | Smoothing factor 0.05 to prevent overconfident probabilities |
| **Gradient Clipping** | Hard clip at global norm 5.0 to prevent LSTM gradient explosions |
| **UNK Muting** | Output layer weights for UNK set to -100.0 — ensures model never outputs unknown tokens |
| **Early Stopping** | Stops training if validation perplexity doesn't improve for 4 epochs (patience=4) |
| **LR Warmup + Decay** | Learning rate starts at 2e-5, decays toward floor of 5e-6 |
| **MoE Diverse Routing** | First 2 epochs use temperature=2.0 to force all 8 experts to activate |
| **Data Split** | 90% train, 10% validation |

### Expert Utilization Monitoring

After each epoch, Gollemer prints a visual breakdown of which experts handled the most tokens:

```
Layer 0 Expert Utilization (Capacity Factor: 1.50):
  Expert 0:     9825 ( 11.9%) #####
  Expert 1:     7357 (  8.9%) ####
  Expert 2:    10830 ( 13.1%) ######
  Expert 3:     7073 (  8.6%) ####
  Expert 4:     7377 (  9.0%) ####
  Expert 5:    14349 ( 17.4%) ########
  Expert 6:     9125 ( 11.1%) #####
  Expert 7:    16429 ( 19.9%) #########
```

A healthy training run shows all experts being used. If Expert 7 dominates, increase the Load Balancing Weight or router temperature.

### Model Checkpoints

| File | Description |
|---|---|
| `gob_models/moe_classification_model.gob` | Latest checkpoint (saved every epoch) |
| `gob_models/moe_classification_model_best.gob` | Best validation perplexity checkpoint |
| `gob_models/seq2seq_output_vocab.gob` | Vocabulary used by the decoder output layer |

---

## 🚀 Inference & Generation (Running the Conversation)

### Starting the LLM (Conversational Mode)

```bash
go run main.go -llm
```

This starts an interactive shell. Gollemer will:
1. Load the Word2Vec model
2. Load the trained MoE model from disk
3. Load the SentenceVocab (decoder output vocabulary)
4. Start listening for your input

### How a Response is Generated

When you type something like `"how are you?"`:

1. **Tokenize:** `["how", "are", "you", "?"]`
2. **Vocab Lookup:** Each token is mapped to its `SentenceVocab` ID (e.g., `how` → 301)
3. **Embed:** 256-dim vectors fetched from the Embedding layer
4. **Encode:** MoE encoder produces a context vector `[1, 4, 256]`
5. **Classify:** Intent = `chat_response`
6. **Decode:** LSTM decoder generates tokens one-by-one, using cross-attention to reference the encoded input
7. **Filter:** Repetition penalty, frequency penalty, and stuck-detector are applied each step
8. **Stop:** Generation halts at the `</s>` (EOS) token or max length (20 tokens)
9. **Print:** The decoded words are joined and printed as the response

### Decoding Strategies

| Strategy | When Used | Description |
|---|---|---|
| **Greedy (Top-1)** | Commands (create, etc.) | Always picks highest probability token |
| **Top-K Sampling** | Chat responses | Samples from K most likely tokens for variety |
| **Top-P (Nucleus)** | Long form responses | Samples from the smallest set of tokens summing to P probability |
| **Beam Search** | High-quality generation | Maintains N candidate sequences, picks best overall |
| **Strict Generate** | Chat responses | Filters out special tokens, enforces structural coherence |

### Conversation Context & Memory

The conversational system maintains **turn-level memory**:
- Each exchange (user input + model response) is stored in a session memory buffer
- Previous turns are used to bias the next response (recency-weighted)
- The session memory persists across the conversation but resets on program exit

---

## Interactive Menu System

Gollemer includes a comprehensive interactive menu to guide you through project creation, management, and AI training. You can access this menu by typing `menu` in the Gollemer shell.

### Main Menu Options

#### 1. 🚀 Start a New Project (Webserver)
Initializes a new Go webserver project.
- **Action:** Prompts for a project name.
- **Result:** Creates a directory with `main.go` (including SQLite setup and a basic handler) and initializes `go.mod`.

#### 2. ➕ Add a Feature
Adds components to your existing project.
- **a. Handler (Backend logic):** Creates a new Go handler function and registers it in your `main.go`.
- **b. Page (Frontend view):** Generates a WASM-compatible Go page using the internal UI framework and registers it in the WASM router.
- **c. Database (Storage):** Creates a new SQLite database file or adds tables if fields are specified.

#### 3. 📂 Manage Files
Basic file system operations within your project context.
- **a. Create File:** Creates a new file (can use templates if learned).
- **b. Create Folder:** Creates a new directory.

#### 4. ▶️ Run Project
Builds and runs your application.
- **Action:** Prompts for the webserver name (defaults to current context).
- **Result:** Compiles the Go code, builds any WASM components, and starts the server.

#### 5. 🧠 Learning & Training
Manage the AI and learning capabilities of Gollemer.
- **1. Show Learning Status:** Displays loaded models, vocabulary sizes, and the current learning source path.
- **2. Change Learning Source:** Updates the directory Gollemer scans for templates and code patterns.
- **3. Teach New Object Word:** Manually adds a new noun/object to the knowledge base.
- **4. Run Training Commands:** Access advanced model training and visualization tools:
    1. Train Word2Vec
    2. Train MoE (Mixture of Experts)
    3. Train Intent Classifier
    4. Train NER (Named Entity Recognition)
    5. Custom Training Module
    6. Visualize Neural Network
    7. Visualize Word2Vec Model
    8. Search Word Neighbors
    9. Visualize Word Relationship
    10. Visualize Word Distribution (2D Plot)
    11. Inspect Model Weights
    12. Visualize Attention Mechanism
    13. Visualize Word Similarity (One vs List)

#### 6. 🎓 Tutorial
Starts an interactive, step-by-step tutorial that guides you through creating a folder, a file, a webserver, and running it.

#### 7. ❓ Help
Displays the general help text with command syntax and examples.

#### 8. 🚪 Exit
Closes the Gollemer application.

#### 9. 💬 Interactive Mode
Returns to the main prompt, allowing you to enter natural language commands directly.

#### 10. ⚙️ Model Configuration
View and update the file paths for the AI models (Word2Vec, MoE, NER) and vocabularies used by Gollemer.

---

## ⚡ Performance Optimization

Gollemer supports **SIMD (Single Instruction, Multiple Data)** acceleration for neural network operations (Matrix Multiplication, Vector Addition, etc.) using the experimental `simd/archsimd` package.

To enable SIMD acceleration, build and run with the `simd` experiment flag:

```bash
GOEXPERIMENT=simd go run main.go
```

This can significantly speed up training and inference, especially for the MoE and Word2Vec models.

---

## 🗂️ Model Files Reference

| File | Purpose |
|---|---|
| `gob_models/word2vec_model.gob` | Pre-trained word embeddings (~61K vocabulary) |
| `gob_models/moe_classification_model.gob` | Main MoE model (encoder + decoder) |
| `gob_models/moe_classification_model_best.gob` | Best checkpoint by validation perplexity |
| `gob_models/seq2seq_output_vocab.gob` | Decoder output vocabulary (SentenceVocab) |
| `gob_models/ner_model.gob` | Named Entity Recognition model |
| `gob_models/query_vocabulary.gob` | Input token vocabulary (Word2Vec-mapped) |
| `trainingdata/human_chat.txt` | Conversational training data (Q/A pairs) |
| `knowledge.json` | Known commands, objects, and model config paths |

---

## 🔧 Training Commands

```bash
# Train the conversational chat model (main seq2seq)
go run main.go -train-chat

# Train the Word2Vec embeddings from scratch
go run main.go -train-word2vec

# Train the MoE intent classifier
go run main.go -train-moe

# Train the Named Entity Recognition model
go run main.go -train-ner

# Train the intent classification pipeline
go run main.go -train-intent-classifier

# Run in interactive LLM / conversational mode
go run main.go -llm
```
