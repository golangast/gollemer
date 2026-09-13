# Gollemer Commands & Makefile Stuff

This project uses a Makefile to manage training, analytics, and maintenance tasks. Below is a comprehensive list of available commands.

## 🛠️ Makefile Commands

### Training
| Command | Description |
|---|---|
| `make train` | Start a fresh curriculum training (clears MoE models, preserves word2vec) |
| `make train-resume` | Start training without cleaning existing model checkpoints |
| `make train-fresh` | Full fresh start — clears ALL models including word2vec, then trains |
| `make train-small` | Run the small social dataset, print loss + memory, and test the model |
| `make train-small-seq2seq` | Run a strict pure Q->A seq2seq tiny demo |
| `make test-small-seq2seq` | Load the tiny seq2seq model and probe a few prompts |
| `make seq2seq-prompt PROMPT="hello"` | Send a custom prompt to the saved tiny seq2seq model |
| `make seq2seq-chat` | Start an interactive tiny seq2seq chat loop with the saved model |
| `make chat` | Start an interactive full MoE chat loop with conversation history |

### Analytics
| Command | Description |
|---|---|
| `make metrics` | Run metrics aggregation and CSV export for edit logs |
| `make export-labels` | Export training examples to CSV for manual labeling |

### Maintenance & Utilities
| Command | Description |
|---|---|
| `make clean` | Remove MoE model checkpoints (preserves word2vec) |
| `make clean-all` | Remove ALL model files including word2vec |
| `make conversing-pb` | Convert conversing.yaml to conversing.pb |
| `make social-replies-pb` | Convert social_replies.yaml to social_replies.pb |
| `make tech-multiturn-pb` | Convert tech_multiturn.yaml to tech_multiturn.pb |
| `make all-pb` | Convert all YAML training files to protobuf |
| `make makefile-pb` | Convert Makefile targets to protobuf training data |
| `make makefile-train` | Train on makefile-generated data only |
| `make chat-makefile` | Start makefile chat with top command predictions |
| `make sel` | Interactive fuzzy finder target selector |
| `make install-hooks` | Install Gollemer Git pre-commit validation hook |