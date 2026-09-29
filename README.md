# Gollemer 🤖

A tiny LLM written in **pure Go** — zero external dependencies.
One chat, five brains: every message is routed to the right one.

```
 ┌─────────────────────────────────────────────────────────┐
 │                      make chat                          │
 │                                                         │
 │   you> what is a goroutine                              │
 │            │                                            │
 │            ▼                                            │
 │      ┌─────────────┐                                    │
 │      │   router    │─── which brain should answer?      │
 │      └──────┬──────┘                                    │
 │             │                                           │
 │    ┌────────┼────────┬────────────┬──────────┐           │
 │    ▼        ▼        ▼            ▼          ▼           │
 │ ┌──────┐ ┌──────┐ ┌───────┐ ┌──────────┐ ┌──────────┐    │
 │ │social│ │  go  │ │gocode │ │  gocli   │ │ makefile │    │
 │ │chat  │ │con-  │ │writes │ │ English  │ │"how do   │    │
 │ │      │ │cepts │ │Go code│ │→ go cmds │ │ i..."    │    │
 │ └──┬───┘ └──┬───┘ └───┬───┘ └────┬─────┘ └────┬─────┘    │
 │    └────────┴─────────┴──────────┴────────────┘          │
 │                         │                               │
 │                         ▼                               │
 │   gollemer [go]> A goroutine is a lightweight thread... │
 └─────────────────────────────────────────────────────────┘
```

The tag (`[go]`, `[social]`, …) always shows which brain answered.

---

## 🚀 Start here

```sh
make start     # the start-here guide (what to do, what it can do, how to expand it)
make chat      # talk to it — clean replies only
make debug-chat # talk to it with the thought process + debug prints shown
```

Try one prompt from each brain:

| Try saying…                              | Brain    | What happens                              |
|------------------------------------------|----------|-------------------------------------------|
| `have you ever played soccer`            | social   | chit-chat, remembers the session          |
| `what is a goroutine`                    | go       | explains the Go concept                   |
| `write a function that sums a slice`     | gocode   | writes Go code (never chats)              |
| `what command formats my code`           | gocli    | gives the exact `gofmt` command, asks [y/n] before running |
| `how do i chat with gollemer`            | makefile | suggests `run make chat`                  |

```sh
make sel       # browse every command in a fuzzy finder (see below)
make explain   # full project overview + command reference
```

---

## 🎯 `make sel` — the command picker

Don't remember the command names? `make sel` lists every target in
columns. Start typing to fuzzy-filter, pick with the arrow keys, and
it runs:

```
 ┌─ make sel ─────────────────────────────────────┐
 │ > cha                                          │
 │   chat            Talk to gollemer…            │
 │   train-chat      …                            │
 └────────────────────────────────────────────────┘
         │ you pick "chat"
         ▼
   runs: go run . -real-chat -domain unified
```

It reads the `## target: description` comments straight out of the
`Makefile`, so the list is never out of date.

---

## ⌨️ Commands

| Command             | What it does                                              |
|---------------------|-----------------------------------------------------------|
| `make start`        | Start-here guide                                          |
| `make chat`         | Talk to Gollemer — one session, five brains, clean replies only |
| `make debug-chat`   | Talk to Gollemer with the thought process + debug prints shown |
| `make sel`          | Pick a command from a columnar fuzzy finder               |
| `make explain`      | Project overview + what each command does                 |
| `make smarter`      | The one-command upgrade: more data → retrained brains → evals → report |
| `make eval`         | Score every brain on its fixed eval suite                 |
| `make train-social`   | Retrain the social brain (128/256 dims)                 |
| `make train-go`       | Retrain the Go concept brain (128/256 dims)             |
| `make train-gocode`   | Retrain the Go code brain (256/512 dims + copy gate)    |
| `make train-gocli`    | Retrain the Go CLI command brain (128/256 dims)         |
| `make train-makefile` | Retrain the makefile command brain (128/256 dims)       |
| `make import FILE=pairs.jsonl` | Import new training pairs through the quality gate |
| `make help`         | List the commands                                         |

---

## 🧠 How it learns

```
 ┌──────────────┐
 │ you write    │
 │ Q&A pairs    │   {"input": "what is a slice",
 │ (JSONL)      │    "output": "a dynamic view into an array...",
 │              │    "domain": "go"}
 └──────┬───────┘
        │ make import FILE=pairs.jsonl
        ▼
 ┌──────────────┐
 │ quality gate │─── bad pairs quarantined (never trained on)
 └──────┬───────┘
        │ good pairs admitted
        ▼
 ┌──────────────────────────────┐
 │ data/training/chat_pairs.jsonl │  ← the one dataset (1,800+ pairs)
 └──────┬───────────────────────┘
        │ make train-go / make smarter
        ▼
 ┌──────────────┐     ┌──────────────┐
 │  retrained   │────▶│  make eval   │─── did the new knowledge stick?
 │  brain (.gob)│     │  fixed suite │
 └──────────────┘     └──────────────┘
```

Three safety nets keep answers reliable:

- **Recall layers** — exact training questions get their trained answer
  verbatim (social + makefile). The tiny net doesn't have to memorize.
- **Go knowledge base** — ~50 curated Go facts answer before the neural
  net is even asked.
- **Mode separation** — the code brain can never chat and the chat brain
  can never emit code. No `I am func doing (x)` mush, by construction.

---

## 📁 Project layout

```
gollemer/
├── main.go                     # entry point: -real-chat, -train-*, -import
├── Makefile                    # all commands (make help / make sel / make start)
├── README.md                   # this file
├── go.mod                      # zero dependencies — stdlib only
│
├── scripts/
│   ├── start.sh                # make start — the start-here guide
│   ├── explain.sh              # make explain — project + command reference
│   ├── smarter.sh              # make smarter — the one-command upgrade
│   ├── *_eval_run.py           # fixed eval suites per brain
│   └── *_eval_cases.json       # eval prompts + expected answers
│
├── cmd/
│   └── tools/
│       ├── goz/                # the fuzzy finder behind make sel
│       ├── moe_inference/      # run a model from the CLI
│       └── ...                 # one-off data + debug utilities
│
├── internal/ai/
│   ├── moe/                    # mixture-of-experts layers (pure Go)
│   ├── neural/
│   │   ├── nn/                 # tensors, autograd, optimizers
│   │   ├── nnu/seq2seq/        # the seq2seq model
│   │   └── nnu/vocab/          # tokenizer + vocabulary
│   ├── training/chat/          # ⭐ the brains live here
│   │   ├── real_seq2seq_chat.go  # chat loops (unified + single-domain)
│   │   ├── real_seq2seq_train.go # training (dims, epochs, checkpoints)
│   │   ├── dataset.go            # router: which brain gets this message?
│   │   ├── goknowledge.go        # curated Go facts (answered first)
│   │   ├── socialrecall.go       # exact-match answers: social
│   │   ├── makefilerecall.go     # exact-match answers: makefile
│   │   ├── conversation.go       # session memory (/history, /forget)
│   │   └── *_test.go             # router, recall, knowledge, routing tests
│   └── ...
│
└── data/
    ├── training/
    │   ├── chat_pairs.jsonl      # ⭐ the one dataset — every brain's Q&A
    │   └── *_20260929.jsonl      # expansion batches (imported via quality gate)
    └── models/
        └── gob_models/
            ├── real_tiny_seq2seq_social.gob     # chat brain
            ├── real_tiny_seq2seq_go.gob         # Go concepts brain
            ├── real_tiny_seq2seq_gocode.gob     # Go code brain
            ├── real_tiny_seq2seq_gocli.gob      # Go CLI brain
            └── real_tiny_seq2seq_makefile.gob   # makefile brain
```

### Where to look for what

| You want to…                        | Look at                                          |
|-------------------------------------|--------------------------------------------------|
| Change how messages get routed      | `internal/ai/training/chat/dataset.go` (`routeDomain`) |
| Change what a brain knows           | `data/training/chat_pairs.jsonl` + `make import` |
| Fix a wrong deterministic answer    | `goknowledge.go`, `socialrecall.go`, `makefilerecall.go` |
| Tune training (dims, epochs)        | `real_seq2seq_train.go` (`dimsForDomain`)        |
| Change the chat UI (`/history`, …)  | `real_seq2seq_chat.go`, `conversation.go`        |
| Add a command                       | `Makefile` (`## name: description` — `sel` picks it up automatically) |

---

## 🔧 Teach it something new

```sh
# 1. write pairs (one JSON object per line)
cat > my_pairs.jsonl << 'EOF'
{"input": "what is a map in go", "output": "a map is go's hash table: keys to values...", "domain": "go"}
EOF

# 2. import through the quality gate
make import FILE=my_pairs.jsonl

# 3. retrain the brain
make train-go

# 4. check it stuck
make eval
```

Or do it all at once: `make smarter`.

---

## 📊 Current state

- **Pure Go, zero dependencies** (`go.mod` has no `require` block)
- **5 brains**, 1,800+ training pairs, seq2seq + 4-expert MoE
- Social: multiturn 9/10 · Go concepts: curated KB + neural · Gocode: 16/17 ·
  Gocli: 16/16 · Makefile: deterministic recall + neural fallback
