---
description: how to train and test the MoE model with current stability fixes
---
// turbo-all

## Overview

The training pipeline runs a **3-Phase Multi-Domain Curriculum**:

| Phase | Dataset | Frozen Experts | Advance Condition |
|---|---|---|---|
| 1 — Social Core | `conversations.csv` | 8–15 (cartridge) | SVC > 0.05 + coherent, 3× in a row |
| 2 — Cartridge Ingestion | `computer.csv` | 0–7 (social) | 30 stable epochs |
| 3 — Cohesive Tuning | Both | None | Runs to completion |

### Key stability fixes in place
- **Checkpoint loader** — 3-stage format detection (gzip-Checkpoint → gzip-IntentMoE → raw-GOB). No more "encoded unsigned integer out of range" panics.
- **Expert stagnation guard** — Frozen experts (phase-masked with −1e9 logit) are excluded from stagnation counters, preventing spurious `ResetExpertWeights` calls that corrupted the gating network.
- **Word2Vec persistence** — `make clean` backs up and restores `word2vec_model.gob` so the full 1374-token vocabulary survives between training runs.

---

## 1. Pre-train Word2Vec (one-time, or after `make clean-all`)

Only needed if the vocabulary is missing or stale. Skip if `word2vec_model.gob` already exists.

```
make word2vec
```

This saves `word2vec_model.gob` and its `.bak` automatically.

---

## 2. Run the 3-Phase Curriculum (standard)

Clears MoE checkpoints, restores word2vec from backup, then trains.

```
make train
```

Environment: `GOMEMLIMIT=4500MiB GOGC=90 GOMAXPROCS=8`

Watch the logs for Phase transition signals:
```
✅ SVC > 0.0500 + coherent (3/3)
🚀 Phase 1 complete → advancing to Phase 2
```
Checkpoints are saved every 10 epochs to `data/models/gob_models/moe_social_model.gob`.

---

## 3. Resume training (without wiping the model)

```
make train-social
```

Use this to pick up from the last checkpoint without `clean`.

---

## 4. Full cold-start (wipes everything including Word2Vec)

```
make train-fresh
```

---

## 5. Test in interactive mode

```
make llm
```

Verify the inference path in the logs:
- ✅ Good: `🧠 Neural Social Match: Using weights from moe_social_model.gob`
- ⚠️ Fallback: `✅ Social Retrieval (no neural model)` — model not loaded

---

## 6. Inspect a saved model checkpoint

```
go run cmd/tools/inspect_model/main.go data/models/gob_models/moe_social_model.gob
```

Supports all serialization formats (gzip-checkpoint, gzip-model, raw-gob).
Add `--export` to write a JSON summary alongside the file.

---

## 🥧 Pi 3B Mode (900 MB RAM)

Use the `-pi` flag to enable hardware-constrained training on a Raspberry Pi 3B or
any device with ~900 MB of usable RAM. The flag automatically applies:

| Setting | Normal | Pi 3B |
|---|---|---|
| Memory cap (GOMEMLIMIT) | 4 500 MB | 600 MB |
| GC aggressiveness | 90 % | 10 % |
| GOMAXPROCS | 8 | 1 (serial GC) |
| Batch size | from config (8–16) | 1 |
| Accumulation steps | from config | 16 |
| Num experts | from config (8–16) | 4 |
| GPU | user choice | disabled |

**Social-only training on Pi (recommended — smallest model path):**
```
./gollemer -train-social -pi
```

**Combine with -epochs to keep runs short overnight:**
```
./gollemer -train-social -pi -epochs 50
```

> **Note:** Each epoch will take several minutes on a Pi 3B. Start with `-epochs 20`
> to verify the run is stable before committing to a long overnight session.
> Use `Ctrl-C` to stop — the model is checkpointed and training will resume where
> it left off on the next run.
