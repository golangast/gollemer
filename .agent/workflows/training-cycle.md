---
description: how to train and test the MoE model with current stability fixes
---
To train and test your Mixture of Experts model with the latest stability and diversity fixes:

1. Build the updated code to ensure all fixes are applied
// turbo
go build -o gollemer ./cmd/tools/train_moe/main.go

2. Run the chat training loop
   - **Phase 0**: MLM pre-training runs first (5 epochs of fill-in-the-blank) to teach grammar
   - **Phase 1**: Then the main seq2seq training begins with curriculum learning
   - The batch size is reduced to 8 to fit 8GB GPU memory
   - Use -batch-size 4 and -acc-steps 16 if you still hit OOM
// turbo
./gollemer -train-chat -gpu

3. (Optional) Run overfitting test if you suspect signal collapse
   - Use this to verify that the model can learn a single pattern perfectly
   - Note: overfit mode SKIPS MLM pre-training to focus on the single example
./gollemer -overfit -gpu

4. Test in interactive mode
   - The new Retrieval Logic will now prioritize exact matches
./gollemer -llm

---

## 🥧 Pi 3B Mode (900 MB RAM)

Use the `-pi` flag to enable hardware-constrained training on a Raspberry Pi 3B or
any device with ~900 MB of usable RAM. The flag automatically applies:

| Setting | Normal | Pi 3B |
|---|---|---|
| Memory cap (GOMEMLIMIT) | 1 000 MB | 600 MB |
| GC aggressiveness | 20 % | 10 % |
| GOMAXPROCS | all cores | 1 (serial GC) |
| Batch size | from config (4–8) | 1 |
| Accumulation steps | from config (2–4) | 16 |
| Num experts | from config (8) | 4 |
| GPU | user choice | disabled |

**Social-only training on Pi (recommended — smallest model path):**
```
./gollemer -train-social -pi
```

**Chat training on Pi:**
```
./gollemer -train-chat -pi
```

**Combine with -epochs to keep runs short overnight:**
```
./gollemer -train-social -pi -epochs 50
```

> **Note:** Each epoch will take several minutes on a Pi 3B. Start with `-epochs 20`
> to verify the run is stable before committing to a long overnight session.
> Use `Ctrl-C` to stop — the model is checkpointed and training will resume where
> it left off on the next run.
