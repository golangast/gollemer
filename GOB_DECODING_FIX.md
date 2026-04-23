# GOB Decoding Error Fix - Training Resumption

## Problem
When attempting to resume social model training, the system would fail with:
```
⚠️ Failed to load existing model: error decoding model from gob: gob: encoded unsigned integer out of range
```

This caused silent training resets (fresh model initialization) every time resumption was attempted.

## Root Cause
**Format mismatch between save and load:**

- **Save path**: `SaveIntentMoECheckpoint()` in `intent_moe.go:1160-1206`
  - Writes: `gzip.NewWriter(file)` → `gob.Encoder`
  - Format: **Gzip-compressed checkpoint**

- **Load path (buggy)**: `LoadIntentMoEModelFromGOB()` in `intent_moe.go:1252-1268`
  - Reads: raw `gob.Decoder` (no gzip decompression)
  - Expected format: **Raw uncompressed gob**

- **Call site**: `TrainSocialChat()` in `chat.go:1871`
  - Used the legacy loader on a gzip-compressed file
  - Gob decoder attempting to parse compressed bytes as gob → decoding error

The error occurs because Go's gob package expects uncompressed data. When fed gzip bytes, it cannot parse the encoding properly and throws "unsigned integer out of range".

## Evidence

### File format verification:
```bash
$ file data/models/gob_models/moe_social_model.gob
# Output: gzip compressed data, original size modulo 2^32 204176401
```

### Code paths:
1. **Checkpoint save** (intent_moe.go:1178):
   ```go
   gz := gzip.NewWriter(file)
   encoder := gob.NewEncoder(gz)  // Encodes to compressed stream
   ```

2. **Broken resume** (chat.go:1871 - old):
   ```go
   intentModel, err = moe.LoadIntentMoEModelFromGOB(socialModelPath)
   // Uses raw gob decoder → fails on gzip bytes
   ```

3. **Checkpoint load** (intent_moe.go:1223):
   ```go
   gz, err := gzip.NewReader(file)
   decoder := gob.NewDecoder(gz)  // Correctly decompresses first
   ```

## Solution

### New Function: `LoadIntentMoEModelWithFallback()`
Added in `intent_moe.go:1270-1313`

**Features:**
1. **Format detection**: Tries gzip-compressed checkpoint format first
2. **Fallback support**: Falls back to raw gob legacy format if gzip fails
3. **File validation**: Rejects empty files with explicit error
4. **Error clarity**: Reports which formats were attempted

**Algorithm:**
```
1. Check file size (reject if empty)
2. Try gzip decompression:
   a. Create gzip.NewReader
   b. Decode as Checkpoint struct
   c. Extract and return Model if successful
3. Fallback: Seek to start, try raw gob decode
4. Return first successful format, else report both failed
```

### Updated Call Site
Changed in `chat.go:1871`:
```go
// OLD (broken):
intentModel, err = moe.LoadIntentMoEModelFromGOB(socialModelPath)

// NEW (robust):
intentModel, err = moe.LoadIntentMoEModelWithFallback(socialModelPath)
```

## Impact

### What's fixed:
✅ Model resumption now works for gzip-compressed checkpoints
✅ Backward compatibility maintained for legacy raw-gob files
✅ Empty files are rejected explicitly with clear error message
✅ Dual-format detection prevents silent resets

### Training behavior:
- **Before**: Every resume failed → fresh model → lost training progress
- **After**: Resume succeeds → model weights preserved → training continues
- **Fallback**: Old models still load (legacy format support)

## Testing

To verify the fix works:
```bash
# Run training with social data
CGO_ENABLED=0 go run -mod=mod cmd/tools/train_moe/main.go -train-social -gpu

# Let it train a few epochs and save checkpoints
# Stop the process
# Re-run the same command
# Should see: "✅ [CHECKPOINT] Saved model to..." (no "⚠️ Failed to load")
```

Expected log output on successful resume:
```
📥 Resuming training: Loading existing social model from data/models/gob_models/moe_social_model.gob
🔤 Loaded existing vocabulary from data/models/gob_models/social_vocabulary.gob: 4080 tokens
🎭 Training social model for 60 epochs at peak LR=0.000500
```

## Files Modified

1. **`internal/ai/moe/intent_moe.go`**
   - Added `LoadIntentMoEModelWithFallback()` function (lines 1270-1313)
   - Preserves existing functions for backward compatibility

2. **`internal/ai/training/chat/chat.go`**
   - Updated `TrainSocialChat()` to use new loader (line 1871)
   - Changed from `LoadIntentMoEModelFromGOB` → `LoadIntentMoEModelWithFallback`

## Additional Notes

### Related issue found:
Zero-byte checkpoint file detected:
```
data/models/gob_models/moe_social_model_step_200.gob (0 bytes)
```

This indicates a previous checkpoint finalization failure. The new empty-file check will catch this and prevent silently corrupted data.

### Future improvements:
1. Add logic to find and use the latest valid checkpoint if the main one fails
2. Implement atomic file writes throughout (temp→move pattern)
3. Add checksum validation for checkpoint integrity
4. Log the detected format during load for debugging
