package chat

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/transformer"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// transformerCkptPaths returns checkpoint paths for the Transformer model.
// Uses a distinct prefix so it never collides with LSTM checkpoints.
func transformerCkptPaths(projectRoot, domain string) (ckptPath, metaPath string) {
	base := filepath.Join(projectRoot, "data", "models", "gob_models", "transformer_"+domain+".gob")
	return base + ".ckpt", base + ".ckpt.json"
}

func transformerLatestCkptPaths(projectRoot, domain string) (latestPath, latestMetaPath string) {
	base := filepath.Join(projectRoot, "data", "models", "gob_models", "transformer_"+domain+".gob")
	return base + ".latest", base + ".latest.json"
}

// saveTransformerCheckpoint writes model weights and resume sidecar.
func saveTransformerCheckpoint(model *transformer.Transformer, meta ckptMeta, ckptPath, metaPath string) error {
	if err := os.MkdirAll(filepath.Dir(ckptPath), 0o755); err != nil {
		return err
	}
	tmp := ckptPath + ".tmp"
	if err := model.Save(tmp); err != nil {
		return err
	}
	if err := os.Rename(tmp, ckptPath); err != nil {
		return err
	}
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(metaPath+".tmp", data, 0o644); err != nil {
		return err
	}
	return os.Rename(metaPath+".tmp", metaPath)
}

// loadTransformerCheckpoint restores a Transformer from checkpoint.
func loadTransformerCheckpoint(curFP uint64, ckptPath, metaPath string) (*transformer.Transformer, ckptMeta, error) {
	var meta ckptMeta
	data, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, meta, fmt.Errorf("no checkpoint sidecar: %w", err)
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, meta, fmt.Errorf("bad checkpoint sidecar: %w", err)
	}
	if meta.VocabFP != curFP {
		return nil, meta, fmt.Errorf("checkpoint vocab fingerprint %d != current %d", meta.VocabFP, curFP)
	}
	model, err := transformer.LoadTransformer(ckptPath)
	if err != nil {
		return nil, meta, fmt.Errorf("loading checkpoint weights: %w", err)
	}
	return model, meta, nil
}

// transformerBatch holds padded batch tensors.
type transformerBatch struct {
	src       *tensor.Tensor // [batch, src_seq]
	tgtIn     *tensor.Tensor // [batch, tgt_seq-1] (shifted right, BOS prepended)
	tgtOut    [][]int        // [batch][tgt_seq-1] labels (shifted left, EOS appended)
	srcSeqLen int
	tgtSeqLen int
}

// makeTransformerBatch pads a slice of encodedPairs into tensors.
func makeTransformerBatch(pairs []encodedPair, v *vocab.Vocabulary) (*transformerBatch, error) {
	batch := len(pairs)
	if batch == 0 {
		return nil, fmt.Errorf("empty batch")
	}
	maxSrc, maxTgt := 0, 0
	for _, p := range pairs {
		if len(p.input) > maxSrc {
			maxSrc = len(p.input)
		}
		if len(p.target) > maxTgt {
			maxTgt = len(p.target)
		}
	}
	// tgtIn is target shifted right (drop last token, BOS at start already)
	// tgtOut is target shifted left (drop BOS)
	tgtLen := maxTgt - 1
	if tgtLen < 1 {
		return nil, fmt.Errorf("target too short")
	}

	padID := 0 // assuming 0 is pad; adjust if vocab has specific pad
	srcData := make([]float32, batch*maxSrc)
	tgtInData := make([]float32, batch*tgtLen)
	tgtOut := make([][]int, batch)

	for b, p := range pairs {
		// Source: pad to maxSrc
		for i := 0; i < maxSrc; i++ {
			if i < len(p.input) {
				srcData[b*maxSrc+i] = float32(p.input[i])
			} else {
				srcData[b*maxSrc+i] = float32(padID)
			}
		}
		// Target input: p.target[0:maxTgt-1] (includes BOS, excludes last)
		// Target output: p.target[1:maxTgt] (excludes BOS, includes EOS)
		tgtOut[b] = make([]int, tgtLen)
		for i := 0; i < tgtLen; i++ {
			var inTok, outTok int
			if i < len(p.target)-1 {
				inTok = p.target[i]
			} else {
				inTok = padID
			}
			if i+1 < len(p.target) {
				outTok = p.target[i+1]
			} else {
				outTok = padID
			}
			tgtInData[b*tgtLen+i] = float32(inTok)
			tgtOut[b][i] = outTok
		}
	}

	return &transformerBatch{
		src:       tensor.NewTensor([]int{batch, maxSrc}, srcData, false),
		tgtIn:     tensor.NewTensor([]int{batch, tgtLen}, tgtInData, false),
		tgtOut:    tgtOut,
		srcSeqLen: maxSrc,
		tgtSeqLen: tgtLen,
	}, nil
}

// transformerCrossEntropy computes mean cross-entropy loss and gradient.
// logits: [batch, seq, vocab], labels: [batch][seq], padID excluded.
func transformerCrossEntropy(logits *tensor.Tensor, labels [][]int, padID, vocabSize int) (float32, *tensor.Tensor, error) {
	batch := len(labels)
	if batch == 0 {
		return 0, nil, fmt.Errorf("no labels")
	}
	seq := len(labels[0])
	dLogits := tensor.NewTensor([]int{batch, seq, vocabSize}, make([]float32, batch*seq*vocabSize), false)

	var totalLoss float64
	valid := 0
	for b := 0; b < batch; b++ {
		for t := 0; t < seq; t++ {
			label := labels[b][t]
			if label == padID {
				continue
			}
			if label < 0 || label >= vocabSize {
				return 0, nil, fmt.Errorf("label %d out of range", label)
			}
			base := (b*seq + t) * vocabSize
			mx := logits.Data[base]
			for vv := 1; vv < vocabSize; vv++ {
				if logits.Data[base+vv] > mx {
					mx = logits.Data[base+vv]
				}
			}
			var sum float64
			for vv := 0; vv < vocabSize; vv++ {
				e := math.Exp(float64(logits.Data[base+vv] - mx))
				dLogits.Data[base+vv] = float32(e)
				sum += e
			}
			totalLoss += math.Log(sum) - float64(logits.Data[base+label]-mx)
			inv := float32(1 / sum)
			for vv := 0; vv < vocabSize; vv++ {
				dLogits.Data[base+vv] *= inv
			}
			dLogits.Data[base+label] -= 1
			valid++
		}
	}
	if valid == 0 {
		return 0, nil, fmt.Errorf("no valid tokens")
	}
	// Normalize gradient by valid count
	inv := float32(1.0 / float64(valid))
	for i := range dLogits.Data {
		dLogits.Data[i] *= inv
	}
	return float32(totalLoss / float64(valid)), dLogits, nil
}

// RunTransformerTraining trains a Transformer on the given domain.
// This is v1: simplified training loop, no MoE, no copy mechanism.
func RunTransformerTraining(projectRoot, domain string, resume bool) error {
	dd, err := prepareDomainTrainData(projectRoot, domain)
	if err != nil {
		return err
	}
	trainPairs, v, tok := dd.trainPairs, dd.v, dd.tok

	// Encode pairs
	maxTgt := maxTargetTokens(domain)
	encoded := make([]encodedPair, 0, len(trainPairs))
	for _, p := range trainPairs {
		in := encodeInput(tok, p.Input)
		tg := encodeTarget(tok, v, p.Output)
		if len(in) == 0 || len(tg) < 2 || len(tg) > maxTgt {
			continue
		}
		encoded = append(encoded, encodedPair{input: in, target: tg})
	}
	if len(encoded) == 0 {
		return fmt.Errorf("no encodable training pairs")
	}

	// Sort by input length for efficient batching
	sort.Slice(encoded, func(i, j int) bool { return len(encoded[i].input) < len(encoded[j].input) })

	vocabSize := v.Size()
	log.Printf("[TRANSFORMER] domain=%q: %d pairs, vocab=%d", domain, len(encoded), vocabSize)

	// Model config
	cfg := transformer.TransformerConfig{
		DModel:    512,
		NumHeads:  8,
		NumLayers: 6,
		DFF:       2048,
		MaxLen:    512,
		VocabSize: vocabSize,
	}

	curFP := vocabFingerprint(v)
	ckptPath, metaPath := transformerCkptPaths(projectRoot, domain)
	latestPath, latestMetaPath := transformerLatestCkptPaths(projectRoot, domain)

	// Training state
	startEpoch := 1
	lr := float32(0.001)
	best := float32(math.MaxFloat32)
	epochsNoImprove := 0
	var model *transformer.Transformer

	if resume {
		if m, meta, err := loadTransformerCheckpoint(curFP, ckptPath, metaPath); err != nil {
			log.Printf("[TRANSFORMER] -resume: checkpoint unusable (%v); starting fresh", err)
		} else {
			model = m
			startEpoch = meta.Epoch + 1
			lr = meta.LR
			best = float32(meta.Best)
			epochsNoImprove = meta.EpochsNoImprove
			log.Printf("[TRANSFORMER] RESUMED from epoch %d, best=%.4f", meta.Epoch, best)
			// Prefer latest if ahead
			if lm, lmeta, lerr := loadTransformerCheckpoint(curFP, latestPath, latestMetaPath); lerr == nil && lmeta.Epoch > meta.Epoch {
				model = lm
				startEpoch = lmeta.Epoch + 1
				lr = lmeta.LR
				best = float32(lmeta.Best)
				epochsNoImprove = lmeta.EpochsNoImprove
				log.Printf("[TRANSFORMER] using latest-weights (epoch %d)", lmeta.Epoch)
			}
		}
	}

	if model == nil {
		model, err = transformer.NewTransformer(cfg)
		if err != nil {
			return fmt.Errorf("new transformer: %w", err)
		}
		// Try to load pretrained weights (vocab-independent layers only)
		pretrainCkpt := filepath.Join(projectRoot, "data", "models", "gob_models", "transformer_pretrain.gob.ckpt")
		pretrainLatest := filepath.Join(projectRoot, "data", "models", "gob_models", "transformer_pretrain.gob.latest")
		pretrainPath := ""
		if _, err := os.Stat(pretrainLatest); err == nil {
			pretrainPath = pretrainLatest
		} else if _, err := os.Stat(pretrainCkpt); err == nil {
			pretrainPath = pretrainCkpt
		}
		if pretrainPath != "" {
			if pm, err := transformer.LoadTransformer(pretrainPath); err == nil {
				// Copy vocab-independent layers: encoder/decoder layers and positional encoding
				// (embeddings and output projection are vocab-specific, keep the new ones)
				if len(pm.EncoderLayers) == len(model.EncoderLayers) && len(pm.DecoderLayers) == len(model.DecoderLayers) {
					copyTransformerLayers(pm, model)
					log.Printf("[TRANSFORMER] loaded pretrained weights from %s (layers only, embeddings re-initialized for fine-tune vocab)", pretrainPath)
				} else {
					log.Printf("[TRANSFORMER] pretrain arch mismatch, training from scratch")
				}
			} else {
				log.Printf("[TRANSFORMER] could not load pretrain weights (%v), training from scratch", err)
			}
		}
		log.Printf("[TRANSFORMER] new model, ~%d params", model.NumParams())
	}

	// Optimizer
	opt := nn.NewOptimizer(model.Parameters(), lr, 1.0)

	const batchSize = 32
	const maxEpochs = 200
	const patience = 20 // early stopping patience

	numBatches := (len(encoded) + batchSize - 1) / batchSize
	log.Printf("[TRANSFORMER] %d batches/epoch, lr=%.6f", numBatches, lr)

	for epoch := startEpoch; epoch <= maxEpochs; epoch++ {
		var epochLoss float64
		batchesDone := 0

		// Shuffle indices each epoch (simple deterministic shuffle)
		indices := make([]int, len(encoded))
		for i := range indices {
			indices[i] = i
		}
		// Fisher-Yates with epoch as seed for reproducibility
		for i := len(indices) - 1; i > 0; i-- {
			j := (epoch*7919 + i*104729) % (i + 1)
			indices[i], indices[j] = indices[j], indices[i]
		}
		// Bucket by length: stable sort shuffled indices by input length.
		// Groups similar lengths (minimal padding) while keeping random order
		// within each length group and different group order each epoch.
		sort.SliceStable(indices, func(i, j int) bool {
			return len(encoded[indices[i]].input) < len(encoded[indices[j]].input)
		})

		for b := 0; b < numBatches; b++ {
			start := b * batchSize
			end := start + batchSize
			if end > len(encoded) {
				end = len(encoded)
			}
			batchPairs := make([]encodedPair, 0, end-start)
			for _, idx := range indices[start:end] {
				batchPairs = append(batchPairs, encoded[idx])
			}

			tb, err := makeTransformerBatch(batchPairs, v)
			if err != nil {
				continue
			}

			// Forward
			logits, err := model.Forward(tb.src, tb.tgtIn)
			if err != nil {
				log.Printf("[TRANSFORMER] forward failed: %v", err)
				continue
			}

			// Loss
			loss, dLogits, err := transformerCrossEntropy(logits, tb.tgtOut, 0, vocabSize)
			if err != nil {
				continue
			}

			// Backward
			if err := logits.Backward(dLogits); err != nil {
				log.Printf("[TRANSFORMER] backward failed: %v", err)
				continue
			}

			// Optimizer step
			opt.Step()
			// Zero grads - need to clear grad buffers
			for _, p := range model.Parameters() {
				if p.Grad != nil {
					for i := range p.Grad.Data {
						p.Grad.Data[i] = 0
					}
				}
			}

			epochLoss += float64(loss)
			batchesDone++
		}

		if batchesDone == 0 {
			return fmt.Errorf("no batches completed in epoch %d", epoch)
		}
		avgLoss := float32(epochLoss / float64(batchesDone))

		improved := avgLoss < best
		if improved {
			best = avgLoss
			epochsNoImprove = 0
		} else {
			epochsNoImprove++
		}

		log.Printf("[TRANSFORMER] epoch=%d avg_loss=%.6f best=%.6f lr=%.6f", epoch, avgLoss, best, lr)

		// Checkpoint best
		meta := ckptMeta{
			Epoch:              epoch,
			Best:               float64(best),
			LastCheckpointBest: float64(best),
			LR:                 lr,
			EpochsNoImprove:    epochsNoImprove,
			VocabFP:            curFP,
		}
		if improved {
			if err := saveTransformerCheckpoint(model, meta, ckptPath, metaPath); err != nil {
				log.Printf("[TRANSFORMER] checkpoint save failed: %v", err)
			} else {
				log.Printf("[TRANSFORMER] checkpointed best model (loss=%.6f)", best)
			}
		}
		// Always save latest
		if err := saveTransformerCheckpoint(model, meta, latestPath, latestMetaPath); err != nil {
			log.Printf("[TRANSFORMER] latest save failed: %v", err)
		}

		// LR decay on plateau
		if epochsNoImprove > 0 && epochsNoImprove%10 == 0 {
			lr *= 0.5
			// Update optimizer LR - recreate with new LR
			opt = nn.NewOptimizer(model.Parameters(), lr, 1.0)
			log.Printf("[TRANSFORMER] LR decayed to %.6f", lr)
		}

		// Early stopping
		if epochsNoImprove >= patience {
			log.Printf("[TRANSFORMER] early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
			break
		}

		// NaN check
		if math.IsNaN(float64(avgLoss)) || math.IsInf(float64(avgLoss), 0) {
			return fmt.Errorf("loss diverged to NaN/Inf at epoch %d", epoch)
		}
	}

	// Save final model to production path
	finalPath := filepath.Join(projectRoot, "data", "models", "gob_models", "transformer_"+domain+".gob")
	if err := model.Save(finalPath); err != nil {
		return fmt.Errorf("saving final model: %w", err)
	}
	log.Printf("[TRANSFORMER] training complete, best loss=%.6f, saved to %s", best, finalPath)
	return nil
}

// Ensure imports are used
var _ = tokenizer.Tokenizer{}
var _ = vocab.Vocabulary{}

// copyTransformerLayers copies vocab-independent weights from a pretrained
// Transformer to a new model. Encoder/decoder layers and positional encoding
// are transferred; embeddings and output projection (vocab-specific) are kept.
func copyTransformerLayers(src, dst *transformer.Transformer) {
	for i := range dst.EncoderLayers {
		if i < len(src.EncoderLayers) {
			dst.EncoderLayers[i] = src.EncoderLayers[i]
		}
	}
	for i := range dst.DecoderLayers {
		if i < len(src.DecoderLayers) {
			dst.DecoderLayers[i] = src.DecoderLayers[i]
		}
	}
	if src.PosEncoding != nil {
		dst.PosEncoding = src.PosEncoding
	}
}
