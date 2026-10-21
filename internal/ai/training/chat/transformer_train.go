package chat

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/transformer"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// Mid-transformer experiment trainer (branch mid-transformer, Phase A).
//
// This is the fix-nan pair trainer ported onto the current tree and
// reworked for the experiment protocol:
//
//   - Data comes from an explicit pairs file (not a chat-dataset domain),
//     tokenized with the subword splitter in transformer_tokenize.go.
//     The vocabulary is built from the TRAIN pairs only; a held-out file
//     may be supplied for a coverage report but is never trained on and
//     never contributes vocab.
//   - Stability fixes vs the October trainer (whose recipe — post-norm,
//     constant LR 0.001, no warmup, no clipping — NaN'd from random
//     init): the layers are now pre-norm (see nnu/transformer/layers.go),
//     the LR follows linear warmup + cosine decay with peak 3e-4 (the
//     pattern the fix-nan pretrain path already had), and gradients are
//     globally norm-clipped at 1.0 before every optimizer step.
//   - From-scratch only: the old trainer's pretrained-layer loading is
//     dropped on purpose — the experiment's gate is meaningless if the
//     model starts from weights trained elsewhere.
//   - Artifacts use the old trainer's naming scheme
//     (data/models/gob_models/transformer_<exp>.gob + .ckpt/.latest
//     sidecars) so they can never collide with the LSTM production files.

const (
	transformerPeakLR      = 3e-4 // from-scratch peak (October used 1e-3 constant)
	transformerMinLR       = 3e-5 // cosine floor: 10% of peak
	transformerWarmup      = 200  // linear warmup steps
	transformerClipNorm    = 1.0  // global grad-norm clip
	transformerBatchSize   = 16   // max pairs per batch
	transformerBatchTokens = 2048 // default max padded source tokens per batch (batchMaxLen x batchSize)
	transformerMaxEpochs   = 200  // ported from the October trainer
	transformerPatience    = 20   // early-stopping patience, ported
)

// transformerConfigM is the agreed mid-size configuration: ~13-17M
// params depending on vocab, vs the abandoned 108M config.
func transformerConfigM(vocabSize int) transformer.TransformerConfig {
	return transformer.TransformerConfig{
		DModel:    256,
		NumHeads:  8,
		NumLayers: 6,
		DFF:       1024,
		MaxLen:    512,
		VocabSize: vocabSize,
	}
}

// transformerLRSchedule returns the LR for a 0-based step: linear warmup
// to the peak, then cosine decay toward the floor across totalSteps.
// Pattern copied from the fix-nan pretrain schedule (which had warmup +
// cosine while the pair trainer had neither).
func transformerLRSchedule(step, totalSteps int) float32 {
	if step < transformerWarmup {
		return transformerPeakLR * float32(step+1) / float32(transformerWarmup)
	}
	if totalSteps <= transformerWarmup {
		return transformerPeakLR
	}
	progress := float64(step-transformerWarmup) / float64(totalSteps-transformerWarmup)
	if progress > 1 {
		progress = 1
	}
	cosine := 0.5 * (1 + math.Cos(math.Pi*progress))
	return float32(float64(transformerMinLR) + float64(transformerPeakLR-transformerMinLR)*cosine)
}

// buildTransformerBatches packs pair indices into length-homogeneous
// batches under two caps: at most transformerBatchSize pairs, and at most
// batchTokens padded source tokens (batch max length x batch size). The
// token cap is a memory guard: the autograd graph retains every layer's
// [heads, seq, seq] attention scores until backward, so a flat shuffled
// batch of 16 long snippets (seq ~500) OOM-killed the VM inside step 1's
// backward. Batch composition is deterministic; the caller shuffles
// batch order per epoch.
func buildTransformerBatches(encoded []encodedPair, batchTokens int) [][]int {
	order := make([]int, len(encoded))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return len(encoded[order[a]].input) < len(encoded[order[b]].input)
	})
	var batches [][]int
	var cur []int
	curMax := 0
	flush := func() {
		if len(cur) > 0 {
			batches = append(batches, cur)
			cur = nil
			curMax = 0
		}
	}
	for _, idx := range order {
		l := len(encoded[idx].input)
		newMax := curMax
		if l > newMax {
			newMax = l
		}
		if len(cur) > 0 && (len(cur) >= transformerBatchSize || newMax*(len(cur)+1) > batchTokens) {
			flush()
			newMax = l
		}
		cur = append(cur, idx)
		curMax = newMax
	}
	flush()
	return batches
}

// transformerExpPaths returns the artifact paths for an experiment name:
// the final model, best checkpoint, latest checkpoint, and their sidecars.
func transformerExpPaths(projectRoot, expName string) (finalPath, ckptPath, metaPath, latestPath, latestMetaPath, vocabPath string) {
	base := filepath.Join(projectRoot, "data", "models", "gob_models", "transformer_"+expName+".gob")
	return base, base + ".ckpt", base + ".ckpt.json", base + ".latest", base + ".latest.json", base + ".vocab.gob"
}

// saveTransformerCheckpoint writes model weights and the resume sidecar
// (same ckptMeta format as the seq2seq trainer). Ported from fix-nan.
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

// loadTransformerCheckpoint restores a Transformer from a checkpoint,
// refusing when the vocab fingerprint moved. Ported from fix-nan.
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
	src    *tensor.Tensor // [batch, src_seq]
	tgtIn  *tensor.Tensor // [batch, tgt_seq-1] (shifted right, BOS at start)
	tgtOut [][]int        // [batch][tgt_seq-1] labels (shifted left, EOS at end)
}

// makeTransformerBatch pads a slice of encodedPairs into tensors.
// Ported from fix-nan (pad id is the vocab's PaddingTokenID, 0).
func makeTransformerBatch(pairs []encodedPair) (*transformerBatch, error) {
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
	tgtLen := maxTgt - 1
	if tgtLen < 1 {
		return nil, fmt.Errorf("target too short")
	}

	const padID = 0
	srcData := make([]float32, batch*maxSrc)
	tgtInData := make([]float32, batch*tgtLen)
	tgtOut := make([][]int, batch)

	for b, p := range pairs {
		for i := 0; i < maxSrc; i++ {
			if i < len(p.input) {
				srcData[b*maxSrc+i] = float32(p.input[i])
			} else {
				srcData[b*maxSrc+i] = float32(padID)
			}
		}
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
		src:    tensor.NewTensor([]int{batch, maxSrc}, srcData, false),
		tgtIn:  tensor.NewTensor([]int{batch, tgtLen}, tgtInData, false),
		tgtOut: tgtOut,
	}, nil
}

// transformerCrossEntropy computes mean cross-entropy loss and gradient.
// logits: [batch, seq, vocab], labels: [batch][seq], padID excluded.
// Ported from fix-nan.
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
	inv := float32(1.0 / float64(valid))
	for i := range dLogits.Data {
		dLogits.Data[i] *= inv
	}
	return float32(totalLoss / float64(valid)), dLogits, nil
}

// buildTransformerVocab builds the experiment vocabulary from pairs:
// every subword piece of every input and output. Used by the trainer
// (TRAIN pairs only — held-out data must never contribute) and by the
// eval harness when it has to rebuild the same vocabulary for an
// untrained-model plumbing run.
func buildTransformerVocab(pairs []ChatPair) *vocab.Vocabulary {
	v := vocab.NewVocabulary()
	for _, p := range pairs {
		for _, t := range subwordTokens(p.Input) {
			v.AddToken(t)
		}
		for _, t := range subwordTokens(p.Output) {
			v.AddToken(t)
		}
	}
	return v
}

// RunTransformerPairTraining trains the mid-size Transformer on a pairs
// file. pairsPath is the TRAINING file — the vocabulary is built from it
// alone. heldoutPath, when non-empty, is used only for a vocab-coverage
// report. maxSteps > 0 selects smoke mode: train exactly that many
// optimizer steps, print the loss trace, save nothing. batchTokens <= 0
// uses the default padded-token budget (transformerBatchTokens); the VM
// smoke passes a smaller budget because a step's cost scales with
// padded tokens.
func RunTransformerPairTraining(projectRoot, pairsPath, heldoutPath, expName string, resume bool, maxSteps int, batchTokens int) error {
	pairs, err := LoadChatDataset(pairsPath)
	if err != nil {
		return fmt.Errorf("loading pairs: %w", err)
	}
	if len(pairs) == 0 {
		return fmt.Errorf("no pairs in %s", pairsPath)
	}

	// Vocabulary from the TRAIN pairs only.
	v := buildTransformerVocab(pairs)
	vocabSize := v.Size()
	cfg := transformerConfigM(vocabSize)

	// Encode, dropping pairs that exceed MaxLen (the question sits at the
	// end of these inputs, so truncating would teach question-less code).
	encoded := make([]encodedPair, 0, len(pairs))
	dropped := 0
	for _, p := range pairs {
		in := encodeSubword(p.Input, v)
		outToks := encodeSubword(p.Output, v)
		tg := make([]int, 0, len(outToks)+2)
		tg = append(tg, v.BosID)
		tg = append(tg, outToks...)
		tg = append(tg, v.EosID)
		if len(in) == 0 || len(in) > cfg.MaxLen || len(tg) > cfg.MaxLen {
			dropped++
			continue
		}
		encoded = append(encoded, encodedPair{input: in, target: tg})
	}
	if len(encoded) == 0 {
		return fmt.Errorf("no encodable pairs under MaxLen=%d", cfg.MaxLen)
	}
	log.Printf("[TRANSFORMER] exp=%q pairs=%d kept=%d dropped_too_long=%d vocab=%d",
		expName, len(pairs), len(encoded), dropped, vocabSize)

	// Held-out coverage report: what fraction of held-out pieces (and of
	// held-out outputs, whole) the train vocab can express. Report only —
	// the held-out file is never trained on and never feeds the vocab.
	if heldoutPath != "" {
		held, err := LoadChatDataset(heldoutPath)
		if err != nil {
			return fmt.Errorf("loading held-out pairs: %w", err)
		}
		var totalPieces, knownPieces, fullOutputs int
		for _, p := range held {
			for _, t := range subwordTokens(p.Input) {
				totalPieces++
				if _, ok := v.WordToToken[t]; ok {
					knownPieces++
				}
			}
			outOK := true
			for _, t := range subwordTokens(p.Output) {
				totalPieces++
				if _, ok := v.WordToToken[t]; ok {
					knownPieces++
				} else {
					outOK = false
				}
			}
			if outOK {
				fullOutputs++
			}
		}
		if totalPieces > 0 {
			log.Printf("[TRANSFORMER] held-out coverage: pieces %d/%d (%.1f%%), outputs fully expressible %d/%d (%.1f%%)",
				knownPieces, totalPieces, 100*float64(knownPieces)/float64(totalPieces),
				fullOutputs, len(held), 100*float64(fullOutputs)/float64(len(held)))
		}
	}

	curFP := vocabFingerprint(v)
	finalPath, ckptPath, metaPath, latestPath, latestMetaPath, vocabPath := transformerExpPaths(projectRoot, expName)

	// Model: resume from checkpoint when asked, else from scratch.
	var model *transformer.Transformer
	startEpoch := 1
	best := float32(math.MaxFloat32)
	epochsNoImprove := 0
	if resume {
		if m, meta, err := loadTransformerCheckpoint(curFP, ckptPath, metaPath); err != nil {
			log.Printf("[TRANSFORMER] -resume: best checkpoint unusable (%v); starting fresh", err)
		} else {
			model = m
			startEpoch = meta.Epoch + 1
			best = float32(meta.Best)
			epochsNoImprove = meta.EpochsNoImprove
			log.Printf("[TRANSFORMER] RESUMED from epoch %d, best=%.4f", meta.Epoch, best)
			if lm, lmeta, lerr := loadTransformerCheckpoint(curFP, latestPath, latestMetaPath); lerr == nil && lmeta.Epoch > meta.Epoch {
				model = lm
				startEpoch = lmeta.Epoch + 1
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
	} else if model.Config.VocabSize != vocabSize {
		return fmt.Errorf("checkpoint vocab size %d != current %d", model.Config.VocabSize, vocabSize)
	}

	params := model.Parameters()
	exactParams := 0
	for _, p := range params {
		exactParams += len(p.Data)
	}
	log.Printf("[TRANSFORMER] config M: d=%d heads=%d layers=%d+%d dff=%d maxlen=%d params exact=%d (NumParams est=%d)",
		cfg.DModel, cfg.NumHeads, cfg.NumLayers, cfg.NumLayers, cfg.DFF, cfg.MaxLen, exactParams, model.NumParams())

	opt := nn.NewOptimizer(params, transformerPeakLR, transformerClipNorm)
	if batchTokens <= 0 {
		batchTokens = transformerBatchTokens
	}
	batches := buildTransformerBatches(encoded, batchTokens)
	numBatches := len(batches)
	totalSteps := transformerMaxEpochs * numBatches
	if maxSteps > 0 {
		totalSteps = maxSteps
	}
	log.Printf("[TRANSFORMER] %d pairs, %d batches/epoch, batch<=%d pairs/%d padded src tokens, total planned steps=%d, warmup=%d, peak lr=%.1e, clip=%.1f",
		len(encoded), numBatches, transformerBatchSize, batchTokens, totalSteps, transformerWarmup, transformerPeakLR, transformerClipNorm)

	rng := rand.New(rand.NewSource(42))
	step := 0
	trainStart := time.Now()
	var recent []float32 // rolling window for the trace's average

	printStep := func(loss, lr float32) {
		recent = append(recent, loss)
		if len(recent) > 20 {
			recent = recent[1:]
		}
		if step <= 10 || step == 1 || step == 50 || step == 100 || step == 250 || step == 500 || step%500 == 0 || step == maxSteps {
			var avg float32
			for _, l := range recent {
				avg += l
			}
			avg /= float32(len(recent))
			log.Printf("[TRANSFORMER] step=%d loss=%.4f avg%d=%.4f lr=%.3e elapsed=%.1fs",
				step, loss, len(recent), avg, lr, time.Since(trainStart).Seconds())
		}
	}

	for epoch := startEpoch; ; epoch++ {
		// Batch composition is fixed (length-homogeneous); shuffle the
		// batch order each epoch.
		epochBatches := make([][]int, len(batches))
		copy(epochBatches, batches)
		rng.Shuffle(len(epochBatches), func(i, j int) {
			epochBatches[i], epochBatches[j] = epochBatches[j], epochBatches[i]
		})
		var epochLoss float64
		epochBatchesDone := 0

		for _, batchIdx := range epochBatches {
			batchPairs := make([]encodedPair, 0, len(batchIdx))
			for _, idx := range batchIdx {
				batchPairs = append(batchPairs, encoded[idx])
			}
			tb, err := makeTransformerBatch(batchPairs)
			if err != nil {
				continue
			}

			lr := transformerLRSchedule(step, totalSteps)
			opt.SetLearningRate(lr)

			logits, err := model.Forward(tb.src, tb.tgtIn)
			if err != nil {
				return fmt.Errorf("step %d forward: %w", step+1, err)
			}
			loss, dLogits, err := transformerCrossEntropy(logits, tb.tgtOut, v.PaddingTokenID, vocabSize)
			if err != nil {
				return fmt.Errorf("step %d loss: %w", step+1, err)
			}
			if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
				return fmt.Errorf("loss diverged to NaN/Inf at step %d (epoch %d)", step+1, epoch)
			}
			if err := logits.Backward(dLogits); err != nil {
				return fmt.Errorf("step %d backward: %w", step+1, err)
			}
			opt.ClipGradients()
			opt.Step()
			opt.ZeroGrad()

			step++
			epochLoss += float64(loss)
			epochBatchesDone++
			printStep(loss, lr)

			if maxSteps > 0 && step >= maxSteps {
				elapsed := time.Since(trainStart).Seconds()
				log.Printf("[TRANSFORMER] smoke complete: %d steps in %.1fs (%.3f s/step, %.2f steps/s)",
					step, elapsed, elapsed/float64(step), float64(step)/elapsed)
				return nil
			}
		}

		if epochBatchesDone == 0 {
			return fmt.Errorf("no batches completed in epoch %d", epoch)
		}
		avgLoss := float32(epochLoss / float64(epochBatchesDone))
		improved := avgLoss < best
		if improved {
			best = avgLoss
			epochsNoImprove = 0
		} else {
			epochsNoImprove++
		}
		log.Printf("[TRANSFORMER] epoch=%d avg_loss=%.6f best=%.6f epochs_no_improve=%d", epoch, avgLoss, best, epochsNoImprove)

		// Checkpoints: best on improvement, latest every epoch (ported).
		meta := ckptMeta{
			Epoch:              epoch,
			Best:               float64(best),
			LastCheckpointBest: float64(best),
			LR:                 transformerLRSchedule(step, totalSteps),
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
		if err := saveTransformerCheckpoint(model, meta, latestPath, latestMetaPath); err != nil {
			log.Printf("[TRANSFORMER] latest save failed: %v", err)
		}

		if epochsNoImprove >= transformerPatience {
			log.Printf("[TRANSFORMER] early stopping at epoch %d (no improvement for %d epochs)", epoch, transformerPatience)
			break
		}
		if epoch >= transformerMaxEpochs {
			break
		}
	}

	if err := model.Save(finalPath); err != nil {
		return fmt.Errorf("saving final model: %w", err)
	}
	if err := v.Save(vocabPath); err != nil {
		return fmt.Errorf("saving vocab: %w", err)
	}
	log.Printf("[TRANSFORMER] training complete, best loss=%.6f, saved to %s (vocab %s)", best, finalPath, vocabPath)
	return nil
}
