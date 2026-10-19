package chat

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/transformer"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// Pretraining constants
const (
	pretrainSeqLen    = 256  // tokens per training sequence (reduced from 512 for memory)
	pretrainBatchSize = 8    // sequences per batch (reduced from 32 for memory)
	pretrainVocabSize = 8000 // max vocabulary size
	pretrainWarmup    = 1000 // linear warmup steps
	pretrainMaxSteps  = 50000
	pretrainBaseLR    = 0.001
	pretrainMinLR     = 0.0001
)

// pretrainPaths returns paths for pretraining checkpoints and vocab.
func pretrainPaths(projectRoot string) (ckptPath, metaPath, vocabPath, finalPath string) {
	base := filepath.Join(projectRoot, "data", "models", "gob_models", "transformer_pretrain.gob")
	return base + ".ckpt", base + ".ckpt.json",
		filepath.Join(projectRoot, "data", "models", "gob_models", "transformer_pretrain_vocab.json"),
		base
}

// savePretrainVocab writes the vocabulary as JSON for fine-tuning reuse.
func savePretrainVocab(v *vocab.Vocabulary, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	// Serialize as token->id map
	data, err := json.MarshalIndent(v.WordToToken, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// loadPretrainVocab reads a saved vocabulary.
func loadPretrainVocab(path string) (*vocab.Vocabulary, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var wordMap map[string]int
	if err := json.Unmarshal(data, &wordMap); err != nil {
		return nil, err
	}
	return vocab.NewVocabularyFromMap(wordMap), nil
}

// pretrainMeta is the checkpoint sidecar for pretraining.
type pretrainMeta struct {
	Step    int     `json:"step"`
	Best    float64 `json:"best"`
	LR      float32 `json:"lr"`
	VocabFP uint64  `json:"vocab_fp"`
}

// chunkCorpus splits token IDs into fixed-length sequences.
// Short final chunks are dropped.
func chunkCorpus(ids []int, seqLen int) [][]int {
	var chunks [][]int
	for i := 0; i+seqLen <= len(ids); i += seqLen {
		chunk := make([]int, seqLen)
		copy(chunk, ids[i:i+seqLen])
		chunks = append(chunks, chunk)
	}
	return chunks
}

// pretrainLRSchedule computes LR with linear warmup then cosine decay.
func pretrainLRSchedule(step int) float32 {
	if step < pretrainWarmup {
		// Linear warmup
		return pretrainBaseLR * float32(step) / float32(pretrainWarmup)
	}
	// Cosine decay from base to min over remaining steps
	progress := float64(step-pretrainWarmup) / float64(pretrainMaxSteps-pretrainWarmup)
	if progress > 1 {
		progress = 1
	}
	cosine := 0.5 * (1 + math.Cos(math.Pi*progress))
	lr := pretrainMinLR + (pretrainBaseLR-pretrainMinLR)*float32(cosine)
	return lr
}

// RunTransformerPretraining pretrains a Transformer as a Go causal language
// model on raw Go code. This is v1: the model learns Go syntax/patterns from
// the corpus, producing weights that fine-tuning can start from.
func RunTransformerPretraining(projectRoot string, resume bool) error {
	corpusPath := filepath.Join(projectRoot, "data", "pretrain", "go_corpus.txt")
	ckptPath, metaPath, vocabPath, finalPath := pretrainPaths(projectRoot)

	// 1. Load or build vocabulary
	var v *vocab.Vocabulary
	var curFP uint64
	startStep := 0
	best := float32(math.MaxFloat32)
	var model *transformer.Transformer

	if resume {
		// Try to load existing vocab and checkpoint
		if lv, err := loadPretrainVocab(vocabPath); err == nil {
			v = lv
			curFP = vocabFingerprint(v)
			log.Printf("[PRETRAIN] loaded vocab (%d tokens)", v.Size())
			// Try to load checkpoint
			if m, meta, err := loadTransformerCheckpoint(curFP, ckptPath, metaPath); err == nil {
				// loadTransformerCheckpoint returns ckptMeta; adapt to pretrainMeta
				model = m
				startStep = meta.Epoch + 1 // reuse Epoch field as step
				best = float32(meta.Best)
				log.Printf("[PRETRAIN] RESUMED from step %d, best=%.4f", meta.Epoch, best)
			} else {
				log.Printf("[PRETRAIN] -resume: checkpoint unusable (%v); starting fresh", err)
			}
		} else {
			log.Printf("[PRETRAIN] -resume: no vocab found (%v); building fresh", err)
		}
	}

	if v == nil {
		// Build vocabulary from corpus
		log.Printf("[PRETRAIN] reading corpus %s", corpusPath)
		data, err := os.ReadFile(corpusPath)
		if err != nil {
			return fmt.Errorf("reading corpus: %w", err)
		}
		text := string(data)
		log.Printf("[PRETRAIN] corpus: %d bytes", len(text))

		// Split into lines for vocab building (BuildGoVocab takes []string)
		lines := strings.Split(text, "\n")
		// Sample lines if too many (vocab building is O(n))
		if len(lines) > 50000 {
			// Deterministic subsample: every nth line
			step := len(lines) / 50000
			var sampled []string
			for i := 0; i < len(lines); i += step {
				sampled = append(sampled, lines[i])
			}
			lines = sampled
		}
		opts := tokenizer.DefaultGoTokenizerOptions()
		v = tokenizer.BuildGoVocab(lines, opts, pretrainVocabSize)
		log.Printf("[PRETRAIN] built vocab: %d tokens", v.Size())

		if err := savePretrainVocab(v, vocabPath); err != nil {
			return fmt.Errorf("saving vocab: %w", err)
		}
		curFP = vocabFingerprint(v)
	}

	// 2. Tokenize corpus into IDs
	log.Printf("[PRETRAIN] tokenizing corpus...")
	data, err := os.ReadFile(corpusPath)
	if err != nil {
		return fmt.Errorf("reading corpus: %w", err)
	}
	goTok := tokenizer.NewGoTokenizer(v, tokenizer.DefaultGoTokenizerOptions())
	ids, err := goTok.EncodeIDs(string(data))
	if err != nil {
		return fmt.Errorf("tokenizing corpus: %w", err)
	}
	log.Printf("[PRETRAIN] tokenized: %d tokens", len(ids))

	// 3. Chunk into sequences
	chunks := chunkCorpus(ids, pretrainSeqLen)
	log.Printf("[PRETRAIN] %d sequences of %d tokens", len(chunks), pretrainSeqLen)
	if len(chunks) == 0 {
		return fmt.Errorf("no training sequences")
	}

	// 4. Create model
	if model == nil {
		cfg := transformer.TransformerConfig{
			DModel:    512,
			NumHeads:  8,
			NumLayers: 6,
			DFF:       2048,
			MaxLen:    512,
			VocabSize: v.Size(),
		}
		model, err = transformer.NewTransformer(cfg)
		if err != nil {
			return fmt.Errorf("new transformer: %w", err)
		}
		log.Printf("[PRETRAIN] new model, ~%d params", model.NumParams())
	}

	// 5. Optimizer
	opt := nn.NewOptimizer(model.Parameters(), pretrainBaseLR, 1.0)

	vocabSize := v.Size()
	padID := v.PaddingTokenID

	// 6. Training loop (step-based, not epoch-based)
	// Shuffle chunk order deterministically
	order := make([]int, len(chunks))
	for i := range order {
		order[i] = i
	}

	step := startStep
	for step < pretrainMaxSteps {
		// Reshuffle every pass through data
		pass := step / (len(chunks) / pretrainBatchSize)
		for i := len(order) - 1; i > 0; i-- {
			j := (pass*7919 + step*104729 + i*31) % (i + 1)
			order[i], order[j] = order[j], order[i]
		}

		numBatches := len(chunks) / pretrainBatchSize
		for b := 0; b < numBatches && step < pretrainMaxSteps; b++ {
			// Build batch
			batchSeqs := make([][]int, pretrainBatchSize)
			for i := 0; i < pretrainBatchSize; i++ {
				batchSeqs[i] = chunks[order[(b*pretrainBatchSize+i)%len(chunks)]]
			}

			// Input = tokens[0:511], target = tokens[1:512]
			// For the Transformer: src = input, tgt_in = input (decoder sees prefix)
			// The decoder's causal mask ensures autoregressive prediction.
			seqLen := pretrainSeqLen - 1
			srcData := make([]float32, pretrainBatchSize*seqLen)
			tgtInData := make([]float32, pretrainBatchSize*seqLen)
			labels := make([][]int, pretrainBatchSize)
			for bi, seq := range batchSeqs {
				labels[bi] = make([]int, seqLen)
				for t := 0; t < seqLen; t++ {
					srcData[bi*seqLen+t] = float32(seq[t])
					tgtInData[bi*seqLen+t] = float32(seq[t])
					labels[bi][t] = seq[t+1]
				}
			}
			src := tensor.NewTensor([]int{pretrainBatchSize, seqLen}, srcData, false)
			tgtIn := tensor.NewTensor([]int{pretrainBatchSize, seqLen}, tgtInData, false)

			// Forward
			logits, err := model.Forward(src, tgtIn)
			if err != nil {
				log.Printf("[PRETRAIN] forward failed at step %d: %v", step, err)
				step++
				continue
			}

			// Loss
			loss, dLogits, err := transformerCrossEntropy(logits, labels, padID, vocabSize)
			if err != nil {
				step++
				continue
			}

			// NaN check
			if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
				return fmt.Errorf("loss diverged at step %d", step)
			}

			// Backward
			if err := logits.Backward(dLogits); err != nil {
				log.Printf("[PRETRAIN] backward failed at step %d: %v", step, err)
				step++
				continue
			}

			// LR schedule
			lr := pretrainLRSchedule(step)
			opt.SetLearningRate(lr)

			// Optimizer step
			opt.Step()

			// Zero grads
			for _, p := range model.Parameters() {
				if p.Grad != nil {
					for i := range p.Grad.Data {
						p.Grad.Data[i] = 0
					}
				}
			}

			if loss < best {
				best = loss
			}

			if step%100 == 0 {
				log.Printf("[PRETRAIN] step=%d loss=%.4f best=%.4f lr=%.6f", step, loss, best, lr)
			}

			// Checkpoint every 1000 steps
			if step%1000 == 0 && step > startStep {
				meta := ckptMeta{
					Epoch:              step, // reuse Epoch as step counter
					Best:               float64(best),
					LastCheckpointBest: float64(best),
					LR:                 opt.GetLearningRate(),
					EpochsNoImprove:    0,
					VocabFP:            curFP,
				}
				if err := saveTransformerCheckpoint(model, meta, ckptPath, metaPath); err != nil {
					log.Printf("[PRETRAIN] checkpoint failed: %v", err)
				} else {
					log.Printf("[PRETRAIN] checkpointed at step %d (loss=%.4f)", step, loss)
				}
			}

			step++
		}
	}

	// Save final model
	if err := model.Save(finalPath); err != nil {
		return fmt.Errorf("saving final model: %w", err)
	}
	log.Printf("[PRETRAIN] complete: %d steps, best loss=%.4f, saved to %s", step, best, finalPath)
	return nil
}
