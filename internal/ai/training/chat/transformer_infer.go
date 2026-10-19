package chat

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/transformer"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// RunTransformerTest loads a trained Transformer for a domain and runs a single
// prompt through it with greedy decoding. Used for quick early checks.
func RunTransformerTest(projectRoot, domain, prompt string) error {
	// Load vocab via domain data prep (builds the same vocab as training)
	dd, err := prepareDomainTrainData(projectRoot, domain)
	if err != nil {
		return fmt.Errorf("prepare data: %w", err)
	}
	v, tok := dd.v, dd.tok

	// Load the model checkpoint (prefer .latest, fall back to .ckpt)
	ckptPath, metaPath := transformerCkptPaths(projectRoot, domain)
	latestPath, latestMetaPath := transformerLatestCkptPaths(projectRoot, domain)
	curFP := vocabFingerprint(v)

	var model *transformer.Transformer
	// Try latest first
	if m, meta, err := loadTransformerCheckpoint(curFP, latestPath, latestMetaPath); err == nil && meta.Epoch > 0 {
		model = m
		log.Printf("[TRANSFORMER-TEST] loaded latest checkpoint (epoch %d)", meta.Epoch)
	} else if m, meta, err := loadTransformerCheckpoint(curFP, ckptPath, metaPath); err == nil {
		model = m
		log.Printf("[TRANSFORMER-TEST] loaded checkpoint (epoch %d)", meta.Epoch)
	} else {
		// Try loading without fingerprint check (early checkpoints may differ)
		if _, err := os.Stat(latestPath); err == nil {
			if m, err := transformer.LoadTransformer(latestPath); err == nil {
				model = m
				log.Printf("[TRANSFORMER-TEST] loaded latest (no fingerprint check)")
			}
		}
		if model == nil {
			if _, err := os.Stat(ckptPath); err == nil {
				if m, err := transformer.LoadTransformer(ckptPath); err == nil {
					model = m
					log.Printf("[TRANSFORMER-TEST] loaded ckpt (no fingerprint check)")
				}
			}
		}
	}
	if model == nil {
		return fmt.Errorf("no transformer checkpoint found for domain %q (looked in %s)", domain, filepath.Dir(ckptPath))
	}

	// Tokenize the prompt
	srcIDs := encodeInput(tok, prompt)
	if len(srcIDs) == 0 {
		return fmt.Errorf("prompt produced no tokens")
	}
	// Build src tensor: [1, seqLen]
	srcData := make([]float32, len(srcIDs))
	for i, id := range srcIDs {
		srcData[i] = float32(id)
	}
	src := tensor.NewTensor([]int{1, len(srcIDs)}, srcData, false)

	// Greedy decode with repetition penalty
	maxLen := 128
	// Start with BOS (assume 1 = BOS, 2 = EOS; adjust if vocab differs)
	bosID := 1
	eosID := 2
	genIDs := []int{bosID}
	// Track generated tokens for repetition penalty
	seen := make(map[int]int)
	seen[bosID] = 1

	for len(genIDs) < maxLen {
		tgtData := make([]float32, len(genIDs))
		for i, id := range genIDs {
			tgtData[i] = float32(id)
		}
		tgt := tensor.NewTensor([]int{1, len(genIDs)}, tgtData, false)

		logits, err := model.Forward(src, tgt)
		if err != nil {
			return fmt.Errorf("forward: %w", err)
		}
		// logits shape: [1, tgtLen, vocabSize]; take last position
		vocabSize := v.Size()
		tgtLen := len(genIDs)
		offset := (tgtLen - 1) * vocabSize
		bestID := 0
		bestVal := float32(-1e30)
		for i := 0; i < vocabSize && offset+i < len(logits.Data); i++ {
			val := logits.Data[offset+i]
			// Repetition penalty: reduce score for already-seen tokens
			if count, ok := seen[i]; ok {
				val /= float32(1 + count) // divide by 1 + times seen
			}
			if val > bestVal {
				bestVal = val
				bestID = i
			}
		}
		if bestID == eosID {
			break
		}
		// Stop if repeating the same token 5+ times in a row
		if len(genIDs) >= 5 {
			allSame := true
			for j := len(genIDs) - 5; j < len(genIDs); j++ {
				if genIDs[j] != bestID {
					allSame = false
					break
				}
			}
			if allSame && genIDs[len(genIDs)-1] == bestID {
				break
			}
		}
		genIDs = append(genIDs, bestID)
		seen[bestID]++
	}

	// Detokenize (skip BOS)
	outIDs := genIDs[1:]
	out := v.Decode(outIDs)
	fmt.Printf("\n=== Transformer (%s) output ===\n%s\n================================\n", domain, out)
	return nil
}
