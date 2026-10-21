package chat

import (
	"fmt"
	"log"
	"math"
	"os"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/transformer"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// Eval harness for the mid-transformer experiment (branch mid-transformer).
//
// RunTransformerEval loads a transformer saved by RunTransformerPairTraining
// for an experiment name and scores it on a held-out pairs file with greedy
// decoding and exact-match "echo" scoring — the gate the experiment was
// scoped around (Stage 1: can attention learn to copy a file:line answer
// for code it never trained on, where the LSTM scored 0?).
//
// Scoring definitions (fixed in advance, mirrored in the tests):
//   - Detokenization is vocab.Decode: pieces joined with single spaces,
//     stopping at EOS, skipping pad. The expected side is put through the
//     same form — canonicalOutputForm joins the expected output's subword
//     pieces identically — so an exact match means the model generated
//     exactly the expected piece sequence.
//   - Expressible: every piece of the expected output exists in the
//     checkpoint's train-built vocabulary (the same whole-output check the
//     trainer's held-out coverage report prints). An inexpressible output
//     can never be generated exactly (its missing pieces encode to UNK), so
//     the headline verdict is read on the expressible subset, with the raw
//     all-pairs number reported alongside.
//   - Held-out pairs whose encoded input exceeds the model's MaxLen are
//     skipped (the trainer drops such pairs from training too) and count
//     as misses; the count is reported.

// transformerEvalMaxGen caps greedy generation. Held-out answers in the
// grounding set are file:line strings (~15-40 pieces); 64 is generous and
// keeps a degenerate model from burning minutes per pair.
const transformerEvalMaxGen = 64

// transformerEvalSampleFailures is how many misses are printed for eyeballing.
const transformerEvalSampleFailures = 10

// canonicalOutputForm renders an expected output in the exact string form
// vocab.Decode produces for generated pieces: subword pieces joined with
// single spaces.
func canonicalOutputForm(output string) string {
	return strings.Join(subwordTokens(output), " ")
}

// outputExpressible reports whether every subword piece of output exists
// in v — the same whole-output expressibility check the trainer's
// held-out coverage report uses.
func outputExpressible(output string, v *vocab.Vocabulary) bool {
	for _, t := range subwordTokens(output) {
		if _, ok := v.WordToToken[t]; !ok {
			return false
		}
	}
	return true
}

// exactMatchOutput reports whether the generated token IDs detokenize to
// exactly the expected output's canonical form.
func exactMatchOutput(genIDs []int, expected string, v *vocab.Vocabulary) bool {
	return v.Decode(genIDs) == canonicalOutputForm(expected)
}

// argmaxLast returns the index of the largest value in vals (first on ties).
func argmaxFirst(vals []float32) int {
	best := 0
	bestVal := float32(math.Inf(-1))
	for i, val := range vals {
		if val > bestVal {
			bestVal = val
			best = i
		}
	}
	return best
}

// greedyGenerateTransformer decodes greedily from srcIDs: encode once,
// then repeatedly decode the growing prefix and take the argmax at the
// last position. Stops at EOS (or pad, which a sane model never emits as
// content) or after maxGen tokens. Returns the generated content IDs
// (no BOS/EOS).
func greedyGenerateTransformer(model *transformer.Transformer, v *vocab.Vocabulary, srcIDs []int, maxGen int) ([]int, error) {
	srcData := make([]float32, len(srcIDs))
	for i, id := range srcIDs {
		srcData[i] = float32(id)
	}
	src := tensor.NewTensor([]int{1, len(srcIDs)}, srcData, false)
	memory, err := model.Encode(src)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}

	vocabSize := v.Size()
	prefix := []int{v.BosID}
	var out []int
	for len(out) < maxGen {
		tgtData := make([]float32, len(prefix))
		for i, id := range prefix {
			tgtData[i] = float32(id)
		}
		tgt := tensor.NewTensor([]int{1, len(prefix)}, tgtData, false)
		logits, err := model.Decode(tgt, memory)
		if err != nil {
			return nil, fmt.Errorf("decode step %d: %w", len(out), err)
		}
		if len(logits.Data) < len(prefix)*vocabSize {
			return nil, fmt.Errorf("decode step %d: logits too small (%d, want >= %d)", len(out), len(logits.Data), len(prefix)*vocabSize)
		}
		base := (len(prefix) - 1) * vocabSize
		next := argmaxFirst(logits.Data[base : base+vocabSize])
		if next == v.EosID || next == v.PaddingTokenID {
			break
		}
		out = append(out, next)
		prefix = append(prefix, next)
	}
	return out, nil
}

// loadTransformerForEval resolves the vocabulary and weights for an
// experiment, reusing the trainer's artifact scheme (transformerExpPaths)
// and checkpoint loader. Weight preference: the final saved model, then
// the best checkpoint, then the latest checkpoint (both via the trainer's
// fingerprint-checked loader). The vocabulary comes from the trainer's
// saved vocab file when present, else is rebuilt from pairsPath exactly
// as the trainer builds it. When no weights exist at all, it returns a
// freshly initialized UNTRAINED model (config M) so the eval pipeline can
// be plumbing-checked before any training run; the label says so.
func loadTransformerForEval(projectRoot, expName, pairsPath string) (model *transformer.Transformer, v *vocab.Vocabulary, label string, err error) {
	finalPath, ckptPath, metaPath, latestPath, latestMetaPath, vocabPath := transformerExpPaths(projectRoot, expName)

	if _, statErr := os.Stat(vocabPath); statErr == nil {
		v, err = vocab.LoadVocabulary(vocabPath)
		if err != nil {
			return nil, nil, "", fmt.Errorf("loading saved vocab %s: %w", vocabPath, err)
		}
	} else if pairsPath != "" {
		var pairs []ChatPair
		pairs, err = LoadChatDataset(pairsPath)
		if err != nil {
			return nil, nil, "", fmt.Errorf("loading pairs to rebuild vocab: %w", err)
		}
		if len(pairs) == 0 {
			return nil, nil, "", fmt.Errorf("no pairs in %s to rebuild vocab from", pairsPath)
		}
		v = buildTransformerVocab(pairs)
	} else {
		return nil, nil, "", fmt.Errorf("no saved vocab at %s and no -pairs file to rebuild it from", vocabPath)
	}

	if _, statErr := os.Stat(finalPath); statErr == nil {
		model, err = transformer.LoadTransformer(finalPath)
		if err != nil {
			return nil, nil, "", fmt.Errorf("loading final model %s: %w", finalPath, err)
		}
		if model.Config.VocabSize != v.Size() {
			return nil, nil, "", fmt.Errorf("final model vocab size %d != vocab %d", model.Config.VocabSize, v.Size())
		}
		return model, v, "final model " + finalPath, nil
	}

	curFP := vocabFingerprint(v)
	if m, meta, lerr := loadTransformerCheckpoint(curFP, ckptPath, metaPath); lerr == nil {
		return m, v, fmt.Sprintf("best checkpoint %s (epoch %d, best %.4f)", ckptPath, meta.Epoch, meta.Best), nil
	}
	if m, meta, lerr := loadTransformerCheckpoint(curFP, latestPath, latestMetaPath); lerr == nil {
		return m, v, fmt.Sprintf("latest checkpoint %s (epoch %d, best %.4f)", latestPath, meta.Epoch, meta.Best), nil
	}

	model, err = transformer.NewTransformer(transformerConfigM(v.Size()))
	if err != nil {
		return nil, nil, "", fmt.Errorf("initializing untrained model: %w", err)
	}
	return model, v, "UNTRAINED fresh init (no checkpoint found for exp " + expName + ")", nil
}

// RunTransformerEval scores the transformer for expName on the held-out
// pairs file with greedy decoding and exact-match echo scoring. pairsPath
// is only used to rebuild the vocabulary when the trainer's saved vocab
// file is absent (e.g. the untrained plumbing check).
func RunTransformerEval(projectRoot, expName, heldoutPath, pairsPath string) error {
	if heldoutPath == "" {
		return fmt.Errorf("-eval-transformer needs -heldout <heldout.jsonl>")
	}
	held, err := LoadChatDataset(heldoutPath)
	if err != nil {
		return fmt.Errorf("loading held-out pairs: %w", err)
	}
	if len(held) == 0 {
		return fmt.Errorf("no held-out pairs in %s", heldoutPath)
	}

	model, v, label, err := loadTransformerForEval(projectRoot, expName, pairsPath)
	if err != nil {
		return err
	}

	fmt.Printf("[EVAL] exp=%q weights: %s\n", expName, label)
	fmt.Printf("[EVAL] vocab=%d held-out pairs=%d maxlen=%d max-gen=%d\n", v.Size(), len(held), model.Config.MaxLen, transformerEvalMaxGen)

	var expressible, matchAll, matchExpressible, skippedTooLong, genErrors int
	type sampleFailure struct{ input, expected, got string }
	var failures []sampleFailure

	for i, p := range held {
		expr := outputExpressible(p.Output, v)
		if expr {
			expressible++
		}

		srcIDs := encodeSubword(p.Input, v)
		matched := false
		var got string
		switch {
		case len(srcIDs) == 0 || len(srcIDs) > model.Config.MaxLen:
			skippedTooLong++
			got = "<input too long for MaxLen; skipped>"
		default:
			genIDs, gerr := greedyGenerateTransformer(model, v, srcIDs, transformerEvalMaxGen)
			if gerr != nil {
				genErrors++
				got = "<generation error: " + gerr.Error() + ">"
			} else {
				got = v.Decode(genIDs)
				matched = exactMatchOutput(genIDs, p.Output, v)
			}
		}
		if matched {
			matchAll++
			if expr {
				matchExpressible++
			}
		} else if len(failures) < transformerEvalSampleFailures {
			failures = append(failures, sampleFailure{input: p.Input, expected: p.Output, got: got})
		}

		if (i+1)%200 == 0 {
			log.Printf("[EVAL] %d/%d pairs scored", i+1, len(held))
		}
	}

	pct := func(n, d int) float64 {
		if d == 0 {
			return 0
		}
		return 100 * float64(n) / float64(d)
	}
	fmt.Printf("\n[EVAL] results for exp=%q (%s)\n", expName, label)
	fmt.Printf("[EVAL] total pairs:            %d\n", len(held))
	fmt.Printf("[EVAL] expressible outputs:    %d (%.1f%% of total)\n", expressible, pct(expressible, len(held)))
	fmt.Printf("[EVAL] exact matches overall:  %d (%.1f%% of %d)\n", matchAll, pct(matchAll, len(held)), len(held))
	fmt.Printf("[EVAL] exact matches expressible subset: %d (%.1f%% of %d)\n", matchExpressible, pct(matchExpressible, expressible), expressible)
	if skippedTooLong > 0 {
		fmt.Printf("[EVAL] skipped (input > MaxLen, counted as misses): %d\n", skippedTooLong)
	}
	if genErrors > 0 {
		fmt.Printf("[EVAL] generation errors (counted as misses): %d\n", genErrors)
	}

	if len(failures) > 0 {
		fmt.Printf("\n[EVAL] sample failures (up to %d):\n", transformerEvalSampleFailures)
		for j, f := range failures {
			fmt.Printf("--- failure %d\n  input:    %s\n  expected: %s\n  got:      %s\n",
				j+1, truncateForDisplay(f.input, 200), f.expected, f.got)
		}
	}
	return nil
}

// truncateForDisplay shortens s to at most n runes for eyeball output.
func truncateForDisplay(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}
