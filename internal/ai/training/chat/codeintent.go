package chat

import (
	"path/filepath"
	"sort"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// CodeIntentDomain is the code-understanding domain: input is Go code plus
// a natural-language question about it, output is the answer. It is
// trained by distillation — the analyzer (teacher) generates correct
// (code+question -> answer) triples, and the model learns to do what the
// analyzer does, from data instead of hand-written rules.
const CodeIntentDomain = "codeintent"

// CodeIntentBigDomain is the bigger-model experiment (2026-10-17, John's
// "Do a and c"): 512/1024 dims vs the 256/512 v3. Tests whether capacity
// (not just architecture) is the bottleneck for code understanding.
// Trains on the same data as codeintent; saves to a separate .gob.
const CodeIntentBigDomain = "codeintent-big"

// CopyExpDomain is the copy-skill falsification experiment (2026-10-16):
// a grounding-only domain ("which file is X in" / "what is the signature
// of X") testing whether the codeintent architecture can learn precise
// copying from input to output on held-out repos. Same architecture as
// codeintent (cross-attention, copy gate, aux loss, pos enc) at 128/256
// for speed. If held-out file echo stays zero here, the architecture
// cannot learn copying and no data/param trick will fix it.
const CopyExpDomain = "copyexp"

// IsCodeIntentFamily reports whether a domain uses the codeintent model
// architecture and training pipeline (cross-attention, encoder-side copy
// gate, pointer-generator aux loss, no synonym augmentation, capped
// vocab). Both codeintent and the copyexp falsification experiment do.
func IsCodeIntentFamily(domain string) bool {
	return domain == CodeIntentDomain || domain == CopyExpDomain || domain == CodeIntentBigDomain
}

// CodeIntentDatasetRelPath is the distillation training data location
// relative to the project root.
const CodeIntentDatasetRelPath = "data/training/codeintent_pairs.jsonl"

// CodeIntentHeldoutRelPath is the held-out evaluation data: pairs from
// repos the model never trains on, measuring cross-repo generalization.
const CodeIntentHeldoutRelPath = "data/training/codeintent_heldout.jsonl"

// CodeIntentDatasetPath is the distillation training data location.
func CodeIntentDatasetPath(projectRoot string) string {
	return filepath.Join(projectRoot, CodeIntentDatasetRelPath)
}

// CopyExpDatasetRelPath is the copy-experiment training data location
// relative to the project root: grounding-only pairs.
const CopyExpDatasetRelPath = "data/training/copyexp_pairs.jsonl"

// CopyExpDatasetPath is the copy-experiment training data location.
func CopyExpDatasetPath(projectRoot string) string {
	return filepath.Join(projectRoot, CopyExpDatasetRelPath)
}

// buildCappedVocab builds a frequency-capped vocabulary for the codeintent
// domain: the long tail of rare identifiers explodes the output layer
// (256x17k) and dominates training time, so only the top-N tokens by
// frequency are kept. The ordering is fully deterministic — frequency
// descending, ties broken by token string ascending — because the training
// vocabulary is hashed into the checkpoint fingerprint that -resume uses to
// validate a checkpoint. (An earlier version sorted by frequency alone over
// randomized map iteration order, so every process built a different vocab
// and -resume always refused the checkpoint.) It lived in
// real_seq2seq_train.go on the main line; the merge kept the
// composer-pieces trainer, so it moved here with the codeintent domain's
// other definitions.
func buildCappedVocab(trainPairs []ChatPair, cap int) *vocab.Vocabulary {
	v := vocab.NewVocabulary()
	freq := map[string]int{}
	for _, p := range trainPairs {
		for _, t := range tokenizeLower(p.Input) {
			freq[t]++
		}
		for _, t := range tokenizeLower(p.Output) {
			freq[t]++
		}
	}
	type kv struct {
		tok string
		n   int
	}
	sorted := make([]kv, 0, len(freq))
	for t, n := range freq {
		sorted = append(sorted, kv{t, n})
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].n != sorted[j].n {
			return sorted[i].n > sorted[j].n
		}
		return sorted[i].tok < sorted[j].tok
	})
	for i, kv := range sorted {
		if i >= cap {
			break
		}
		v.AddToken(kv.tok)
	}
	return v
}
