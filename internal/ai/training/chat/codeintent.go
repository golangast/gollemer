package chat

import (
	"path/filepath"
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
