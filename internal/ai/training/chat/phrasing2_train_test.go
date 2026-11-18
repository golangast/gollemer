package chat

import (
	"math/rand"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// phrasing2Root resolves the repo root from this file's
// location (internal/ai/training/chat -> ../../../..).
func phrasing2Root(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "..")
}

// TestPhrasing2DataPrepAndEncoding exercises the real v2 data:
// the split carves a validation slice, the vocabulary carries
// the indexed placeholders, and masked encodings actually use
// them (the arm-A mechanism, on real pairs).
func TestPhrasing2DataPrepAndEncoding(t *testing.T) {
	root := phrasing2Root(t)
	train, val, v, _, err := preparePhrasing2TrainData(root)
	if err != nil {
		t.Skipf("phrasing2 data unavailable: %v", err)
	}
	if len(val) == 0 || len(train) < 10000 {
		t.Fatalf("split = %d train / %d val, want >10000 train and a nonempty val slice", len(train), len(val))
	}
	for k := 1; k <= seq2seq.MaxIndexedUnk; k++ {
		if id := v.GetTokenID(seq2seq.IndexedUnkToken(k)); id == v.UnkID {
			t.Fatalf("vocabulary is missing placeholder %s", seq2seq.IndexedUnkToken(k))
		}
	}
	// The validation slice must be training data, never the
	// judge: spot-check that validation inputs are messy
	// sentences whose outputs are canonical (both files share
	// the shapes, but the split itself is the guarantee — the
	// judge file is never opened by the trainer).
	rng := rand.New(rand.NewSource(7))
	withPlaceholder, maskedPairs := 0, 0
	for _, p := range train[:2000] {
		ep := encodePhrasing2Pair(v, p, rng)
		seen := map[int]bool{}
		for _, id := range ep.target {
			if w := v.GetWord(id); strings.HasPrefix(w, "unk") && w != "unk" {
				seen[id] = true
			}
		}
		if len(seen) > 0 {
			maskedPairs++
		}
		if len(seen) > 1 {
			withPlaceholder++ // at least one pair distinguishes 2+ unknowns
		}
	}
	if maskedPairs == 0 {
		t.Fatal("no masked pairs in the first 2000: the masking discipline is not firing")
	}
	if withPlaceholder == 0 {
		t.Fatal("no pair used 2+ distinct placeholders: indexed disambiguation is not exercised")
	}
}

// TestPhrasing2IndexedMaskingConsistency: in an all-slots
// masked encoding of a two-name ask, the input and the target
// must use the SAME placeholder for the same value, and
// different placeholders for the two values.
func TestPhrasing2IndexedMaskingConsistency(t *testing.T) {
	root := phrasing2Root(t)
	_, _, v, _, err := preparePhrasing2TrainData(root)
	if err != nil {
		t.Skipf("phrasing2 data unavailable: %v", err)
	}
	p := phrasingPair{
		Input:  "add instantiation of struct jim to handler jake",
		Output: "add instantiation of struct Jim to handler Jake",
		Domain: PhrasingDomain,
		Slots:  []string{"Jim", "Jake"},
	}
	// Force the all-slots mode by trying seeds until the
	// encoding masks (mode is rng-driven per pair).
	for seed := int64(0); seed < 200; seed++ {
		rng := rand.New(rand.NewSource(seed))
		ep := encodePhrasing2Pair(v, p, rng)
		words := map[int]string{}
		hasPlaceholder := false
		for _, id := range ep.target {
			w := v.GetWord(id)
			words[id] = w
			if strings.HasPrefix(w, "unk") && w != "unk" {
				hasPlaceholder = true
			}
		}
		if !hasPlaceholder {
			continue
		}
		// One-slot mode also produces a placeholder; this test
		// wants the all-slots mode: neither name may survive
		// literally in the target.
		jimID, jakeID := v.GetTokenID("jim"), v.GetTokenID("jake")
		literal := false
		for _, id := range ep.target {
			if id == jimID || id == jakeID {
				literal = true
			}
		}
		if literal {
			continue
		}
		// Decode target ids back to words and check: exactly
		// two distinct placeholders, each repeated for its
		// value's tokens, and the input uses the same ids.
		distinct := map[int]bool{}
		for _, id := range ep.target {
			if strings.HasPrefix(words[id], "unk") && words[id] != "unk" {
				distinct[id] = true
			}
		}
		if len(distinct) != 2 {
			t.Fatalf("seed %d: target uses %d distinct placeholders, want 2 (ids %v)", seed, len(distinct), ep.target)
		}
		inSet := map[int]bool{}
		for _, id := range ep.input {
			inSet[id] = true
		}
		for id := range distinct {
			if !inSet[id] {
				t.Fatalf("seed %d: placeholder %s in target but not in input", seed, words[id])
			}
		}
		return
	}
	t.Fatal("no seed produced a masked encoding — masking never fired")
}
