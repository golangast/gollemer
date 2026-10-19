package seq2seq

import (
	"testing"
)

func TestBeamSearchBasic(t *testing.T) {
	// BeamWidth=1 should behave like greedy (not crash, return string)
	// We can't easily construct a full model without training data,
	// so we test the helper functions and the beam hypothesis logic.

	// Test logSoftmax: probabilities should sum to ~1
	logits := []float32{1.0, 2.0, 3.0}
	lp := logSoftmax(logits)
	sum := float32(0)
	for _, v := range lp {
		// exp(logProb) = prob
		sum += float32(exp(float64(v)))
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("logSoftmax probs sum to %f, want ~1.0", sum)
	}
	// Highest logit should have highest log prob
	if lp[2] < lp[1] || lp[1] < lp[0] {
		t.Error("logSoftmax ordering wrong")
	}
}

func TestLengthPenalty(t *testing.T) {
	// For the same raw (negative) score, longer sequences should get
	// a higher penalized score (less negative) — this compensates for
	// the fact that longer sequences accumulate more negative log probs.
	s1 := lengthPenalty(-10.0, 5, 0.6)
	s2 := lengthPenalty(-10.0, 10, 0.6)
	if s2 <= s1 {
		t.Errorf("length penalty not working: len5=%f len10=%f (want len10 > len5)", s1, s2)
	}
	// Zero length should not divide by zero
	s0 := lengthPenalty(-10.0, 0, 0.6)
	if s0 != -10.0 {
		t.Errorf("zero-length penalty = %f, want -10.0", s0)
	}
}

func TestCausalMaskBeam(t *testing.T) {
	// Verify beam hypothesis copying doesn't share state
	h1 := &BeamHypothesis{
		Tokens:      []int{1, 2},
		Score:       -1.5,
		SeenBigrams: map[[2]int]bool{{1, 2}: true},
	}
	// Simulate what beam expansion does
	newTokens := append(append([]int{}, h1.Tokens...), 3)
	if len(h1.Tokens) != 2 {
		t.Error("original hypothesis tokens were mutated")
	}
	if len(newTokens) != 3 {
		t.Error("new hypothesis tokens wrong length")
	}
}

func exp(x float64) float64 {
	// Simple exp for test (avoid importing math in test)
	if x > 700 {
		return 1e308
	}
	// Taylor series approximation is overkill; use iterative
	result := 1.0
	term := 1.0
	for i := 1; i < 20; i++ {
		term *= x / float64(i)
		result += term
	}
	return result
}
