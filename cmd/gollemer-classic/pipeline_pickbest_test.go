package main

import (
	"testing"

	"github.com/golangast/gollemer/pkg/synthesis"
)

// TestPickBestPenaltyFollowsCandidateID is a regression test for a real
// bug: penalties were built in survivor (generation) order, but pickBest
// applied them by ranked position — so after MCTS re-sorted the nodes,
// the wrong candidates were penalized. Penalties must follow the
// candidate ID ("candidate-N" in survivor order), not the rank.
func TestPickBestPenaltyFollowsCandidateID(t *testing.T) {
	// Survivor order: candidate-0 has a WARNING penalty of 25,
	// candidate-1 is clean. Ranked order is the reverse: the clean
	// candidate scored higher and sorts first.
	ranked := []synthesis.CandidateNode{
		{ID: "candidate-1", Score: 70},
		{ID: "candidate-0", Score: 60},
	}
	penalties := []int{25, 0} // survivor order: candidate-0 penalized

	best := pickBest(ranked, penalties)
	if best.node.ID != "candidate-1" {
		t.Errorf("winner = %s, want candidate-1 (penalty misapplied to ranked position)", best.node.ID)
	}
	if best.final != 70 {
		t.Errorf("winner final = %v, want 70 (no penalty on clean candidate)", best.final)
	}
	if best.index != 1 {
		t.Errorf("winner index = %d, want 1 (survivor index of candidate-1)", best.index)
	}

	// Sanity: without the ID mapping the old code would have charged
	// the 25-point penalty to ranked[0] (candidate-1) and picked
	// candidate-0 with final 35 — the exact failure this guards.
}

func TestPickBestUnknownIDGetsNoPenalty(t *testing.T) {
	ranked := []synthesis.CandidateNode{
		{ID: "weird-id", Score: 50},
	}
	best := pickBest(ranked, []int{25})
	if best.final != 50 {
		t.Errorf("final = %v, want 50 (unrecognized ID must not take another candidate's penalty)", best.final)
	}
	if best.index != -1 {
		t.Errorf("index = %d, want -1 for unrecognized ID", best.index)
	}
}
