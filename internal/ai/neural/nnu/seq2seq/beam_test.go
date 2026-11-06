package seq2seq

// Beam-search test: with BeamWidth > 1, PredictWithTrace must be fully
// deterministic (same model + input -> same reply, twice) and the
// thought trace must describe the winning beam's path — one step per
// emitted token, routing recorded per step, expert usage normalized.

import (
	"math"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func TestPredictBeamDeterministicWithTrace(t *testing.T) {
	v := vocab.NewVocabulary() // pad=0, unk=1, bos=2, eos=3
	words := []string{"hello", "hi", "how", "are", "you", "i", "am", "fine", "thanks"}
	for _, w := range words {
		v.AddToken(w)
	}
	pad := v.PaddingTokenID
	bos := v.BosID
	eos := v.EosID
	id := func(w string) int { return v.GetTokenID(w) }

	m, err := NewSeq2Seq(v.Size(), v.Size(), 8, 16, nil, v)
	if err != nil {
		t.Fatalf("NewSeq2Seq: %v", err)
	}
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		t.Fatalf("NewTokenizer: %v", err)
	}
	m.Tokenizer = tok

	// A few training steps so the model emits more than immediate EOS.
	inputs := [][]int{{id("hello"), id("how"), id("are"), id("you")}}
	targets := [][]int{{bos, id("i"), id("am"), id("fine"), eos}}
	opt := nn.NewOptimizer(m.Parameters(), 1e-2, 1.0)
	for i := 0; i < 15; i++ {
		opt.ZeroGrad()
		if _, err := TrainBatch(m, inputs, targets, pad); err != nil {
			t.Fatalf("TrainBatch step %d: %v", i, err)
		}
		opt.ClipGradients()
		opt.Step()
	}

	m.BeamWidth = 4
	answer1, trace1, err := m.PredictWithTrace("hello how are you", 10)
	if err != nil {
		t.Fatalf("PredictWithTrace (beam): %v", err)
	}
	answer2, trace2, err := m.PredictWithTrace("hello how are you", 10)
	if err != nil {
		t.Fatalf("PredictWithTrace (beam, 2nd): %v", err)
	}

	if answer1 != answer2 {
		t.Fatalf("beam decode not deterministic: %q vs %q", answer1, answer2)
	}
	if trace1 == nil || trace2 == nil {
		t.Fatal("beam trace is nil")
	}
	if len(trace1.Steps) == 0 {
		t.Fatal("beam trace has no steps")
	}
	if len(trace1.Steps) != len(trace2.Steps) {
		t.Fatalf("beam trace lengths differ: %d vs %d", len(trace1.Steps), len(trace2.Steps))
	}
	for i, s := range trace1.Steps {
		if s.TokenID != trace2.Steps[i].TokenID || s.Token != trace2.Steps[i].Token {
			t.Fatalf("step %d differs between runs: (%d,%q) vs (%d,%q)",
				i, s.TokenID, s.Token, trace2.Steps[i].TokenID, trace2.Steps[i].Token)
		}
		if len(s.Experts) != 2 {
			t.Fatalf("step %d: experts = %v, want top-2", i, s.Experts)
		}
		var g float32
		for _, w := range s.Gates {
			g += w
		}
		if math.Abs(float64(g-1)) > 1e-5 {
			t.Fatalf("step %d: gates sum = %v, want 1", i, g)
		}
		if len(s.Candidates) == 0 {
			t.Fatalf("step %d: no candidates recorded", i)
		}
		if s.Token == "" {
			t.Fatalf("step %d: empty token text", i)
		}
	}
	if trace1.NumExperts != 4 {
		t.Fatalf("NumExperts = %d, want 4", trace1.NumExperts)
	}
	var total float32
	for _, u := range trace1.ExpertUsage {
		total += u
	}
	if math.Abs(float64(total-1)) > 1e-5 {
		t.Fatalf("expert usage sums to %v, want 1", total)
	}
	t.Logf("answer=%q steps=%d", answer1, len(trace1.Steps))
}
