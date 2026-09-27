package seq2seq

// Thought-trace test: PredictWithTrace must return one routing step per
// generated token, with normalized gate weights and a sane expert histogram.

import (
	"math"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func TestPredictWithTraceCapturesRouting(t *testing.T) {
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

	answer, trace, err := m.PredictWithTrace("hello how are you", 10)
	if err != nil {
		t.Fatalf("PredictWithTrace: %v", err)
	}
	if trace == nil {
		t.Fatal("trace is nil")
	}
	if len(trace.Steps) == 0 {
		t.Fatal("trace has no steps")
	}
	if trace.NumExperts != 4 {
		t.Fatalf("NumExperts = %d, want 4", trace.NumExperts)
	}
	for i, s := range trace.Steps {
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
		for _, e := range s.Experts {
			if e < 0 || e >= 4 {
				t.Fatalf("step %d: expert index %d out of range", i, e)
			}
		}
		if len(s.Candidates) == 0 {
			t.Fatalf("step %d: no candidates recorded", i)
		}
		if s.Token == "" {
			t.Fatalf("step %d: empty token text", i)
		}
	}
	var total float32
	for _, u := range trace.ExpertUsage {
		total += u
	}
	if math.Abs(float64(total-1)) > 1e-5 {
		t.Fatalf("expert usage sums to %v, want 1", total)
	}
	t.Logf("answer=%q steps=%d", answer, len(trace.Steps))
}
