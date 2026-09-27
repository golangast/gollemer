package seq2seq

// Gradient-flow regression test: a few TrainBatch steps on a tiny model must
// reduce the loss. If the BPTT wiring breaks (nil grads, wrong shapes,
// severed encoder/decoder bridge), the loss will not move.

import (
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func TestTrainBatchDecreasesLoss(t *testing.T) {
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
	opt := nn.NewOptimizer(m.Parameters(), 1e-2, 1.0)

	// Two tiny pairs, padded to uniform shapes.
	inputs := [][]int{
		{id("hello"), id("how"), id("are"), id("you")},
		{id("hi"), pad, pad, pad},
	}
	targets := [][]int{
		{bos, id("i"), id("am"), id("fine"), eos},
		{bos, id("hi"), eos, pad, pad},
	}

	opt.ZeroGrad()
	first, err := TrainBatch(m, inputs, targets, pad)
	if err != nil {
		t.Fatalf("first TrainBatch: %v", err)
	}
	opt.ClipGradients()
	opt.Step()

	var last float32
	for i := 0; i < 30; i++ {
		opt.ZeroGrad()
		loss, err := TrainBatch(m, inputs, targets, pad)
		if err != nil {
			t.Fatalf("TrainBatch step %d: %v", i, err)
		}
		opt.ClipGradients()
		opt.Step()
		last = loss
	}
	t.Logf("loss: first=%.4f last=%.4f", first, last)
	if last >= first {
		t.Fatalf("loss did not decrease after training: first=%.4f last=%.4f", first, last)
	}
}
