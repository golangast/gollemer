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

// TestTrainBatchWithCopyDecreasesLoss is the copy-mechanism twin of the
// test above: with the gate enabled, TrainBatch must still run end to end
// (forward bias + backward into gate params, query and history states)
// and the loss must decrease without diverging. A previous copy attempt
// diverged to loss 8-9; this test pins the stable behavior.
func TestTrainBatchWithCopyDecreasesLoss(t *testing.T) {
	v := vocab.NewVocabulary() // pad=0, unk=1, bos=2, eos=3
	words := []string{"hello", "hi", "how", "are", "you", "i", "am", "fine", "thanks", "total", "sum"}
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
	m.Decoder.Copy = NewCopyGate(16)
	opt := nn.NewOptimizer(m.Parameters(), 1e-2, 1.0)

	// Include a repeated identifier ("total") so the copy path has
	// something to point at.
	inputs := [][]int{
		{id("hello"), id("how"), id("are"), id("you")},
		{id("hi"), id("sum"), id("total"), pad},
	}
	targets := [][]int{
		{bos, id("i"), id("am"), id("fine"), eos, pad},
		{bos, id("total"), id("sum"), id("total"), eos, pad},
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
	t.Logf("copy loss: first=%.4f last=%.4f gate=%.3f", first, last,
		m.Decoder.Copy.GateValue(make([]float32, 16)))
	if last >= first {
		t.Fatalf("copy loss did not decrease: first=%.4f last=%.4f", first, last)
	}
	if last > 5.0 {
		t.Fatalf("copy loss suspiciously high (divergence?): %.4f", last)
	}
	// The gate parameters must have received gradients.
	if m.Decoder.Copy.Wg.Grad == nil || m.Decoder.Copy.Bg.Grad == nil {
		t.Fatalf("copy gate params got no gradients")
	}
}
