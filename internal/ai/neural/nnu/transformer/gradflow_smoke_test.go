package transformer

import (
	"math/rand"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// TestGradientFlowSmoke verifies that a full forward+backward pass delivers
// nonzero gradients to every trainable parameter. A parameter with nil/zero
// grad after backward means the autograd graph is broken somewhere upstream.
func TestGradientFlowSmoke(t *testing.T) {
	cfg := TransformerConfig{
		DModel:    16,
		NumHeads:  2,
		NumLayers: 1,
		DFF:       32,
		MaxLen:    16,
		VocabSize: 50,
	}
	model, err := NewTransformer(cfg)
	if err != nil {
		t.Fatalf("NewTransformer: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	batch, srcSeq, tgtSeq := 2, 8, 6
	srcData := make([]float32, batch*srcSeq)
	tgtData := make([]float32, batch*tgtSeq)
	for i := range srcData {
		srcData[i] = float32(rng.Intn(cfg.VocabSize))
	}
	for i := range tgtData {
		tgtData[i] = float32(rng.Intn(cfg.VocabSize))
	}
	src := tensor.NewTensor([]int{batch, srcSeq}, srcData, false)
	tgt := tensor.NewTensor([]int{batch, tgtSeq}, tgtData, false)

	logits, err := model.Forward(src, tgt)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Seed gradient: dL/dlogits = ones / N
	dData := make([]float32, len(logits.Data))
	for i := range dData {
		dData[i] = 1.0 / float32(len(dData))
	}
	dLogits := tensor.NewTensor(logits.Shape, dData, false)
	if err := logits.Backward(dLogits); err != nil {
		t.Fatalf("Backward: %v", err)
	}

	params := model.Parameters()
	dead := 0
	for i, p := range params {
		if p.Grad == nil {
			t.Errorf("param %d (shape %v): nil grad — no gradient flow", i, p.Shape)
			dead++
			continue
		}
		nonzero := 0
		for _, g := range p.Grad.Data {
			if g != 0 {
				nonzero++
			}
		}
		if nonzero == 0 {
			t.Errorf("param %d (shape %v): all-zero grad", i, p.Shape)
			dead++
		}
	}
	t.Logf("params: %d total, %d dead", len(params), dead)
}

// TestTinyTrainLossDrops runs a few real Adam steps on a tiny Transformer and
// asserts the loss actually decreases — the end-to-end proof that the custom
// attention/FFN/norm backward path trains.
func TestTinyTrainLossDrops(t *testing.T) {
	cfg := TransformerConfig{
		DModel:    16,
		NumHeads:  2,
		NumLayers: 1,
		DFF:       32,
		MaxLen:    16,
		VocabSize: 50,
	}
	model, err := NewTransformer(cfg)
	if err != nil {
		t.Fatalf("NewTransformer: %v", err)
	}

	rng := rand.New(rand.NewSource(7))
	batch, srcSeq, tgtSeq := 2, 8, 6
	srcData := make([]float32, batch*srcSeq)
	tgtData := make([]float32, batch*tgtSeq)
	for i := range srcData {
		srcData[i] = float32(rng.Intn(cfg.VocabSize))
	}
	for i := range tgtData {
		tgtData[i] = float32(rng.Intn(cfg.VocabSize))
	}
	src := tensor.NewTensor([]int{batch, srcSeq}, srcData, false)
	tgt := tensor.NewTensor([]int{batch, tgtSeq}, tgtData, false)

	opt := nn.NewOptimizer(model.Parameters(), 1e-3, 1.0)

	// Loss = mean(logits^2): drives all outputs toward zero.
	lossOf := func() (float64, *tensor.Tensor, error) {
		logits, err := model.Forward(src, tgt)
		if err != nil {
			return 0, nil, err
		}
		var sum float64
		for _, v := range logits.Data {
			sum += float64(v * v)
		}
		return sum / float64(len(logits.Data)), logits, nil
	}

	loss0, _, err := lossOf()
	if err != nil {
		t.Fatalf("initial forward: %v", err)
	}

	steps := 20
	for s := 0; s < steps; s++ {
		loss, logits, err := lossOf()
		if err != nil {
			t.Fatalf("step %d forward: %v", s, err)
		}
		_ = loss
		// dL/dlogits = 2*logits/N
		n := float32(len(logits.Data))
		dData := make([]float32, len(logits.Data))
		for i, v := range logits.Data {
			dData[i] = 2 * v / n
		}
		dLogits := tensor.NewTensor(logits.Shape, dData, false)
		if err := logits.Backward(dLogits); err != nil {
			t.Fatalf("step %d backward: %v", s, err)
		}
		opt.Step()
		opt.ZeroGrad()
	}

	lossN, _, err := lossOf()
	if err != nil {
		t.Fatalf("final forward: %v", err)
	}
	t.Logf("loss: %.6f -> %.6f over %d steps", loss0, lossN, steps)
	if lossN >= loss0 {
		t.Errorf("loss did not decrease: %.6f -> %.6f", loss0, lossN)
	}
}
