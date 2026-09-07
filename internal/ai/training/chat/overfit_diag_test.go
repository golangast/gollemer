package chat

import (
	"log"
	"testing"

	"github.com/golangast/gollemer/internal/ai/moe"
	neuralnn "github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// TestOverfitSinglePair drives the real IntentMoE + WeightedCrossEntropy +
// Adam stack on ONE tiny pair. If the training plumbing is correct, loss
// should collapse well below 1.0. If it plateaus near 3.0 (the word-salad
// plateau seen in logtrain.txt), the bug is in Forward/Backward/Parameters.
func TestOverfitSinglePair(t *testing.T) {
	vocab := mainvocab.NewVocabulary()
	for _, tok := range []string{"hello", "there", "<EOS>"} {
		vocab.AddToken(tok)
	}
	v := vocab.GetTokenID("hello")
	h := vocab.GetTokenID("there")
	e := vocab.EosID
	if e < 0 {
		e = vocab.GetTokenID("<EOS>")
	}
	t.Logf("ids: hello=%d there=%d eos=%d vocabSize=%d", v, h, e, vocab.Size())

	m, err := moe.NewHybridIntentMoE(vocab.Size(), 64, 4, vocab.Size(), vocab.Size(), vocab.Size(), 2)
	if err != nil {
		t.Fatalf("model: %v", err)
	}
	m.Decoder, _ = moe.NewRNNDecoder(64, vocab.Size(), 64, 2, 1, 0.0, 4)
	m.RepairArchitecture()
	m.RebuildActiveLayers()
	m.SentenceVocab = vocab
	m.SentenceVocabSize = vocab.Size()
	m.Decoder.ResizeOutputLayer(vocab.Size())
	m.ResizeEmbeddings(vocab.Size())

	// input:  [BOS hello there] -> target: [hello there EOS]
	bos := vocab.BosID
	if bos < 0 {
		bos = vocab.GetTokenID("<BOS>")
	}
	input := tensor.NewTensor([]int{1, 3}, []float32{float32(bos), float32(v), float32(h)}, false)
	target := tensor.NewTensor([]int{1, 3}, []float32{float32(v), float32(h), float32(e)}, false)

	opt := neuralnn.NewOptimizer(m.Parameters(), 0.005, 1.0)
	weights := make([]float32, vocab.Size())
	for i := range weights {
		weights[i] = 1.0
	}

	for epoch := 0; epoch < 300; epoch++ {
		opt.ZeroGrad()
		logits, _, err := m.Forward(0.0, input, target)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		if len(logits) != 1 {
			t.Fatalf("expected 3D logits path, got %d logits", len(logits))
		}
		targets := []int{v, h, e}
		loss, grad := WeightedCrossEntropy(logits[0].ToCPU(), targets, weights, 0, 0)
		if err := m.Backward(grad); err != nil {
			t.Fatalf("backward: %v", err)
		}
		opt.ClipGradients()
		opt.Step()
		m.ClearState()
		if epoch%25 == 0 || epoch == 299 {
			log.Printf("epoch %d loss %.4f", epoch, loss)
		}
		if epoch == 299 && loss > 1.0 {
			t.Errorf("loss did not converge: %.4f (still > 1.0 after 300 steps)", loss)
		}
	}
}

// TestOverfitStableRouter is identical to TestOverfitSinglePair but pins the
// router learning rate to the same value as the body LR. If this converges
// while the default (RouterLR = 15x) does not, the router is thrashing.
func TestOverfitStableRouter(t *testing.T) {
	vocab := mainvocab.NewVocabulary()
	for _, tok := range []string{"hello", "there", "<EOS>"} {
		vocab.AddToken(tok)
	}
	v := vocab.GetTokenID("hello")
	h := vocab.GetTokenID("there")
	e := vocab.EosID
	if e < 0 {
		e = vocab.GetTokenID("<EOS>")
	}

	m, err := moe.NewHybridIntentMoE(vocab.Size(), 64, 4, vocab.Size(), vocab.Size(), vocab.Size(), 2)
	if err != nil {
		t.Fatalf("model: %v", err)
	}
	m.Decoder, _ = moe.NewRNNDecoder(64, vocab.Size(), 64, 2, 1, 0.0, 4)
	m.RepairArchitecture()
	m.RebuildActiveLayers()
	m.SentenceVocab = vocab
	m.SentenceVocabSize = vocab.Size()
	m.Decoder.ResizeOutputLayer(vocab.Size())
	m.ResizeEmbeddings(vocab.Size())

	bos := vocab.BosID
	if bos < 0 {
		bos = vocab.GetTokenID("<BOS>")
	}
	input := tensor.NewTensor([]int{1, 3}, []float32{float32(bos), float32(v), float32(h)}, false)
	target := tensor.NewTensor([]int{1, 3}, []float32{float32(v), float32(h), float32(e)}, false)

	opt := neuralnn.NewOptimizer(m.Parameters(), 0.005, 1.0)
	if s, ok := opt.(interface{ SetRouterLR(lr float32) }); ok {
		s.SetRouterLR(0.005) // same as body LR
	}
	weights := make([]float32, vocab.Size())
	for i := range weights {
		weights[i] = 1.0
	}
	for epoch := 0; epoch < 300; epoch++ {
		opt.ZeroGrad()
		logits, _, err := m.Forward(0.0, input, target)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		loss, grad := WeightedCrossEntropy(logits[0].ToCPU(), []int{v, h, e}, weights, 0, 0)
		if err := m.Backward(grad); err != nil {
			t.Fatalf("backward: %v", err)
		}
		opt.ClipGradients()
		opt.Step()
		m.ClearState()
		if epoch%25 == 0 || epoch == 299 {
			log.Printf("stable-router epoch %d loss %.4f", epoch, loss)
		}
		if epoch == 299 && loss > 0.3 {
			t.Errorf("stable router: loss did not converge: %.4f", loss)
		}
	}
}

// TestOverfitPlainOutput bypasses d.OutputMoE (k=1 routed output head) and
// routes logits through the plain OutputLayer, with a stable router LR.
// Convergence here vs the floor above would implicate the output-side MoE.
func TestOverfitPlainOutput(t *testing.T) {
	vocab := mainvocab.NewVocabulary()
	for _, tok := range []string{"hello", "there", "<EOS>"} {
		vocab.AddToken(tok)
	}
	v := vocab.GetTokenID("hello")
	h := vocab.GetTokenID("there")
	e := vocab.EosID
	if e < 0 {
		e = vocab.GetTokenID("<EOS>")
	}

	m, err := moe.NewHybridIntentMoE(vocab.Size(), 64, 4, vocab.Size(), vocab.Size(), vocab.Size(), 2)
	if err != nil {
		t.Fatalf("model: %v", err)
	}
	m.Decoder, _ = moe.NewRNNDecoder(64, vocab.Size(), 64, 2, 1, 0.0, 4)
	m.RepairArchitecture()
	m.RebuildActiveLayers()
	m.SentenceVocab = vocab
	m.SentenceVocabSize = vocab.Size()
	m.Decoder.ResizeOutputLayer(vocab.Size())
	m.ResizeEmbeddings(vocab.Size())
	m.Decoder.OutputMoE = nil // ← bypass the routed output head
	m.Decoder.OutputLayer, _ = neuralnn.NewLinear(128, vocab.Size())

	bos := vocab.BosID
	if bos < 0 {
		bos = vocab.GetTokenID("<BOS>")
	}
	input := tensor.NewTensor([]int{1, 3}, []float32{float32(bos), float32(v), float32(h)}, false)
	target := tensor.NewTensor([]int{1, 3}, []float32{float32(v), float32(h), float32(e)}, false)

	opt := neuralnn.NewOptimizer(m.Parameters(), 0.005, 1.0)
	if s, ok := opt.(interface{ SetRouterLR(lr float32) }); ok {
		s.SetRouterLR(0.005)
	}
	weights := make([]float32, vocab.Size())
	for i := range weights {
		weights[i] = 1.0
	}
	for epoch := 0; epoch < 300; epoch++ {
		opt.ZeroGrad()
		logits, _, err := m.Forward(0.0, input, target)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		loss, grad := WeightedCrossEntropy(logits[0].ToCPU(), []int{v, h, e}, weights, 0, 0)
		if err := m.Backward(grad); err != nil {
			t.Fatalf("backward: %v", err)
		}
		opt.ClipGradients()
		opt.Step()
		m.ClearState()
		if epoch%50 == 0 || epoch == 299 {
			log.Printf("plain-output epoch %d loss %.4f", epoch, loss)
		}
		if epoch == 299 && loss > 0.05 {
			t.Errorf("plain output: loss did not converge: %.4f", loss)
		}
	}
}
