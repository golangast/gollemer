package chat

import (
	"log"
	"math"
	"testing"

	"github.com/golangast/gollemer/internal/ai/moe"
	neuralnn "github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// TestGradFlowAudit checks which parameters actually receive gradients.
func TestGradFlowAudit(t *testing.T) {
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

	params := m.Parameters()
	t.Logf("total parameters registered: %d", len(params))
	opt := neuralnn.NewOptimizer(params, 0.005, 1.0)
	weights := make([]float32, vocab.Size())
	for i := range weights {
		weights[i] = 1.0
	}

	for epoch := 0; epoch < 3; epoch++ {
		opt.ZeroGrad()
		logits, _, err := m.Forward(0.0, input, target)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		targets := []int{v, h, e}
		loss, grad := WeightedCrossEntropy(logits[0].ToCPU(), targets, weights, 0, 0)
		if err := m.Backward(grad); err != nil {
			t.Fatalf("backward: %v", err)
		}
		// audit
		withGrad, without := 0, 0
		var totalNorm float64
		for i, p := range params {
			if p.Grad == nil {
				without++
				continue
			}
			withGrad++
			var n float64
			for _, g := range p.Grad.Data {
				n += float64(g) * float64(g)
			}
			n = math.Sqrt(n)
			totalNorm += n
			if epoch == 2 && n > 0 {
				log.Printf("param %d shape=%v gradNorm=%.6g", i, p.Shape, n)
			}
		}
		log.Printf("epoch %d loss %.4f | params with grad: %d / %d | summed gradNorm %.6g",
			epoch, loss, withGrad, len(params), totalNorm)
		opt.ClipGradients()
		opt.Step()
		m.ClearState()
	}
}
