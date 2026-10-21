package transformer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// TestInitLossVsSeqLen is a diagnostic/regression probe for the hot-init
// failure found in the mid-transformer Phase A smoke (2026-10-21): at
// Config M depth, initial cross-entropy on LONG source sequences was
// ~110-145 (vs ln(vocab) ~= 8 for short ones), because the pre-norm
// residual stream reaches the output projection unnormalized. A healthy
// init has mean CE near ln(vocab) at every sequence length.
func TestInitLossVsSeqLen(t *testing.T) {
	rand.Seed(42)
	cfg := TransformerConfig{
		VocabSize: 1024,
		DModel:    256,
		NumHeads:  8,
		NumLayers: 6,
		DFF:       1024,
		MaxLen:    512,
	}
	m, err := NewTransformer(cfg)
	if err != nil {
		t.Fatalf("NewTransformer: %v", err)
	}
	meanCE := func(srcLen, tgtLen int) float64 {
		srcData := make([]float32, srcLen)
		for i := range srcData {
			srcData[i] = float32(rand.Intn(cfg.VocabSize))
		}
		tgtData := make([]float32, tgtLen)
		targets := make([]int, tgtLen)
		for i := range tgtData {
			targets[i] = rand.Intn(cfg.VocabSize)
			tgtData[i] = float32(targets[i])
		}
		src := tensor.NewTensor([]int{1, srcLen}, srcData, false)
		tgt := tensor.NewTensor([]int{1, tgtLen}, tgtData, false)
		logits, err := m.Forward(src, tgt)
		if err != nil {
			t.Fatalf("Forward(srcLen=%d): %v", srcLen, err)
		}
		var sumSq, sum float64
		n := len(logits.Data)
		for _, v := range logits.Data {
			sum += float64(v)
		}
		mean := sum / float64(n)
		for _, v := range logits.Data {
			d := float64(v) - mean
			sumSq += d * d
		}
		std := math.Sqrt(sumSq / float64(n))
		var ce float64
		for pos := 0; pos < tgtLen; pos++ {
			row := logits.Data[pos*cfg.VocabSize : (pos+1)*cfg.VocabSize]
			max := float64(row[0])
			for _, v := range row[1:] {
				if float64(v) > max {
					max = float64(v)
				}
			}
			var denom float64
			for _, v := range row {
				denom += math.Exp(float64(v) - max)
			}
			ce += (max + math.Log(denom)) - float64(row[targets[pos]])
		}
		ce /= float64(tgtLen)
		t.Logf("srcLen=%d meanCE=%.3f logitStd=%.2f (uniform CE=%.3f)", srcLen, ce, std, math.Log(float64(cfg.VocabSize)))
		return ce
	}
	short := meanCE(24, 10)
	long := meanCE(384, 10)
	uniform := math.Log(float64(cfg.VocabSize))
	if short > uniform*2.5 {
		t.Errorf("short-sequence init CE %.2f is >2.5x uniform %.2f", short, uniform)
	}
	if long > uniform*2.5 {
		t.Errorf("long-sequence init CE %.2f is >2.5x uniform %.2f: init scale / final-norm problem", long, uniform)
	}
}
