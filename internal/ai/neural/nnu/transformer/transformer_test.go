package transformer

import (
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func TestTransformerForward(t *testing.T) {
	cfg := TransformerConfig{
		DModel:    64,
		NumHeads:  4,
		NumLayers: 2,
		DFF:       128,
		MaxLen:    32,
		VocabSize: 100,
	}
	model, err := NewTransformer(cfg)
	if err != nil {
		t.Fatalf("NewTransformer: %v", err)
	}

	batch, srcSeq, tgtSeq := 2, 10, 8
	srcData := make([]float32, batch*srcSeq)
	tgtData := make([]float32, batch*tgtSeq)
	for i := range srcData {
		srcData[i] = float32(i % cfg.VocabSize)
	}
	for i := range tgtData {
		tgtData[i] = float32(i % cfg.VocabSize)
	}
	src := tensor.NewTensor([]int{batch, srcSeq}, srcData, false)
	tgt := tensor.NewTensor([]int{batch, tgtSeq}, tgtData, false)

	logits, err := model.Forward(src, tgt)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expected := []int{batch, tgtSeq, cfg.VocabSize}
	for i, d := range expected {
		if logits.Shape[i] != d {
			t.Errorf("shape[%d] = %d, want %d", i, logits.Shape[i], d)
		}
	}

	t.Logf("Params: ~%d", model.NumParams())
}

func TestCausalMask(t *testing.T) {
	mask := CausalMask(4)
	// Position 0 should only see itself
	if mask.Data[0*4+1] != -1e9 {
		t.Error("position 0 should not see position 1")
	}
	if mask.Data[0*4+0] != 0 {
		t.Error("position 0 should see itself")
	}
	// Position 3 should see all
	for j := 0; j < 4; j++ {
		if mask.Data[3*4+j] != 0 {
			t.Errorf("position 3 should see position %d", j)
		}
	}
}
