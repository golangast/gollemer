package seq2seq

// Gob round-trip test for the MoE-augmented decoder: the portable .gob model
// must carry the router and all experts.

import (
	"os"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func TestMoESaveLoadRoundTrip(t *testing.T) {
	v := vocab.NewVocabulary()
	for _, w := range []string{"hello", "hi"} {
		v.AddToken(w)
	}
	m, err := NewSeq2Seq(v.Size(), v.Size(), 16, 32, nil, v)
	if err != nil {
		t.Fatal(err)
	}
	if m.Decoder.MoE == nil {
		t.Fatal("MoE is nil after NewSeq2Seq")
	}
	before := m.Decoder.MoE.Router.Weights.Data[0]
	beforeE := m.Decoder.MoE.Experts[2].Fc2.Weights.Data[0]
	f, _ := os.CreateTemp("", "moe-*.gob")
	path := f.Name()
	f.Close()
	defer os.Remove(path)
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	m2, err := Load(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if m2.Decoder.MoE == nil {
		t.Fatal("MoE is nil after Load")
	}
	if got := m2.Decoder.MoE.Router.Weights.Data[0]; got != before {
		t.Fatalf("router weight mismatch: %v vs %v", got, before)
	}
	if got := m2.Decoder.MoE.Experts[2].Fc2.Weights.Data[0]; got != beforeE {
		t.Fatalf("expert weight mismatch: %v vs %v", got, beforeE)
	}
	if len(m2.Decoder.MoE.Experts) != 4 {
		t.Fatalf("experts = %d, want 4", len(m2.Decoder.MoE.Experts))
	}
	if m2.Decoder.MoE.TopK != 2 {
		t.Fatalf("topK = %d, want 2", m2.Decoder.MoE.TopK)
	}
	if len(m.Parameters()) != len(m2.Parameters()) {
		t.Fatalf("param count %d vs %d", len(m.Parameters()), len(m2.Parameters()))
	}
}
