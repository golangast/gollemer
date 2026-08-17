package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

type stubExpert struct{}

func (stubExpert) Forward(input *tensor.Tensor) (*tensor.Tensor, error)     { return input, nil }
func (stubExpert) Backward(grad *tensor.Tensor) error                       { return nil }
func (stubExpert) Parameters() []*tensor.Tensor                             { return nil }
func (stubExpert) Inputs() []*tensor.Tensor                                 { return nil }
func (stubExpert) Description() string                                      { return "stub" }
func (stubExpert) SetMode(training bool)                                    {}
func (stubExpert) ClearState()                                              {}
func (stubExpert) ClipWeights(maxVal float32)                               {}
func (stubExpert) EvolutionaryReset(winner moe.Expert, jitterScale float32) {}
func (stubExpert) Shake(intensity float32)                                  {}
func (stubExpert) IsStagnant() bool                                         { return false }
func (stubExpert) UpdateHealth(wasUsed bool)                                {}
func (stubExpert) ToGPU()                                                   {}
func (stubExpert) Resize(newOutputDim int)                                  {}
func (stubExpert) SyncParameters() error                                    { return nil }
func (stubExpert) GetID() int                                               { return 0 }
func (stubExpert) GetContext() []float32                                    { return nil }
func (stubExpert) RestoreContext(ctx []float32)                             {}

func TestApplyPhaseFreezeDoesNotFreezeAllExpertsForSmallModel(t *testing.T) {
	layer := &moe.MoELayer{
		Experts:      make([]moe.Expert, 8),
		ExpertFrozen: make([]bool, 8),
	}
	for i := range layer.Experts {
		layer.Experts[i] = stubExpert{}
	}

	applyPhaseFreeze([]*moe.MoELayer{layer}, 8, 16)

	active := 0
	for _, frozen := range layer.ExpertFrozen {
		if !frozen {
			active++
		}
	}
	if active < 4 {
		t.Fatalf("oversized freeze range should still leave the lower half active; got %d active experts", active)
	}
}

func TestApplyPhaseFreezeFallsBackToUpperHalfForSmallModel(t *testing.T) {
	layer := &moe.MoELayer{
		Experts:      make([]moe.Expert, 4),
		ExpertFrozen: make([]bool, 4),
	}
	for i := range layer.Experts {
		layer.Experts[i] = stubExpert{}
	}

	applyPhaseFreeze([]*moe.MoELayer{layer}, 8, 16)

	for i := 0; i < 2; i++ {
		if layer.ExpertFrozen[i] {
			t.Fatalf("phase 1 social experts should remain active for a 4-expert model; expert %d froze", i)
		}
	}
	for i := 2; i < 4; i++ {
		if !layer.ExpertFrozen[i] {
			t.Fatalf("phase 1 should freeze the upper-half experts for a 4-expert model; expert %d stayed active", i)
		}
	}
}

func TestApplyPhaseFreezeRespectsSixteenExpertSocialPhase(t *testing.T) {
	layer := &moe.MoELayer{
		Experts:      make([]moe.Expert, 16),
		ExpertFrozen: make([]bool, 16),
	}
	for i := range layer.Experts {
		layer.Experts[i] = stubExpert{}
	}

	applyPhaseFreeze([]*moe.MoELayer{layer}, 8, 16)

	for i := 0; i < 8; i++ {
		if layer.ExpertFrozen[i] {
			t.Fatalf("phase 1 social experts should stay active, but expert %d is frozen", i)
		}
	}
	for i := 8; i < 16; i++ {
		if !layer.ExpertFrozen[i] {
			t.Fatalf("phase 1 should freeze experts 8..15, but expert %d is active", i)
		}
	}
}

func TestSimdSoftmaxF32NormalizesLargeEqualLogits(t *testing.T) {
	data := []float32{1000, 1000, 1000}
	sum := moe.SimdSoftmaxF32(data)
	if sum <= 0 {
		t.Fatalf("softmax must return a positive normalization constant; got %v", sum)
	}
	for i, v := range data {
		if v < 0 || v > 1 {
			t.Fatalf("softmax probability for index %d out of range: %v", i, v)
		}
	}
	if absF32(data[0]-0.33333334) > 1e-5 || absF32(data[1]-0.33333334) > 1e-5 || absF32(data[2]-0.33333334) > 1e-5 {
		t.Fatalf("expected uniform logits to stay uniform after softmax, got %v", data)
	}
}

func TestAssessSentenceFormationDetectsEarlyStageOutput(t *testing.T) {
	label, _, reason := assessSentenceFormation("the the of and a is to <unk>")
	if !strings.Contains(label, "Early") && !strings.Contains(label, "Fragmented") {
		t.Fatalf("expected early-stage or fragmented label, got %q (reason=%q)", label, reason)
	}
	if !strings.Contains(reason, "repetition") && !strings.Contains(reason, "<unk>") && !strings.Contains(reason, "dominating") {
		t.Fatalf("expected repetition or <unk> diagnosis, got %q", reason)
	}
}

func TestAssessSentenceFormationDetectsEmergingSentence(t *testing.T) {
	label, _, reason := assessSentenceFormation("model can process vast amounts of data quickly.")
	if !strings.Contains(label, "Emerging") && !strings.Contains(label, "Strong") {
		t.Fatalf("expected sentence-forming label, got %q (reason=%q)", label, reason)
	}
	if !strings.Contains(reason, "sentence") && !strings.Contains(reason, "coherent") {
		t.Fatalf("expected coherent sentence reasoning, got %q", reason)
	}
}

func absF32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
