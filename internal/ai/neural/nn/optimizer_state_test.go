package nn

import (
	"math"
	"testing"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// setGrads installs deterministic gradients derived from the step
// index so consecutive steps see different gradients, like training.
func setGrads(params []*Tensor, step int) {
	for pi, p := range params {
		g := make([]float32, len(p.Data))
		for i := range g {
			g[i] = float32(0.05*(float64(pi)+1) + 0.01*float64(i) + 0.02*float64(step))
			if (i+step)%3 == 0 {
				g[i] = -g[i]
			}
		}
		p.Grad = NewTensor(p.Shape, g, false)
	}
}

func cloneParams(params []*Tensor) []*Tensor {
	out := make([]*Tensor, len(params))
	for i, p := range params {
		data := make([]float32, len(p.Data))
		copy(data, p.Data)
		out[i] = NewTensor(p.Shape, data, true)
	}
	return out
}

// The rerun fix in one property: an Adam whose state is snapshotted
// and restored into a fresh optimizer over the same weights takes
// the IDENTICAL next update. This is what v1's resumes lacked — they
// restarted Adam from zero against mid-flight weights.
func TestAdamStateRoundTripIdenticalNextUpdate(t *testing.T) {
	mk := func() []*Tensor {
		return []*Tensor{
			NewTensor([]int{2}, []float32{0.5, -0.25}, true),
			NewTensor([]int{3}, []float32{0.1, 0.2, -0.3}, true),
		}
	}
	params := mk()
	adam, ok := NewOptimizer(params, 0.01, 1.0).(*Adam)
	if !ok {
		t.Fatal("optimizer is not *Adam")
	}
	for s := 0; s < 3; s++ {
		setGrads(params, s)
		adam.Step()
	}
	snap := adam.SnapshotState()
	if snap.T != 3 {
		t.Fatalf("snapshot step = %d, want 3", snap.T)
	}
	atSnap := cloneParams(params)

	// What the original optimizer does next.
	setGrads(params, 3)
	adam.Step()
	want := cloneParams(params)

	// A fresh optimizer over the snapshot weights + restored state
	// must reproduce that update exactly.
	resumed := cloneParams(atSnap)
	adam2, ok := NewOptimizer(resumed, 0.01, 1.0).(*Adam)
	if !ok {
		t.Fatal("optimizer is not *Adam")
	}
	if err := adam2.RestoreState(snap); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	setGrads(resumed, 3)
	adam2.Step()
	for i := range want {
		for j := range want[i].Data {
			if math.Abs(float64(resumed[i].Data[j]-want[i].Data[j])) > 1e-6 {
				t.Fatalf("param %d[%d]: resumed %v, uninterrupted %v — restored Adam did not reproduce the next update",
					i, j, resumed[i].Data[j], want[i].Data[j])
			}
		}
	}

	// And the control: a fresh Adam (v1 resume behavior) must NOT
	// reproduce it — its bias-corrected first step is a different,
	// larger update.
	fresh := cloneParams(atSnap)
	adam3, _ := NewOptimizer(fresh, 0.01, 1.0).(*Adam)
	setGrads(fresh, 3)
	adam3.Step()
	diff := 0.0
	for i := range want {
		for j := range want[i].Data {
			diff += math.Abs(float64(fresh[i].Data[j] - want[i].Data[j]))
		}
	}
	if diff < 1e-4 {
		t.Fatalf("fresh-Adam control barely differs from the uninterrupted update (diff %v); the round-trip property is vacuous", diff)
	}
}

// RestoreState must refuse a state that does not match the
// optimizer's parameters, leaving the optimizer untouched.
func TestAdamRestoreStateRefusesMismatch(t *testing.T) {
	params := []*Tensor{NewTensor([]int{2}, []float32{0.5, -0.25}, true)}
	adam, _ := NewOptimizer(params, 0.01, 1.0).(*Adam)
	setGrads(params, 0)
	adam.Step()

	bad := &OptimizerState{T: 9, MData: [][]float32{{0, 0}, {0, 0}}, VData: [][]float32{{0, 0}, {0, 0}}}
	if err := adam.RestoreState(bad); err == nil {
		t.Fatal("RestoreState accepted a 2-param state for a 1-param optimizer")
	}
	badShape := &OptimizerState{T: 9, MData: [][]float32{{0, 0, 0}}, VData: [][]float32{{0, 0, 0}}}
	if err := adam.RestoreState(badShape); err == nil {
		t.Fatal("RestoreState accepted a shape-mismatched state")
	}
	if err := adam.RestoreState(nil); err == nil {
		t.Fatal("RestoreState accepted nil")
	}
	if got := adam.SnapshotState().T; got != 1 {
		t.Fatalf("optimizer step = %d after refused restores, want 1 (untouched)", got)
	}
}
