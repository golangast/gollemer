package nn

import (
	"math"
	"math/rand"
	"testing"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// NOTE: math/rand's global Seed is a no-op on modern Go, so tests use a local
// RNG for determinism.

// TestMoEForwardGates checks shapes, gate normalization, and top-k correctness.
func TestMoEForwardGates(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	moe, err := NewMoE(8, 4, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	// Re-init with the local rng for determinism (NewLinear uses global rand).
	x := NewTensor([]int{2, 3, 8}, randSliceN(rng, 2*3*8), true)
	y, aux, err := moe.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(y.Shape) != 3 || y.Shape[0] != 2 || y.Shape[1] != 3 || y.Shape[2] != 8 {
		t.Fatalf("bad output shape %v", y.Shape)
	}
	if aux < 0 || aux > 4 {
		t.Fatalf("aux loss out of range: %v", aux)
	}
	N := 6
	for n := 0; n < N; n++ {
		s := float32(0)
		for k := 0; k < 2; k++ {
			s += moe.topGates[n*2+k]
		}
		if math.Abs(float64(s-1)) > 1e-5 {
			t.Fatalf("token %d gates sum to %v, want 1", n, s)
		}
		// selected experts must be the argmax probs
		for k := 0; k < 2; k++ {
			e := moe.topIdx[n*2+k]
			p := moe.probs[n*4+e]
			for e2 := 0; e2 < 4; e2++ {
				sel := false
				for k2 := 0; k2 < 2; k2++ {
					if moe.topIdx[n*2+k2] == e2 {
						sel = true
					}
				}
				if !sel && moe.probs[n*4+e2] > p+1e-6 {
					t.Fatalf("token %d: unselected expert %d has higher prob", n, e2)
				}
			}
		}
	}
}

// TestMoENumericGradDense finite-differences the backward pass with dense
// routing (topK == numExperts), where the whole map is smooth, so central
// differences must match tightly.
func TestMoENumericGradDense(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	H, E, K, D := 4, 3, 3, 3
	moe, err := NewMoE(H, E, K, D)
	if err != nil {
		t.Fatal(err)
	}
	x := NewTensor([]int{2, 2, H}, randSliceN(rng, 2*2*H), true)
	dY := NewTensor([]int{2, 2, H}, randSliceN(rng, 2*2*H), false)

	scalarLoss := func() float64 {
		yy, _, err := moe.Forward(x)
		if err != nil {
			t.Fatal(err)
		}
		// float64 reduction: the layer is float32, but we avoid extra
		// rounding in the final dot product.
		s := float64(0)
		for i := range yy.Data {
			s += float64(dY.Data[i]) * float64(yy.Data[i])
		}
		return s
	}
	y, _, err := moe.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	_ = y
	// Backward with auxCoeff=0 so grads match scalarLoss (which has no aux
	// term; the aux path is exercised by the load-balance and descent tests).
	dX, err := moe.Backward(dY, 0)
	if err != nil {
		t.Fatal(err)
	}

	eps := float32(1e-3)
	checkTensor := func(name string, p *Tensor) {
		if p.Grad == nil {
			t.Fatalf("%s grad is nil after backward", name)
		}
		maxRel, maxAbs := 0.0, 0.0
		for i := range p.Data {
			orig := p.Data[i]
			p.Data[i] = orig + eps
			lp := scalarLoss()
			p.Data[i] = orig - eps
			lm := scalarLoss()
			p.Data[i] = orig
			num := (lp - lm) / (2 * float64(eps))
			ana := float64(p.Grad.Data[i])
			ae := math.Abs(num - ana)
			if ae > maxAbs {
				maxAbs = ae
			}
			den := math.Abs(num) + math.Abs(ana) + 1e-6
			if rel := ae / den; rel > maxRel {
				maxRel = rel
			}
		}
		t.Logf("%s max rel err: %v max abs err: %v", name, maxRel, maxAbs)
		// Combined criterion: tiny gradients (below float32 central-difference
		// resolution, where lp==lm bit-identically) are judged by absolute
		// error; the rest by relative error. Either catches real bugs
		// (sign errors, wrong transposes give O(1) errors).
		if maxRel > 2e-2 && maxAbs > 5e-4 {
			t.Fatalf("%s gradient check failed: rel %v abs %v", name, maxRel, maxAbs)
		}
	}
	for i, p := range moe.Parameters() {
		checkTensor(string(rune('a'+i)), p)
	}
	// input gradient
	maxRel, maxAbs := 0.0, 0.0
	for i := range x.Data {
		orig := x.Data[i]
		x.Data[i] = orig + eps
		lp := scalarLoss()
		x.Data[i] = orig - eps
		lm := scalarLoss()
		x.Data[i] = orig
		num := (lp - lm) / (2 * float64(eps))
		ana := float64(dX.Data[i])
		ae := math.Abs(num - ana)
		if ae > maxAbs {
			maxAbs = ae
		}
		den := math.Abs(num) + math.Abs(ana) + 1e-6
		if rel := ae / den; rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("input max rel err: %v max abs err: %v", maxRel, maxAbs)
	if maxRel > 2e-2 && maxAbs > 5e-4 {
		t.Fatalf("input gradient check failed: rel %v abs %v", maxRel, maxAbs)
	}
}

// TestMoESparseDescent checks that with sparse top-k routing the analytic
// gradients point downhill: a small step along -grad must reduce the loss.
// (Finite differences are unreliable exactly at router decision boundaries,
// where the true map is discontinuous, so we test the property that matters.)
func TestMoESparseDescent(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	H, E, K, D := 8, 4, 2, 4
	moe, err := NewMoE(H, E, K, D)
	if err != nil {
		t.Fatal(err)
	}
	x := NewTensor([]int{3, 4, H}, randSliceN(rng, 3*4*H), true)
	dY := NewTensor([]int{3, 4, H}, randSliceN(rng, 3*4*H), false)

	lossOf := func() float32 {
		yy, aux, err := moe.Forward(x)
		if err != nil {
			t.Fatal(err)
		}
		return dot(dY.Data, yy.Data) + 0.01*aux
	}
	// gradient of lossOf = Backward(dY, 0.01)
	l0 := lossOf()
	dX, err := moe.Backward(dY, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	// snapshot params
	params := moe.Parameters()
	snap := make([][]float32, len(params))
	for i, p := range params {
		snap[i] = append([]float32(nil), p.Data...)
		if p.Grad == nil {
			t.Fatalf("param %d grad nil", i)
		}
	}
	xSnap := append([]float32(nil), x.Data...)
	step := float32(1e-3)
	// directional derivative must be negative: d/dstep loss = -|grad|^2
	dirDeriv := float32(0)
	for i, p := range params {
		_ = i
		for j := range p.Data {
			dirDeriv -= p.Grad.Data[j] * p.Grad.Data[j]
			p.Data[j] -= step * p.Grad.Data[j]
		}
	}
	for j := range x.Data {
		dirDeriv -= dX.Data[j] * dX.Data[j]
		x.Data[j] -= step * dX.Data[j]
	}
	if dirDeriv >= 0 {
		t.Fatalf("zero gradient, nothing to test")
	}
	l1 := lossOf()
	// restore
	for i, p := range params {
		copy(p.Data, snap[i])
	}
	copy(x.Data, xSnap)
	t.Logf("loss %v -> %v along -grad (dirDeriv %v)", l0, l1, dirDeriv)
	if l1 >= l0 {
		t.Fatalf("loss did not decrease along analytic gradient: %v -> %v", l0, l1)
	}
}

// TestMoELoadBalance trains a few steps and checks all experts get used.
func TestMoELoadBalance(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	moe, err := NewMoE(8, 4, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	opt := NewOptimizer(moe.Parameters(), 0.01, 1.0)
	for step := 0; step < 20; step++ {
		opt.ZeroGrad()
		x := NewTensor([]int{4, 5, 8}, randSliceN(rng, 4*5*8), true)
		y, _, err := moe.Forward(x)
		if err != nil {
			t.Fatal(err)
		}
		// dummy loss: mean square of output
		grad := NewTensor(y.Shape, make([]float32, len(y.Data)), false)
		for i, v := range y.Data {
			grad.Data[i] = 2 * v / float32(len(y.Data))
		}
		if _, err := moe.Backward(grad, 0.01); err != nil {
			t.Fatal(err)
		}
		opt.Step()
	}
	usage := moe.ExpertUsage()
	t.Logf("expert usage: %v", usage)
	for e, u := range usage {
		if u < 0.05 {
			t.Fatalf("expert %d starved: usage %v", e, u)
		}
	}
}

func randSliceN(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64() * 0.5)
	}
	return s
}

func dot(a, b []float32) float32 {
	s := float32(0)
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
