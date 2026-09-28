package seq2seq

import (
	"math"
	"math/rand"
	"testing"
)

// TestCopyGateInitNearlyClosed verifies the gate starts small so a fresh
// model behaves like the no-copy baseline until copying proves useful.
func TestCopyGateInitNearlyClosed(t *testing.T) {
	g := NewCopyGate(8)
	q := []float32{0.5, -0.3, 0.1, 0.2, -0.4, 0.6, -0.1, 0.3}
	got := g.GateValue(q)
	want := float32(1.0 / (1.0 + math.Exp(2.0))) // sigmoid(-2)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("initial gate = %v, want sigmoid(-2) = %v", got, want)
	}
	if len(g.Parameters()) != 2 {
		t.Fatalf("Parameters() = %d tensors, want 2", len(g.Parameters()))
	}
	var nilGate *CopyGate
	if nilGate.Parameters() != nil {
		t.Fatalf("nil gate Parameters() should be nil")
	}
}

// TestCopyPrefersHistoryToken checks the pointer behavior deterministically:
// with a single history position the attention is exactly 1.0 there, so the
// history token keeps its logit while every other token is penalized.
func TestCopyPrefersHistoryToken(t *testing.T) {
	g := NewCopyGate(4)
	vocab := 6
	gen := []float32{0, 0, 0, 0, 0, 0}
	query := []float32{1, 0, 0, 0}
	hist := [][]float32{{0.2, 0.1, -0.3, 0.4}}
	c := g.forwardCopyStep(query, gen, hist, []int{3})
	if c.nh != 1 || math.Abs(float64(c.w[0]-1.0)) > 1e-5 {
		t.Fatalf("single-position attention weight = %v, want 1.0", c.w)
	}
	// History token 3: bias = gate * log(1+eps) ~= 0.
	if math.Abs(float64(gen[3])) > 1e-4 {
		t.Fatalf("history token logit changed by %v, want ~0", gen[3])
	}
	// Other tokens: bias = gate * log(eps) < 0 (suppressed).
	for v := 0; v < vocab; v++ {
		if v == 3 {
			continue
		}
		if gen[v] >= -0.5 {
			t.Fatalf("non-history token %d not suppressed: logit %v", v, gen[v])
		}
	}
	// Empty history: logits untouched.
	gen2 := []float32{1, 2, 3}
	g.forwardCopyStep(query, gen2, nil, nil)
	for i, x := range []float32{1, 2, 3} {
		if gen2[i] != x {
			t.Fatalf("empty-history forward modified logits")
		}
	}
}

// TestCopyBackwardNumericGradient is a finite-difference check of
// backwardCopyStep against forwardCopyStep. The previous copy attempt
// diverged (loss 8-9); a wrong manual gradient is the prime suspect, so
// this test must stay green.
func TestCopyBackwardNumericGradient(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	H, V, nh := 5, 7, 4

	g := NewCopyGate(H)
	for i := range g.Wg.Data {
		g.Wg.Data[i] = float32(rng.NormFloat64()) * 0.3
	}
	g.Bg.Data[0] = float32(rng.NormFloat64()) * 0.5

	query := make([]float32, H)
	for i := range query {
		query[i] = float32(rng.NormFloat64())
	}
	gen := make([]float32, V)
	for i := range gen {
		gen[i] = float32(rng.NormFloat64())
	}
	hist := make([][]float32, nh)
	for i := range hist {
		hist[i] = make([]float32, H)
		for h := range hist[i] {
			hist[i][h] = float32(rng.NormFloat64())
		}
	}
	toks := []int{2, 5, 2, 0} // token 2 appears twice: tests the scatter

	// Scalar loss: sum of final logits (exercises every output).
	lossFn := func(q, gg []float32, hs [][]float32) float32 {
		cp := append([]float32(nil), gg...)
		g.forwardCopyStep(q, cp, hs, toks)
		var s float32
		for _, x := range cp {
			s += x
		}
		return s
	}
	dLogits := make([]float32, V)
	for i := range dLogits {
		dLogits[i] = 1 // d(sum)/d(final) = 1
	}

	genCp := append([]float32(nil), gen...)
	c := g.forwardCopyStep(query, genCp, hist, toks)
	dQuery, dHist := g.backwardCopyStep(dLogits, c, hist, toks)

	const eps = 1e-3
	// Central-difference every input against the analytic gradients.
	for h := 0; h < H; h++ {
		orig := g.Wg.Data[h]
		g.Wg.Data[h] = orig + eps
		lp := lossFn(query, gen, hist)
		g.Wg.Data[h] = orig - eps
		lm := lossFn(query, gen, hist)
		g.Wg.Data[h] = orig
		num := (lp - lm) / (2 * eps)
		an := g.Wg.Grad.Data[h]
		if math.Abs(float64(an-num)) > 3e-2 {
			t.Errorf("dWg[%d]: analytic=%v numeric=%v", h, an, num)
		}
	}
	orig := g.Bg.Data[0]
	g.Bg.Data[0] = orig + eps
	lp := lossFn(query, gen, hist)
	g.Bg.Data[0] = orig - eps
	lm := lossFn(query, gen, hist)
	g.Bg.Data[0] = orig
	if num := (lp - lm) / (2 * eps); math.Abs(float64(g.Bg.Grad.Data[0]-num)) > 3e-2 {
		t.Errorf("dBg: analytic=%v numeric=%v", g.Bg.Grad.Data[0], num)
	}
	for h := 0; h < H; h++ {
		orig := query[h]
		query[h] = orig + eps
		lp := lossFn(query, gen, hist)
		query[h] = orig - eps
		lm := lossFn(query, gen, hist)
		query[h] = orig
		if num := (lp - lm) / (2 * eps); math.Abs(float64(dQuery[h]-num)) > 3e-2 {
			t.Errorf("dQuery[%d]: analytic=%v numeric=%v", h, dQuery[h], num)
		}
	}
	for i := 0; i < nh; i++ {
		for h := 0; h < H; h++ {
			orig := hist[i][h]
			hist[i][h] = orig + eps
			lp := lossFn(query, gen, hist)
			hist[i][h] = orig - eps
			lm := lossFn(query, gen, hist)
			hist[i][h] = orig
			if num := (lp - lm) / (2 * eps); math.Abs(float64(dHist[i][h]-num)) > 3e-2 {
				t.Errorf("dHist[%d][%d]: analytic=%v numeric=%v", i, h, dHist[i][h], num)
			}
		}
	}
}
