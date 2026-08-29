package tools

import (
	"math"
	"math/rand"
	"testing"
)

// ─── Minimal Dense MLP Math (first principles) ───────────────────────────────

// layer is a plain linear layer.
type layer struct {
	w, b []float32
	in   int
	out  int
}

func newLayer(in, out int, rng *rand.Rand) *layer {
	std := float32(math.Sqrt(2.0 / float64(in)))
	w := make([]float32, in*out)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * std
	}
	return &layer{w: w, b: make([]float32, out), in: in, out: out}
}

func (l *layer) forward(x []float32) []float32 {
	z := make([]float32, l.out)
	for j := 0; j < l.out; j++ {
		var s float32
		base := j * l.in
		for i := 0; i < l.in; i++ {
			s += l.w[base+i] * x[i]
		}
		z[j] = s + l.b[j]
	}
	return z
}

func relu(x []float32) []float32 {
	out := make([]float32, len(x))
	for i := range x {
		if x[i] > 0 {
			out[i] = x[i]
		}
	}
	return out
}

func softmax(z []float32) []float32 {
	max := float32(math.Inf(-1))
	for _, v := range z {
		if v > max {
			max = v
		}
	}
	var sum float32
	out := make([]float32, len(z))
	for i := range z {
		out[i] = float32(math.Exp(float64(z[i] - max)))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// negLogProb computes -log(p[target]) to compare class likelihoods.
func negLogProb(p []float32, target int) float32 {
	return -float32(math.Log(float64(p[target])))
}

// sigmoid is used by the XOR check.
func sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(-float64(x)))) }

// ─── Sanity Checks ───────────────────────────────────────────────────────────

// TestSanityLinearForward verifies linear layer output shape and that a manual
// dot-product matches the batched computation.
func TestSanityLinearForward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	l := newLayer(3, 2, rng)
	x := []float32{1, 2, 3}
	z := l.forward(x)
	if len(z) != 2 {
		t.Fatalf("output dim = %d, want 2", len(z))
	}
	expect0 := l.w[0]*1 + l.w[1]*2 + l.w[2]*3 + l.b[0]
	if math.Abs(float64(z[0]-expect0)) > 1e-5 {
		t.Fatalf("manual dot product mismatch: got %v, want %v", z[0], expect0)
	}
}

// TestSanityReLU verifies ReLU clamps negatives and keeps positives.
func TestSanityReLU(t *testing.T) {
	x := []float32{-2, 0, 3}
	r := relu(x)
	if r[0] != 0 || r[1] != 0 {
		t.Fatalf("negatives not clamped to 0: got %v", r)
	}
	if r[2] != 3 {
		t.Fatalf("positive not kept: got %v", r)
	}
}

// TestSanitySoftmax verifies numerical stability on large logits and that
// equal logits yield a uniform distribution that sums to 1.
func TestSanitySoftmax(t *testing.T) {
	z := []float32{1000, 1000, 1000} // numerically stable
	p := softmax(z)
	var sum float32
	for _, v := range p {
		sum += v
	}
	if math.IsNaN(float64(sum)) || math.IsInf(float64(sum), 0) {
		t.Fatalf("softmax not finite: sum=%v", sum)
	}
	if math.Abs(float64(sum-1)) > 1e-5 {
		t.Fatalf("softmax does not sum to 1: sum=%v", sum)
	}
	for i := range p {
		if math.Abs(float64(p[i]-1.0/3.0)) > 1e-5 {
			t.Fatalf("equal logits not uniform: p=%v", p)
		}
	}
}

// TestSanityLearnAND verifies a 2-layer ReLU MLP can learn the AND function.
func TestSanityLearnAND(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	inputs := [][]float32{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	labels := []int{0, 0, 0, 1}

	h1 := newLayer(2, 8, rng)
	h2 := newLayer(8, 2, rng)

	lr := float32(0.05)
	correct := 0
	for step := 0; step < 500; step++ {
		for i := range inputs {
			a1 := relu(h1.forward(inputs[i]))
			z2 := h2.forward(a1)
			p := softmax(z2)

			// derivative of -log p[target] w.r.t. z2 (softmax cross-entropy)
			dz2 := make([]float32, len(p))
			for j := range p {
				dz2[j] = p[j]
				if j == labels[i] {
					dz2[j] -= 1
				}
			}

			// h2 grads
			for j := 0; j < h2.out; j++ {
				for k := 0; k < h2.in; k++ {
					h2.w[j*h2.in+k] -= lr * dz2[j] * a1[k]
				}
				h2.b[j] -= lr * dz2[j]
			}

			// backprop to a1 (ReLU derivative)
			da1 := make([]float32, len(a1))
			for k := range a1 {
				for j := 0; j < h2.out; j++ {
					da1[k] += dz2[j] * h2.w[j*h2.in+k]
				}
				if a1[k] <= 0 {
					da1[k] = 0
				}
			}

			// h1 grads
			for j := 0; j < h1.out; j++ {
				for k := 0; k < h1.in; k++ {
					h1.w[j*h1.in+k] -= lr * da1[j] * inputs[i][k]
				}
				h1.b[j] -= lr * da1[j]
			}
		}

		// evaluate every 100 steps
		if step%100 == 99 {
			correct = 0
			for i := range inputs {
				a1 := relu(h1.forward(inputs[i]))
				p := softmax(h2.forward(a1))
				pred := 0
				if p[1] > p[0] {
					pred = 1
				}
				if pred == labels[i] {
					correct++
				}
			}
			if correct == 4 {
				break
			}
		}
	}
	if correct != 4 {
		t.Fatalf("MLP failed to learn AND: correct=%d", correct)
	}
}

// TestSanityCrossEntropy verifies the correct class has lower loss.
func TestSanityCrossEntropy(t *testing.T) {
	good := softmax([]float32{0.1, 2.0})
	bad := softmax([]float32{2.0, 0.1})
	lGood := negLogProb(good, 1)
	lBad := negLogProb(bad, 1)
	if !(lGood < lBad) {
		t.Fatalf("correct class should have lower loss: lGood=%.4f lBad=%.4f", lGood, lBad)
	}
}

// TestSanityLearnXOR verifies a 2-4-1 sigmoid MLP can learn XOR, confirming
// the dense stack has enough non-linear capacity to fit non-linearly separable
// functions (the classic first-principles backprop test).
func TestSanityLearnXOR(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	inputs := [][]float32{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	labels := []int{0, 1, 1, 0}

	h1 := newLayer(2, 4, rng)
	h2 := newLayer(4, 1, rng)

	correct := 0
	lr := float32(0.5)
	for step := 0; step < 2000; step++ {
		for i := range inputs {
			// forward
			a1 := make([]float32, 4)
			z1 := h1.forward(inputs[i])
			for k := range z1 {
				a1[k] = sigmoid(z1[k])
			}
			z2 := h2.forward(a1)
			p := sigmoid(z2[0])

			// binary cross-entropy gradient: dL/dz2 = (p - target) for sigmoid+BCE
			target := float32(labels[i])
			dz2 := p - target

			// h2 grads
			for j := 0; j < h2.out; j++ {
				for k := 0; k < h2.in; k++ {
					h2.w[j*h2.in+k] -= lr * dz2 * a1[k]
				}
				h2.b[j] -= lr * dz2
			}

			// backprop to a1
			da1 := make([]float32, 4)
			for k := range da1 {
				da1[k] = dz2 * h2.w[0*h2.in+k]
				// sigmoid derivative: a1[k]*(1-a1[k])
				da1[k] *= a1[k] * (1 - a1[k])
			}

			// h1 grads
			for j := 0; j < h1.out; j++ {
				for k := 0; k < h1.in; k++ {
					h1.w[j*h1.in+k] -= lr * da1[j] * inputs[i][k]
				}
				h1.b[j] -= lr * da1[j]
			}
		}
		if step%200 == 199 {
			correct = 0
			for i := range inputs {
				z1 := h1.forward(inputs[i])
				var a1 [4]float32
				for k := range z1 {
					a1[k] = sigmoid(z1[k])
				}
				z2 := h2.forward(a1[:])
				p := sigmoid(z2[0])
				pred := 0
				if p > 0.5 {
					pred = 1
				}
				if pred == labels[i] {
					correct++
				}
			}
			if correct == 4 {
				break
			}
		}
	}
	if correct != 4 {
		t.Fatalf("MLP failed to learn XOR: correct=%d", correct)
	}
}
