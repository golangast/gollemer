package moe

import (
	"math"
	"math/rand"
)

// TemporalEncoder is a lightweight single-layer GRU that reads a sequence of
// frame tokens produced by VisionEncoder and outputs a single hidden-state
// vector that captures motion (pan, tilt, object translation).
//
// Architecture (all pure Go, zero CGO):
//
//	Input  : sequence of T frame tokens, each of size InputDim  (e.g. 512)
//	Output : single hidden state of size HiddenDim (e.g. 128)
//
// GRU equations (per time-step t):
//
//	z_t = sigmoid(Wz * x_t + Uz * h_{t-1} + bz)   // update gate
//	r_t = sigmoid(Wr * x_t + Ur * h_{t-1} + br)   // reset gate
//	n_t = tanh(   Wn * x_t + Un * (r_t ⊙ h_{t-1}) + bn)  // candidate
//	h_t = (1 - z_t) ⊙ h_{t-1}  +  z_t ⊙ n_t
type TemporalEncoder struct {
	InputDim  int
	HiddenDim int

	// Update gate weights
	Wz    []float32 // [HiddenDim x InputDim]
	Uz    []float32 // [HiddenDim x HiddenDim]
	Bz    []float32 // [HiddenDim]
	GradWz []float32
	GradUz []float32
	GradBz []float32

	// Reset gate weights
	Wr    []float32
	Ur    []float32
	Br    []float32
	GradWr []float32
	GradUr []float32
	GradBr []float32

	// Candidate weights
	Wn    []float32
	Un    []float32
	Bn    []float32
	GradWn []float32
	GradUn []float32
	GradBn []float32

	// Cache for backward pass
	lastFrameTokens [][]float32
	lastHiddens     [][]float32 // h_0 … h_T
	lastZ           [][]float32
	lastR           [][]float32
	lastN           [][]float32
}

// NewTemporalEncoder initialises a TemporalEncoder with Xavier weights.
func NewTemporalEncoder(inputDim, hiddenDim int) *TemporalEncoder {
	alloc := func(n int) []float32 { return make([]float32, n) }
	xavier := func(n int, fanIn int) []float32 {
		w := make([]float32, n)
		limit := float32(math.Sqrt(1.0 / float64(fanIn)))
		for i := range w {
			w[i] = rand.Float32()*2*limit - limit
		}
		return w
	}

	hd, id := hiddenDim, inputDim
	return &TemporalEncoder{
		InputDim:  id,
		HiddenDim: hd,

		Wz: xavier(hd*id, id), Uz: xavier(hd*hd, hd), Bz: alloc(hd),
		Wr: xavier(hd*id, id), Ur: xavier(hd*hd, hd), Br: alloc(hd),
		Wn: xavier(hd*id, id), Un: xavier(hd*hd, hd), Bn: alloc(hd),

		GradWz: alloc(hd * id), GradUz: alloc(hd * hd), GradBz: alloc(hd),
		GradWr: alloc(hd * id), GradUr: alloc(hd * hd), GradBr: alloc(hd),
		GradWn: alloc(hd * id), GradUn: alloc(hd * hd), GradBn: alloc(hd),
	}
}

// sigmoid and tanh helpers
func sigm(x float32) float32 { return float32(1.0 / (1.0 + math.Exp(float64(-x)))) }
func tanhF(x float32) float32 { return float32(math.Tanh(float64(x))) }

// mvMul computes y[i] = sum_j W[i*cols+j] * x[j]  (row-major matrix × vector)
func mvMul(W []float32, x []float32, rows, cols int) []float32 {
	y := make([]float32, rows)
	for i := 0; i < rows; i++ {
		var s float32
		for j := 0; j < cols; j++ {
			s += W[i*cols+j] * x[j]
		}
		y[i] = s
	}
	return y
}

// vadd returns a+b element-wise (allocating result)
func vadd(a, b []float32) []float32 {
	c := make([]float32, len(a))
	for i := range a {
		c[i] = a[i] + b[i]
	}
	return c
}

// Forward runs the GRU over a sequence of frame tokens and returns the final
// hidden state, which encodes the motion across all frames.
// frameTokens: [][]float32 of length T, each of size InputDim.
func (te *TemporalEncoder) Forward(frameTokens [][]float32) []float32 {
	T := len(frameTokens)
	hd := te.HiddenDim
	id := te.InputDim

	te.lastFrameTokens = frameTokens
	te.lastHiddens = make([][]float32, T+1)
	te.lastZ = make([][]float32, T)
	te.lastR = make([][]float32, T)
	te.lastN = make([][]float32, T)

	// h_0 = zeros
	te.lastHiddens[0] = make([]float32, hd)

	for t := 0; t < T; t++ {
		x := frameTokens[t]
		h := te.lastHiddens[t]

		// Update gate: z = sigmoid(Wz*x + Uz*h + bz)
		zRaw := vadd(vadd(mvMul(te.Wz, x, hd, id), mvMul(te.Uz, h, hd, hd)), te.Bz)
		z := make([]float32, hd)
		for i := range zRaw { z[i] = sigm(zRaw[i]) }

		// Reset gate: r = sigmoid(Wr*x + Ur*h + br)
		rRaw := vadd(vadd(mvMul(te.Wr, x, hd, id), mvMul(te.Ur, h, hd, hd)), te.Br)
		r := make([]float32, hd)
		for i := range rRaw { r[i] = sigm(rRaw[i]) }

		// r ⊙ h
		rh := make([]float32, hd)
		for i := range rh { rh[i] = r[i] * h[i] }

		// Candidate: n = tanh(Wn*x + Un*(r⊙h) + bn)
		nRaw := vadd(vadd(mvMul(te.Wn, x, hd, id), mvMul(te.Un, rh, hd, hd)), te.Bn)
		n := make([]float32, hd)
		for i := range nRaw { n[i] = tanhF(nRaw[i]) }

		// New hidden: h_new = (1-z)⊙h + z⊙n
		hNew := make([]float32, hd)
		for i := range hNew { hNew[i] = (1-z[i])*h[i] + z[i]*n[i] }

		te.lastZ[t] = z
		te.lastR[t] = r
		te.lastN[t] = n
		te.lastHiddens[t+1] = hNew
	}

	return te.lastHiddens[T]
}

// Backward performs BPTT through the GRU and updates weights via SGD.
// dh is the gradient of the loss w.r.t. the final hidden state (size HiddenDim).
// Returns the gradient with respect to the input sequence (size T x InputDim).
func (te *TemporalEncoder) Backward(dh []float32, lr float32) [][]float32 {
	T := len(te.lastFrameTokens)
	hd := te.HiddenDim
	id := te.InputDim

	// Reset accumulated gradients
	for i := range te.GradWz { te.GradWz[i] = 0 }
	for i := range te.GradUz { te.GradUz[i] = 0 }
	for i := range te.GradBz { te.GradBz[i] = 0 }
	for i := range te.GradWr { te.GradWr[i] = 0 }
	for i := range te.GradUr { te.GradUr[i] = 0 }
	for i := range te.GradBr { te.GradBr[i] = 0 }
	for i := range te.GradWn { te.GradWn[i] = 0 }
	for i := range te.GradUn { te.GradUn[i] = 0 }
	for i := range te.GradBn { te.GradBn[i] = 0 }

	dhNext := make([]float32, hd)
	copy(dhNext, dh)
	
	dxTokens := make([][]float32, T)

	for t := T - 1; t >= 0; t-- {
		x := te.lastFrameTokens[t]
		h := te.lastHiddens[t]
		z := te.lastZ[t]
		r := te.lastR[t]
		n := te.lastN[t]

		// dh_t (combines gradient from loss and from t+1)
		dht := dhNext

		// d_n: grad through z⊙n part of h
		dn := make([]float32, hd)
		for i := range dn { dn[i] = dht[i] * z[i] * (1 - n[i]*n[i]) } // tanh'

		// d_z
		dz := make([]float32, hd)
		for i := range dz { dz[i] = dht[i] * (n[i] - h[i]) * z[i] * (1 - z[i]) } // sigmoid'

		// d_r (flows from Wn path)
		dr := make([]float32, hd)
		// ∂L/∂r_i = sum_j Un[j,i] * dn_j * h_i  (simplified: Un^T * (dn ⊙ h))
		for j := 0; j < hd; j++ {
			for i := 0; i < hd; i++ {
				dr[i] += te.Un[j*hd+i] * dn[j] * h[i]
			}
		}
		for i := range dr { dr[i] *= r[i] * (1 - r[i]) }

		// dx (gradient w.r.t input frame x)
		dx := make([]float32, id)
		for j := 0; j < id; j++ {
			for i := 0; i < hd; i++ {
				dx[j] += te.Wz[i*id+j] * dz[i]
				dx[j] += te.Wr[i*id+j] * dr[i]
				dx[j] += te.Wn[i*id+j] * dn[i]
			}
		}
		dxTokens[t] = dx

		// Accumulate weight gradients
		// Wz, Wr, Wn: [hd x id]
		for i := 0; i < hd; i++ {
			for j := 0; j < id; j++ {
				te.GradWz[i*id+j] += dz[i] * x[j]
				te.GradWr[i*id+j] += dr[i] * x[j]
				te.GradWn[i*id+j] += dn[i] * x[j]
			}
		}
		// Uz, Ur, Un: [hd x hd]
		for i := 0; i < hd; i++ {
			for j := 0; j < hd; j++ {
				te.GradUz[i*hd+j] += dz[i] * h[j]
				te.GradUr[i*hd+j] += dr[i] * h[j]
				te.GradUn[i*hd+j] += dn[i] * r[j] * h[j]
			}
		}
		// Biases
		for i := 0; i < hd; i++ {
			te.GradBz[i] += dz[i]
			te.GradBr[i] += dr[i]
			te.GradBn[i] += dn[i]
		}

		// Propagate gradient to h_{t-1}
		dhPrev := make([]float32, hd)
		for i := range dhPrev {
			dhPrev[i] = dht[i] * (1 - z[i]) // direct path
		}
		// Through Uz (update gate)
		for j := 0; j < hd; j++ {
			for i := 0; i < hd; i++ {
				dhPrev[j] += te.Uz[i*hd+j] * dz[i]
				dhPrev[j] += te.Ur[i*hd+j] * dr[i]
				dhPrev[j] += te.Un[i*hd+j] * dn[i] * r[i]
			}
		}
		dhNext = dhPrev
	}

	// Apply SGD
	for i := range te.Wz { te.Wz[i] -= lr * te.GradWz[i] }
	for i := range te.Uz { te.Uz[i] -= lr * te.GradUz[i] }
	for i := range te.Bz { te.Bz[i] -= lr * te.GradBz[i] }
	for i := range te.Wr { te.Wr[i] -= lr * te.GradWr[i] }
	for i := range te.Ur { te.Ur[i] -= lr * te.GradUr[i] }
	for i := range te.Br { te.Br[i] -= lr * te.GradBr[i] }
	for i := range te.Wn { te.Wn[i] -= lr * te.GradWn[i] }
	for i := range te.Un { te.Un[i] -= lr * te.GradUn[i] }
	for i := range te.Bn { te.Bn[i] -= lr * te.GradBn[i] }
	
	return dxTokens
}
