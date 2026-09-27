package nn

// Mixture-of-Experts (MoE) layer with top-k routing, in the style of
// Switch Transformer / Mixtral but tiny: a router picks the top-k experts
// per token, each expert is a small two-layer FFN, and the outputs are
// combined by renormalized gate weights.
//
// Pure Go, no dependencies. Forward/backward are manual (raw slices +
// MatMulRaw) so the layer slots into the seq2seq trainer's hand-rolled BPTT.
// Experts are evaluated densely (all experts see all tokens, non-selected
// contributions are masked to zero) — simpler and exactly equivalent in
// gradient flow to sparse evaluation at this scale.
//
// The layer also computes the standard load-balancing auxiliary loss
// (E * sum_e mean_gate_e * frac_e) whose gradient is applied inside
// Backward, preventing router collapse onto a single expert.

import (
	"fmt"
	"math"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// Expert is one feed-forward expert: Linear -> tanh -> Linear.
type Expert struct {
	Fc1 *Linear
	Fc2 *Linear
}

// NewExpert creates an expert mapping hiddenDim -> expertDim -> hiddenDim.
func NewExpert(hiddenDim, expertDim int) (*Expert, error) {
	fc1, err := NewLinear(hiddenDim, expertDim)
	if err != nil {
		return nil, fmt.Errorf("moe expert fc1: %w", err)
	}
	fc2, err := NewLinear(expertDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("moe expert fc2: %w", err)
	}
	return &Expert{Fc1: fc1, Fc2: fc2}, nil
}

// Parameters returns the expert's learnable tensors.
func (e *Expert) Parameters() []*Tensor {
	params := []*Tensor{}
	params = append(params, e.Fc1.Parameters()...)
	params = append(params, e.Fc2.Parameters()...)
	return params
}

// MoE is a mixture of Experts with a learned router.
type MoE struct {
	NumExperts int
	TopK       int
	HiddenDim  int
	ExpertDim  int

	Router  *Linear
	Experts []*Expert

	// Forward caches (unexported: never serialized).
	input     *Tensor
	probs     []float32   // [N*E] full router probabilities
	topIdx    []int       // [N*K] selected expert per token
	topGates  []float32   // [N*K] renormalized gate weights
	expertH1  [][]float32 // [E][N*D] tanh activations per expert
	expertOut [][]float32 // [E][N*H] outputs per expert
	frac      []float32   // [E] fraction of tokens routed to each expert
	batch     int
	seqLen    int
}

// NewMoE creates a mixture of numExperts experts with top-k routing.
func NewMoE(hiddenDim, numExperts, topK, expertDim int) (*MoE, error) {
	if numExperts < 1 || topK < 1 || topK > numExperts {
		return nil, fmt.Errorf("moe: invalid numExperts=%d topK=%d", numExperts, topK)
	}
	router, err := NewLinear(hiddenDim, numExperts)
	if err != nil {
		return nil, fmt.Errorf("moe router: %w", err)
	}
	experts := make([]*Expert, numExperts)
	for i := range experts {
		experts[i], err = NewExpert(hiddenDim, expertDim)
		if err != nil {
			return nil, fmt.Errorf("moe expert %d: %w", i, err)
		}
	}
	return &MoE{
		NumExperts: numExperts,
		TopK:       topK,
		HiddenDim:  hiddenDim,
		ExpertDim:  expertDim,
		Router:     router,
		Experts:    experts,
	}, nil
}

// Parameters returns all learnable tensors (router + experts).
func (m *MoE) Parameters() []*Tensor {
	params := []*Tensor{}
	params = append(params, m.Router.Parameters()...)
	for _, e := range m.Experts {
		params = append(params, e.Parameters()...)
	}
	return params
}

// ExpertUsage returns the fraction of tokens routed to each expert in the
// last forward pass (for monitoring load balance).
func (m *MoE) ExpertUsage() []float32 {
	out := make([]float32, m.NumExperts)
	copy(out, m.frac)
	return out
}

// LastRouting returns the top-k expert indices and gate weights from the most
// recent Forward call, one entry per token (copies; safe to keep).
func (m *MoE) LastRouting() (idx []int, gates []float32) {
	n := m.batch * m.seqLen
	idx = make([]int, n*m.TopK)
	gates = make([]float32, n*m.TopK)
	copy(idx, m.topIdx)
	copy(gates, m.topGates)
	return idx, gates
}

// ensureGrad allocates zero gradient tensors for all parameters on first use.
func (m *MoE) ensureGrad() {
	for _, p := range m.Parameters() {
		if p.RequiresGrad && p.Grad == nil {
			p.Grad = NewTensor(p.Shape, make([]float32, len(p.Data)), false)
		}
	}
}

// addBias adds a bias vector to every row of a [rows*cols] matrix in place.
func addBias(data []float32, rows, cols int, bias *Tensor) {
	if bias == nil {
		return
	}
	bd := bias.Data
	for r := 0; r < rows; r++ {
		base := r * cols
		for c := 0; c < cols; c++ {
			data[base+c] += bd[c]
		}
	}
}

// transposeCopy returns the transpose of a [rows*cols] row-major matrix.
func transposeCopy(data []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = data[r*cols+c]
		}
	}
	return out
}

// Forward routes each token through its top-k experts.
// x: [B,T,H] or [B,H] (treated as T=1). Returns y with x's shape and the
// load-balancing auxiliary loss.
func (m *MoE) Forward(x *Tensor) (*Tensor, float32, error) {
	var batch, seqLen, H int
	switch len(x.Shape) {
	case 2:
		batch, seqLen, H = x.Shape[0], 1, x.Shape[1]
	case 3:
		batch, seqLen, H = x.Shape[0], x.Shape[1], x.Shape[2]
	default:
		return nil, 0, fmt.Errorf("moe forward: unsupported input rank %d", len(x.Shape))
	}
	if H != m.HiddenDim {
		return nil, 0, fmt.Errorf("moe forward: hidden dim %d != %d", H, m.HiddenDim)
	}
	N := batch * seqLen
	E, K, D := m.NumExperts, m.TopK, m.ExpertDim

	// Router logits: [N,H] @ [H,E].
	logits := make([]float32, N*E)
	MatMulRaw(x.Data, m.Router.Weights.Data, logits, N, E, H)
	addBias(logits, N, E, m.Router.Biases)

	// Softmax -> probs.
	probs := make([]float32, N*E)
	for n := 0; n < N; n++ {
		mx := logits[n*E]
		for e := 1; e < E; e++ {
			if logits[n*E+e] > mx {
				mx = logits[n*E+e]
			}
		}
		s := float32(0)
		for e := 0; e < E; e++ {
			v := float32(math.Exp(float64(logits[n*E+e] - mx)))
			probs[n*E+e] = v
			s += v
		}
		inv := 1 / s
		for e := 0; e < E; e++ {
			probs[n*E+e] *= inv
		}
	}

	// Top-k selection per token (E is tiny; simple selection).
	topIdx := make([]int, N*K)
	topGates := make([]float32, N*K)
	for n := 0; n < N; n++ {
		// copy probs with indices, pick K largest
		best := make([]int, K)
		bestV := make([]float32, K)
		for k := 0; k < K; k++ {
			bestV[k] = -1
		}
		for e := 0; e < E; e++ {
			v := probs[n*E+e]
			for k := 0; k < K; k++ {
				if v > bestV[k] {
					// shift down
					for j := K - 1; j > k; j-- {
						bestV[j] = bestV[j-1]
						best[j] = best[j-1]
					}
					bestV[k] = v
					best[k] = e
					break
				}
			}
		}
		s := float32(0)
		for k := 0; k < K; k++ {
			s += bestV[k]
		}
		for k := 0; k < K; k++ {
			topIdx[n*K+k] = best[k]
			topGates[n*K+k] = bestV[k] / s
		}
	}

	// Experts, evaluated densely.
	expertH1 := make([][]float32, E)
	expertOut := make([][]float32, E)
	for e := 0; e < E; e++ {
		h1 := make([]float32, N*D)
		MatMulRaw(x.Data, m.Experts[e].Fc1.Weights.Data, h1, N, D, H)
		addBias(h1, N, D, m.Experts[e].Fc1.Biases)
		for i, v := range h1 {
			h1[i] = float32(math.Tanh(float64(v)))
		}
		out := make([]float32, N*H)
		MatMulRaw(h1, m.Experts[e].Fc2.Weights.Data, out, N, H, D)
		addBias(out, N, H, m.Experts[e].Fc2.Biases)
		expertH1[e] = h1
		expertOut[e] = out
	}

	// Combine by gates.
	yData := make([]float32, N*H)
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			e := topIdx[n*K+k]
			g := topGates[n*K+k]
			oe := expertOut[e]
			base := n * H
			for h := 0; h < H; h++ {
				yData[base+h] += g * oe[base+h]
			}
		}
	}

	// Load-balancing aux loss: E * sum_e mean_p_e * frac_e.
	frac := make([]float32, E)
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			frac[topIdx[n*K+k]] += 1.0 / float32(N)
		}
	}
	aux := float32(0)
	invN := 1.0 / float32(N)
	for e := 0; e < E; e++ {
		meanP := float32(0)
		for n := 0; n < N; n++ {
			meanP += probs[n*E+e]
		}
		aux += meanP * invN * frac[e]
	}
	aux *= float32(E)

	// Cache for backward.
	m.input = x
	m.probs = probs
	m.topIdx = topIdx
	m.topGates = topGates
	m.expertH1 = expertH1
	m.expertOut = expertOut
	m.frac = frac
	m.batch = batch
	m.seqLen = seqLen

	y := NewTensor(x.Shape, yData, true)
	return y, aux, nil
}

// Backward propagates dY through the MoE, accumulating parameter gradients.
// auxCoeff scales the load-balancing auxiliary loss gradient (0.01 typical).
// Returns dX with the input's shape.
func (m *MoE) Backward(dY *Tensor, auxCoeff float32) (*Tensor, error) {
	if m.input == nil {
		return nil, fmt.Errorf("moe backward called before forward")
	}
	if len(dY.Data) != len(m.input.Data) {
		return nil, fmt.Errorf("moe backward: dY size %d != input size %d", len(dY.Data), len(m.input.Data))
	}
	m.ensureGrad()

	N := m.batch * m.seqLen
	H, E, K, D := m.HiddenDim, m.NumExperts, m.TopK, m.ExpertDim
	xData := m.input.Data
	dYd := dY.Data

	dX := make([]float32, N*H)
	dProbs := make([]float32, N*E) // dL/d probs[n,e]
	xT := transposeCopy(xData, N, H)

	// Per-expert backward. Only tokens routed to expert e contribute.
	for e := 0; e < E; e++ {
		// dOut_e[n,h] = dY[n,h] * gate(n,e).
		dOut := make([]float32, N*H)
		for n := 0; n < N; n++ {
			for k := 0; k < K; k++ {
				if m.topIdx[n*K+k] != e {
					continue
				}
				g := m.topGates[n*K+k]
				nb := n * H
				for h := 0; h < H; h++ {
					dOut[nb+h] += g * dYd[nb+h]
				}
			}
		}
		h1 := m.expertH1[e]
		W2 := m.Experts[e].Fc2.Weights.Data

		// dW2 = h1^T @ dOut : [D,H].
		h1T := transposeCopy(h1, N, D)
		dW2 := make([]float32, D*H)
		MatMulRaw(h1T, dOut, dW2, D, H, N)
		gW2 := m.Experts[e].Fc2.Weights.Grad.Data
		for i, v := range dW2 {
			gW2[i] += v
		}
		// dB2.
		if m.Experts[e].Fc2.Biases != nil {
			gB2 := m.Experts[e].Fc2.Biases.Grad.Data
			for n := 0; n < N; n++ {
				nb := n * H
				for h := 0; h < H; h++ {
					gB2[h] += dOut[nb+h]
				}
			}
		}
		// dH1 = (dOut @ W2^T) * tanh'(h1).
		W2T := transposeCopy(W2, D, H) // [H,D]
		dH1pre := make([]float32, N*D)
		MatMulRaw(dOut, W2T, dH1pre, N, D, H)
		dH1 := make([]float32, N*D)
		for i := range dH1 {
			t := h1[i]
			dH1[i] = dH1pre[i] * (1 - t*t)
		}
		// dW1 = x^T @ dH1 : [H,D].
		dW1 := make([]float32, H*D)
		MatMulRaw(xT, dH1, dW1, H, D, N)
		gW1 := m.Experts[e].Fc1.Weights.Grad.Data
		for i, v := range dW1 {
			gW1[i] += v
		}
		// dB1.
		if m.Experts[e].Fc1.Biases != nil {
			gB1 := m.Experts[e].Fc1.Biases.Grad.Data
			for n := 0; n < N; n++ {
				nb := n * D
				for d := 0; d < D; d++ {
					gB1[d] += dH1[nb+d]
				}
			}
		}
		// dX += dH1 @ W1^T : [N,H].
		W1T := transposeCopy(m.Experts[e].Fc1.Weights.Data, H, D) // [D,H]
		dXe := make([]float32, N*H)
		MatMulRaw(dH1, W1T, dXe, N, H, D)
		for i, v := range dXe {
			dX[i] += v
		}

		// Router: dGate for this expert's (n,k) slots, then quotient rule
		// through the renormalization g_e = p_e / S.
		oe := m.expertOut[e]
		for n := 0; n < N; n++ {
			for k := 0; k < K; k++ {
				if m.topIdx[n*K+k] != e {
					continue
				}
				dg := float32(0)
				nb := n * H
				for h := 0; h < H; h++ {
					dg += dYd[nb+h] * oe[nb+h]
				}
				S := float32(0)
				for kk := 0; kk < K; kk++ {
					S += m.probs[n*E+m.topIdx[n*K+kk]]
				}
				invS2 := 1 / (S * S)
				pe := m.probs[n*E+e]
				for kk := 0; kk < K; kk++ {
					j := m.topIdx[n*K+kk]
					var dpdj float32
					if j == e {
						dpdj = dg * (S - pe) * invS2
					} else {
						dpdj = dg * (-pe) * invS2
					}
					dProbs[n*E+j] += dpdj
				}
			}
		}
	}

	// Load-balancing aux loss gradient: d(aux)/d p_{n,e} = auxCoeff * E * frac_e / N
	// (frac treated as constant, per Switch Transformer).
	if auxCoeff != 0 {
		scale := auxCoeff * float32(E) / float32(N)
		for n := 0; n < N; n++ {
			for e := 0; e < E; e++ {
				dProbs[n*E+e] += scale * m.frac[e]
			}
		}
	}

	// Softmax backward per token: dL_j = p_j * (dp_j - sum_e dp_e * p_e).
	// Then router param grads and dX.
	dLogits := make([]float32, N*E)
	WR := m.Router.Weights.Data
	WRT := transposeCopy(WR, H, E) // [E,H]
	for n := 0; n < N; n++ {
		dot := float32(0)
		for e := 0; e < E; e++ {
			dot += dProbs[n*E+e] * m.probs[n*E+e]
		}
		for e := 0; e < E; e++ {
			dLogits[n*E+e] = m.probs[n*E+e] * (dProbs[n*E+e] - dot)
		}
		// dX[n] += dLogits[n] @ WRouter^T
		for h := 0; h < H; h++ {
			s := float32(0)
			for e := 0; e < E; e++ {
				s += dLogits[n*E+e] * WRT[e*H+h]
			}
			dX[n*H+h] += s
		}
	}
	// dWRouter = x^T @ dLogits : [H,E].
	dWR := make([]float32, H*E)
	MatMulRaw(xT, dLogits, dWR, H, E, N)
	gWR := m.Router.Weights.Grad.Data
	for i, v := range dWR {
		gWR[i] += v
	}
	if m.Router.Biases != nil {
		gBR := m.Router.Biases.Grad.Data
		for n := 0; n < N; n++ {
			for e := 0; e < E; e++ {
				gBR[e] += dLogits[n*E+e]
			}
		}
	}

	dXt := NewTensor(m.input.Shape, dX, false)
	return dXt, nil
}
