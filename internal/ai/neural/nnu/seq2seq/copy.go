package seq2seq

// Minimal pointer/copy mechanism over the decoder's own output history.
//
// Motivation: the tiny seq2seq decoder must regenerate identifier tokens
// (e.g. the accumulator `total` in a sum loop) from its LSTM state alone,
// and at long range it drops them ("total += n" -> "+= n"). The copy
// mechanism lets the decoder point back at a token it already emitted: at
// each step it attends over its own past decoder states, scatters the
// attention mass onto the tokens that produced those states to form a copy
// distribution over the vocabulary, and mixes it with the generation
// logits under a learned gate.
//
// Stability design (a full probability-mixture copy variant diverged at
// the small model size, loss 8-9 vs the 0.02 baseline):
//   - The copy contribution is an ADDITIVE BIAS in logit space, not a
//     mixture of distributions:
//         final[v] = genLogit[v] + gate * log(copyMass[v] + eps)
//     The existing softmax-cross-entropy loss and its backward pass are
//     untouched; gradients flow through the bias into the attention
//     scores and the gate via closed-form, numerically guarded
//     expressions (see backwardCopy).
//   - Scaled dot-product attention with NO learned projection matrices:
//     score_i = (query . hist_i) / sqrt(H). The only new parameters are
//     the gate vector + bias, initialized so the gate starts nearly
//     closed (gate ~= 0.12): the model behaves like the baseline until
//     copying proves useful.
//   - The gate and all intermediate quantities are guarded against
//     log(0)/division-by-zero with an explicit epsilon.

import (
	"math"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// copyEps guards the log of the copy mass against log(0).
const copyEps = 1e-6

// CopyGate is the learnable part of the copy mechanism: a single logistic
// gate over the decoder state deciding how much copy bias to apply.
type CopyGate struct {
	Wg        *tensor.Tensor // [H] gate weights, zero-initialized
	Bg        *tensor.Tensor // [1] gate bias, initialized to -2 (gate starts ~0.12)
	HiddenDim int
	// UsePosEnc adds sinusoidal positional encodings to the attention keys.
	// This is for the encoder-side copy gate: encoder LSTM states can be
	// too similar across positions (mean cosine sim ~0.70) for dot-product
	// attention to focus sharply. Positional encodings give each position
	// a distinctive signature. Not used for the history copy gate (gocode),
	// where recency is already encoded.
	UsePosEnc bool
}

// NewCopyGate creates a copy gate for a decoder with the given hidden dim.
func NewCopyGate(hiddenDim int) *CopyGate {
	wg := tensor.NewTensor([]int{hiddenDim}, make([]float32, hiddenDim), true)
	wg.RequiresGrad = true
	bg := tensor.NewTensor([]int{1}, []float32{-2.0}, true)
	bg.RequiresGrad = true
	return &CopyGate{Wg: wg, Bg: bg, HiddenDim: hiddenDim}
}

// Parameters returns the learnable parameters. Nil-safe so callers can
// treat a nil gate as "no copy mechanism".
func (g *CopyGate) Parameters() []*tensor.Tensor {
	if g == nil {
		return nil
	}
	return []*tensor.Tensor{g.Wg, g.Bg}
}

// GateValue returns the current gate value for a query (inference/debug).
func (g *CopyGate) GateValue(query []float32) float32 {
	z := g.Bg.Data[0]
	for h, q := range query {
		z += g.Wg.Data[h] * q
	}
	return copySigmoid(z)
}

func copySigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// posEncSinusoidal returns the sinusoidal positional encoding for position
// pos with dimension d. Standard transformer formulation.
func posEncSinusoidal(pos, d int) []float32 {
	pe := make([]float32, d)
	for i := 0; i < d; i += 2 {
		div := float32(math.Pow(10000, float64(i)/float64(d)))
		pe[i] = float32(math.Sin(float64(pos) / float64(div)))
		if i+1 < d {
			pe[i+1] = float32(math.Cos(float64(pos) / float64(div)))
		}
	}
	return pe
}
type copyCache struct {
	gate     float32
	w        []float32 // [nh] attention weights over history
	accum    []float32 // [V] copy mass scattered onto vocab ids
	logAccum []float32 // [V] log(accum[v] + eps)
	query    []float32 // [H] copy of the query
	keys     [][]float32 // [nh][H] augmented keys (with pos enc); nil if not used
	nh       int
}

// forwardCopyStep applies the copy bias in place:
// final[v] = genLogits[v] + gate * log(copyMass[v] + eps).
//
// query:      [H] decoder state used for generation (post-MoE)
// genLogits:  [V] generation logits; updated IN PLACE with the copy bias
// histStates: [nh][H] past decoder LSTM states (nh may be 0)
// histTokens: token ids that produced histStates (len nh)
//
// Returns the cache needed by backwardCopyStep.
func (g *CopyGate) forwardCopyStep(query, genLogits []float32, histStates [][]float32, histTokens []int) *copyCache {
	vocabSize := len(genLogits)
	c := &copyCache{
		nh:       len(histStates),
		query:    append([]float32(nil), query...),
		logAccum: make([]float32, vocabSize),
		accum:    make([]float32, vocabSize),
	}

	z := g.Bg.Data[0]
	for h, q := range query {
		z += g.Wg.Data[h] * q
	}
	c.gate = copySigmoid(z)
	if c.nh == 0 {
		// No history: logAccum stays zero, bias is zero, logits unchanged.
		return c
	}

	// Keys for attention: optionally augment with sinusoidal positional
	// encodings so each position has a distinctive signature. (Encoder
	// LSTM states are too similar across positions for sharp dot-product
	// attention otherwise.)
	keys := histStates
	if g.UsePosEnc {
		keys = make([][]float32, c.nh)
		d := len(query)
		for i, hs := range histStates {
			pe := posEncSinusoidal(i, d)
			k := make([]float32, d)
			for h := 0; h < d && h < len(hs); h++ {
				k[h] = hs[h] + pe[h]
			}
			keys[i] = k
		}
		// Store augmented keys for the backward pass (dQuery path).
		c.keys = keys
	}

	invSqrtH := float32(1.0 / math.Sqrt(float64(len(query))))
	scores := make([]float32, c.nh)
	for i, ks := range keys {
		var s float32
		for h, q := range query {
			s += q * ks[h]
		}
		scores[i] = s * invSqrtH
	}
	// Softmax with max-subtraction for numerical stability.
	mx := scores[0]
	for _, s := range scores[1:] {
		if s > mx {
			mx = s
		}
	}
	var sum float64
	c.w = make([]float32, c.nh)
	for i, s := range scores {
		e := math.Exp(float64(s - mx))
		c.w[i] = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := range c.w {
		c.w[i] *= inv
	}
	// Scatter attention mass onto vocabulary ids.
	for i, tok := range histTokens {
		if tok >= 0 && tok < vocabSize {
			c.accum[tok] += c.w[i]
		}
	}
	for v := 0; v < vocabSize; v++ {
		c.logAccum[v] = float32(math.Log(float64(c.accum[v] + copyEps)))
		genLogits[v] += c.gate * c.logAccum[v]
	}
	return c
}

// backwardCopyStep propagates dLogits (dL/d final logits, [V]) back through
// the copy mechanism. It returns dQuery [H] (to add into the generation
// state's gradient) and dHist [nh][H] (to add into the history states'
// gradient), and accumulates the gate parameter gradients into
// g.Wg.Grad / g.Bg.Grad.
func (g *CopyGate) backwardCopyStep(dLogits []float32, c *copyCache, histStates [][]float32, histTokens []int) ([]float32, [][]float32) {
	vocabSize := len(dLogits)
	hDim := len(c.query)

	// dL/dgate = sum_v dLogits[v] * logAccum[v]
	var dGate float32
	for v := 0; v < vocabSize; v++ {
		dGate += dLogits[v] * c.logAccum[v]
	}
	dz := dGate * c.gate * (1 - c.gate)

	// Gate parameter gradients (accumulate, following the nn layers'
	// lazy-Grad convention).
	ensureCopyGrad(g.Wg)
	ensureCopyGrad(g.Bg)
	for h := 0; h < hDim; h++ {
		g.Wg.Grad.Data[h] += dz * c.query[h]
	}
	g.Bg.Grad.Data[0] += dz

	// dQuery: gate path + attention-score path.
	dQuery := make([]float32, hDim)
	for h := 0; h < hDim; h++ {
		dQuery[h] = dz * g.Wg.Data[h]
	}
	var dHist [][]float32
	if c.nh == 0 {
		return dQuery, nil
	}

	// dL/d logAccum[v] = dLogits[v] * gate; each history position i feeds
	// exactly one vocab id tok_i, so dAccum_i = dLogAccum[tok_i]/(accum+eps).
	dAccum := make([]float32, c.nh)
	for i, tok := range histTokens {
		if tok >= 0 && tok < vocabSize {
			dAccum[i] = dLogits[tok] * c.gate / (c.accum[tok] + copyEps)
		}
	}
	// Softmax backward: dw_i = w_i * (dAccum_i - sum_j dAccum_j * w_j).
	var sDot float32
	for i := range dAccum {
		sDot += dAccum[i] * c.w[i]
	}
	invSqrtH := float32(1.0 / math.Sqrt(float64(hDim)))
	dHist = make([][]float32, c.nh)
	// Use augmented keys (with pos enc) for the dQuery path if present;
	// dHist is w.r.t. the original states (pos enc is fixed, so the
	// gradient passes through unchanged).
	keys := histStates
	if c.keys != nil {
		keys = c.keys
	}
	for i := range dAccum {
		dw := c.w[i] * (dAccum[i] - sDot)
		dh := make([]float32, hDim)
		ks := keys[i]
		for h := 0; h < hDim; h++ {
			dQuery[h] += dw * ks[h] * invSqrtH
			dh[h] = dw * c.query[h] * invSqrtH
		}
		dHist[i] = dh
	}
	return dQuery, dHist
}

// ensureCopyGrad lazily allocates the gradient buffer of a parameter,
// mirroring the convention in internal/ai/neural/nn/layers.go.
func ensureCopyGrad(t *tensor.Tensor) {
	if t.Grad == nil {
		t.Grad = tensor.NewTensor(append([]int(nil), t.Shape...), make([]float32, len(t.Data)), false)
	}
}

// stepHistory returns the copy history for batch row b at decode step t:
// the decoder's own past LSTM states (views into decOut, a [B, steps, H]
// tensor) and the input token ids that produced them. Empty (nil, nil) at
// t == 0.
func stepHistory(decOut *tensor.Tensor, decInputIDs [][]int, b, t, steps, hiddenSize int) ([][]float32, []int) {
	if t == 0 {
		return nil, nil
	}
	hist := make([][]float32, t)
	toks := make([]int, t)
	for i := 0; i < t; i++ {
		off := (b*steps + i) * hiddenSize
		hist[i] = decOut.Data[off : off+hiddenSize]
		toks[i] = decInputIDs[b][i]
	}
	return hist, toks
}
