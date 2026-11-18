package seq2seq

// Pointer-generator head over the ENCODER's input positions.
//
// The copy probe (2026-10-29) tests the one mechanism the decl and
// spec experiments had not: their decoders generated every output
// token from a learned distribution over the vocabulary, and on
// unseen combinations they failed exactly on the tokens that were
// already present in the input — member names (~45% slot accuracy)
// and member types (28.9%). A pointer-generator (See et al. 2017)
// makes copying part of the ARCHITECTURE: at each decoder step the
// output distribution is a learned mixture
//
//	final = p_gen * p_vocab + (1 - p_gen) * copyDist
//
// where p_vocab is the usual softmax over the vocabulary, copyDist
// scatters the step's attention weights over the encoder positions
// onto the tokens sitting at those positions, and p_gen is a learned
// scalar gate over [decoder state; attention context]. The mixture
// is the trained output distribution itself — gradients flow through
// p_gen, the attention scores, and both branches (see train_ptrgen.go
// and ptrGenBackward). This is deliberately NOT the CopyGate in
// copy.go (an additive logit bias over the decoder's own history) and
// NOT an eval-time mask: at inference the model sees only the input
// token sequence, and greedy decoding reads this same distribution.
//
// Stability design (a mixture copy variant diverged in this codebase
// before, at the gocode model size): the gate weights start at zero
// and the gate bias at +2, so p_gen ~= 0.88 at initialization — the
// model starts as (mostly) the baseline generator and must LEARN to
// copy where copying pays. The loss clamps the target probability at
// 1e-12, so a confidently wrong step costs ~27 nats, never NaN/Inf.

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// ptrGenProbFloor clamps the mixture probability of a target token
// before the log, bounding the per-token loss.
const ptrGenProbFloor = 1e-12

// PointerGen holds the learnable parameters of the pointer-generator
// head. It is carried on the Decoder (nil = mechanism absent, the
// pre-probe behavior for every other domain).
type PointerGen struct {
	Wq        *tensor.Tensor // [H*H] attention query projection, row-major [H][H]
	Wg        *tensor.Tensor // [2H] generation gate over [decoder state; context]
	Bg        *tensor.Tensor // [1] gate bias, initialized to +2 (p_gen ~= 0.88)
	HiddenDim int
}

// NewPointerGen creates a pointer-generator head for hidden size H.
// The query projection gets small random weights (scale 1/sqrt(H));
// the gate starts closed-form at p_gen = sigmoid(2) for any state.
func NewPointerGen(hiddenDim int) *PointerGen {
	scale := 1.0 / math.Sqrt(float64(hiddenDim))
	wq := make([]float32, hiddenDim*hiddenDim)
	for i := range wq {
		wq[i] = float32(rand.NormFloat64() * scale)
	}
	wqT := tensor.NewTensor([]int{hiddenDim, hiddenDim}, wq, true)
	wqT.RequiresGrad = true
	wgT := tensor.NewTensor([]int{2 * hiddenDim}, make([]float32, 2*hiddenDim), true)
	wgT.RequiresGrad = true
	bgT := tensor.NewTensor([]int{1}, []float32{2.0}, true)
	bgT.RequiresGrad = true
	return &PointerGen{Wq: wqT, Wg: wgT, Bg: bgT, HiddenDim: hiddenDim}
}

// Parameters returns the head's learnable parameters. Nil-safe.
func (p *PointerGen) Parameters() []*tensor.Tensor {
	if p == nil {
		return nil
	}
	return []*tensor.Tensor{p.Wq, p.Wg, p.Bg}
}

// ptrGenCache holds one step's intermediates for the backward pass.
type ptrGenCache struct {
	attn     []float32 // [validLen] attention over encoder positions
	pVocab   []float32 // [V] generation distribution
	copyDist []float32 // [V] copy distribution over the vocabulary
	context  []float32 // [H] attention-weighted encoder state
	query    []float32 // [H] projected decoder state
	decState []float32 // [H] copy of the decoder state
	pGen     float32
	validLen int
}

// ptrGenStep computes the mixture distribution for one decoder step.
//
//	decState:  [H] decoder LSTM state at this step
//	encStates: [encLen][H] encoder states (views are fine)
//	inputIDs:  [encLen] encoder token ids (vocab ids)
//	validLen:  number of leading non-padding encoder positions
//	genLogits: [V] vocabulary logits from the output projection
//
// Returns the final mixture distribution [V] and the backward cache.
func (p *PointerGen) ptrGenStep(decState []float32, encStates [][]float32, inputIDs []int, validLen int, genLogits []float32) ([]float32, *ptrGenCache) {
	H := p.HiddenDim
	V := len(genLogits)
	c := &ptrGenCache{decState: append([]float32(nil), decState...), validLen: validLen}

	// Query projection: q = Wq @ decState.
	c.query = make([]float32, H)
	for i := 0; i < H; i++ {
		var s float32
		row := p.Wq.Data[i*H : (i+1)*H]
		for j := 0; j < H; j++ {
			s += row[j] * decState[j]
		}
		c.query[i] = s
	}

	// Scaled dot-product attention over the valid encoder positions.
	invSqrtH := float32(1.0 / math.Sqrt(float64(H)))
	scores := make([]float32, validLen)
	for i := 0; i < validLen; i++ {
		var s float32
		for h := 0; h < H; h++ {
			s += c.query[h] * encStates[i][h]
		}
		scores[i] = s * invSqrtH
	}
	c.attn = softmax32(scores)

	// Context: attention-weighted encoder state.
	c.context = make([]float32, H)
	for i := 0; i < validLen; i++ {
		a := c.attn[i]
		if a == 0 {
			continue
		}
		for h := 0; h < H; h++ {
			c.context[h] += a * encStates[i][h]
		}
	}

	// Copy distribution: scatter attention onto the tokens at the
	// attended positions.
	c.copyDist = make([]float32, V)
	for i := 0; i < validLen; i++ {
		tok := inputIDs[i]
		if tok >= 0 && tok < V {
			c.copyDist[tok] += c.attn[i]
		}
	}

	// Generation distribution.
	c.pVocab = softmax32(genLogits)

	// Learned gate: p_gen = sigmoid(Wg . [decState; context] + Bg).
	z := p.Bg.Data[0]
	for h := 0; h < H; h++ {
		z += p.Wg.Data[h] * decState[h]
		z += p.Wg.Data[H+h] * c.context[h]
	}
	c.pGen = copySigmoid(z)

	final := make([]float32, V)
	for v := 0; v < V; v++ {
		final[v] = c.pGen*c.pVocab[v] + (1-c.pGen)*c.copyDist[v]
	}
	return final, c
}

// ptrGenBackward propagates dL/d(final[label]) — already scaled by
// the caller's mean-loss factor — through one mixture step. It
// accumulates parameter gradients into the head, and returns the
// gradients for the caller to route onward:
//
//	dLogits:  [V] into the output projection's softmax input
//	dDec:     [H] into the decoder state
//	dEnc:     [validLen][H] into the encoder states
func (p *PointerGen) ptrGenBackward(cache *ptrGenCache, encStates [][]float32, inputIDs []int, label int, dFinal float32) (dLogits, dDec []float32, dEnc [][]float32) {
	H := p.HiddenDim
	V := len(cache.pVocab)

	// Gate path: d p_gen = dFinal * (pVocab[label] - copyDist[label]).
	dpGen := dFinal * (cache.pVocab[label] - cache.copyDist[label])
	dz := dpGen * cache.pGen * (1 - cache.pGen)
	ensureCopyGrad(p.Wg)
	ensureCopyGrad(p.Bg)
	ensureCopyGrad(p.Wq)
	dDec = make([]float32, H)
	dCtx := make([]float32, H)
	for h := 0; h < H; h++ {
		p.Wg.Grad.Data[h] += dz * cache.decState[h]
		p.Wg.Grad.Data[H+h] += dz * cache.context[h]
		dDec[h] += dz * p.Wg.Data[h]
		dCtx[h] = dz * p.Wg.Data[H+h]
	}
	p.Bg.Grad.Data[0] += dz

	// Generation path: d pVocab is nonzero only at the label; push it
	// back through the vocabulary softmax to the logits.
	dpVocabLabel := dFinal * cache.pGen
	var dot float32
	dot = dpVocabLabel * cache.pVocab[label]
	dLogits = make([]float32, V)
	for v := 0; v < V; v++ {
		ind := float32(0)
		if v == label {
			ind = dpVocabLabel
		}
		dLogits[v] = cache.pVocab[v] * (ind - dot)
	}

	// Copy + context paths into the attention weights.
	dAttn := make([]float32, cache.validLen)
	dCopyLabel := dFinal * (1 - cache.pGen)
	for i := 0; i < cache.validLen; i++ {
		if inputIDs[i] == label {
			dAttn[i] += dCopyLabel
		}
		var s float32
		for h := 0; h < H; h++ {
			s += dCtx[h] * encStates[i][h]
		}
		dAttn[i] += s
	}
	dEnc = make([][]float32, cache.validLen)
	for i := range dEnc {
		dEnc[i] = make([]float32, H)
		for h := 0; h < H; h++ {
			dEnc[i][h] += cache.attn[i] * dCtx[h]
		}
	}

	// Attention softmax backward, then through the scaled dot product
	// into the query and the encoder states.
	var attnDot float32
	for i := 0; i < cache.validLen; i++ {
		attnDot += dAttn[i] * cache.attn[i]
	}
	invSqrtH := float32(1.0 / math.Sqrt(float64(H)))
	dQuery := make([]float32, H)
	for i := 0; i < cache.validLen; i++ {
		dScore := cache.attn[i] * (dAttn[i] - attnDot)
		if dScore == 0 {
			continue
		}
		for h := 0; h < H; h++ {
			dQuery[h] += dScore * encStates[i][h] * invSqrtH
			dEnc[i][h] += dScore * cache.query[h] * invSqrtH
		}
	}

	// Query projection backward: q = Wq @ decState.
	for i := 0; i < H; i++ {
		if dQuery[i] == 0 {
			continue
		}
		row := p.Wq.Grad.Data[i*H : (i+1)*H]
		wrow := p.Wq.Data[i*H : (i+1)*H]
		for j := 0; j < H; j++ {
			row[j] += dQuery[i] * cache.decState[j]
			dDec[j] += dQuery[i] * wrow[j]
		}
	}
	return dLogits, dDec, dEnc
}

// GateValue reports the current p_gen for a state/context pair
// (inference diagnostics and tests).
func (p *PointerGen) GateValue(decState, context []float32) float32 {
	z := p.Bg.Data[0]
	for h := 0; h < p.HiddenDim; h++ {
		z += p.Wg.Data[h] * decState[h]
		z += p.Wg.Data[p.HiddenDim+h] * context[h]
	}
	return copySigmoid(z)
}

// softmax32 is a numerically stable softmax over a float32 slice.
func softmax32(x []float32) []float32 {
	out := make([]float32, len(x))
	if len(x) == 0 {
		return out
	}
	mx := x[0]
	for _, v := range x[1:] {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - mx))
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := range out {
		out[i] *= inv
	}
	return out
}

// PredictPtrGen greedily decodes from the pointer-generator mixture.
// It mirrors Predict's decoding discipline — same input
// normalization, same anti-repeat masking (no immediate repeat, no
// repeated bigram) — because the spec probe's numbers were produced
// under that discipline and the copy probe must compare head-to-head.
// There is NO constrained decoding and NO retry: the mixture itself
// is the only mechanism under test.
func (m *Seq2Seq) PredictPtrGen(query string, maxLen int) (string, error) {
	if m.Decoder == nil || m.Decoder.PtrGen == nil {
		return "", fmt.Errorf("seq2seq PredictPtrGen: model has no pointer-generator head")
	}
	if m.Tokenizer == nil {
		return "", fmt.Errorf("seq2seq tokenizer is nil")
	}
	pg := m.Decoder.PtrGen
	H := m.HiddenDim

	query = NormalizeQuery(query)
	inputIDs, err := m.Tokenizer.Encode(query)
	if err != nil {
		return "", fmt.Errorf("failed to tokenize query: %w", err)
	}
	if len(inputIDs) == 0 {
		return "", fmt.Errorf("empty query")
	}
	// At inference every position is valid (no batch padding), but a
	// PAD id can appear if the tokenizer maps something to it; treat
	// the full sequence as valid regardless — it is the real input.
	validLen := len(inputIDs)

	// Encoder: step-by-step 2D pass, recording every position's state
	// for the copy attention.
	encStates, hidden, cell, err := m.encodePositions(inputIDs)
	if err != nil {
		return "", err
	}
	_ = validLen

	outputTokens := []int{}
	seenBigrams := map[[2]int]bool{}
	current := m.OutputVocab.BosID

	for t := 0; t < maxLen; t++ {
		// Decoder step: embed the current token, advance the LSTM.
		emb, err := m.Decoder.Embedding.Forward(tensor.NewTensor([]int{1, 1}, []float32{float32(current)}, true))
		if err != nil {
			return "", fmt.Errorf("ptrgen decoder embedding failed at step %d: %w", t, err)
		}
		emb2D, err := emb.Reshape([]int{1, emb.Shape[2]})
		if err != nil {
			return "", fmt.Errorf("ptrgen decoder embedding reshape at step %d: %w", t, err)
		}
		newHidden, newCell, err := m.Decoder.LSTM.Forward(emb2D, hidden, cell)
		if err != nil {
			return "", fmt.Errorf("ptrgen decoder LSTM failed at step %d: %w", t, err)
		}
		logitsT, err := m.Decoder.Output.Forward(newHidden)
		if err != nil {
			return "", fmt.Errorf("ptrgen output projection failed at step %d: %w", t, err)
		}
		final, _ := pg.ptrGenStep(newHidden.Data[:H], encStates, inputIDs, len(inputIDs), logitsT.Data)

		// Anti-repeat masking, mirroring Predict: zero the previous
		// token and any token that would repeat a bigram.
		if len(outputTokens) > 0 {
			prev := outputTokens[len(outputTokens)-1]
			if prev >= 0 && prev < len(final) {
				final[prev] = 0
			}
			for c := 0; c < len(final); c++ {
				if seenBigrams[[2]int{prev, c}] {
					final[c] = 0
				}
			}
		}
		best := 0
		for i := 1; i < len(final); i++ {
			if final[i] > final[best] {
				best = i
			}
		}
		if len(outputTokens) > 0 {
			seenBigrams[[2]int{outputTokens[len(outputTokens)-1], best}] = true
		}
		outputTokens = append(outputTokens, best)
		if best == m.OutputVocab.EosID {
			break
		}
		current = best
		hidden, cell = newHidden, newCell
	}
	return strings.TrimSpace(m.OutputVocab.Decode(outputTokens)), nil
}

// encodePositions runs the encoder one token at a time and returns
// every position's hidden state plus the final (hidden, cell).
func (m *Seq2Seq) encodePositions(inputIDs []int) ([][]float32, *tensor.Tensor, *tensor.Tensor, error) {
	H := m.HiddenDim
	data := make([]float32, len(inputIDs))
	for i, id := range inputIDs {
		data[i] = float32(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, data, true)
	embedded, err := m.Encoder.Embedding.Forward(inputTensor)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("ptrgen encoder embedding failed: %w", err)
	}
	hidden := tensor.NewTensor([]int{1, H}, make([]float32, H), true)
	cell := tensor.NewTensor([]int{1, H}, make([]float32, H), true)
	states := make([][]float32, len(inputIDs))
	for t := range inputIDs {
		step, err := embedded.Slice(1, t, t+1)
		if err != nil {
			return nil, nil, nil, err
		}
		step2D, err := step.Reshape([]int{1, m.Encoder.Embedding.DimModel})
		if err != nil {
			return nil, nil, nil, err
		}
		hidden, cell, err = m.Encoder.LSTM.Forward(step2D, hidden, cell)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("ptrgen encoder LSTM failed at step %d: %w", t, err)
		}
		states[t] = append([]float32(nil), hidden.Data[:H]...)
	}
	return states, hidden, cell, nil
}
