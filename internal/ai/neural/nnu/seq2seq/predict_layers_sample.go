package seq2seq

// Temperature-sampled inference for LayersModel, the sampling
// twin of PredictLayers — seated in the product 2026-11-18
// after the decode probe measured checked sampling beating
// greedy on the adopted model (240/426 vs 176/426 on the pack-2
// combination holdout at temperature 0.8; layers_decode.go in
// the chat package carries the selection policy). The ONLY
// change from the greedy path is the per-step token choice: the
// encoder block, the seed construction, the top-prefix step,
// the copy setup, the EOS/pad stop, the first-step unk skip,
// and the bigram bookkeeping are renderReply's, byte for byte
// in behavior; the bans argmaxBannedProbs / argmaxBannedLogits
// apply are applied identically, as zeroed probability mass
// (copy branch) or -inf-masked logits (generation branch)
// before the temperature softmax.
//
// The frame is NOT reimplemented: if either frame half is live
// for the call, PredictLayersSampled returns an error and the
// caller serves the greedy render. The adopted model is
// frameless (the copy-only gob), where the frame, its checks,
// and the repair loop are inert by construction.

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// PredictLayersSampled mirrors PredictLayers (encoder block and
// output assembly) but renders the reply by temperature sampling
// instead of greedy argmax. rng is consumed only by the token
// choices; seeding discipline belongs to the caller.
func (m *LayersModel) PredictLayersSampled(inputIDs []int, surfaces []string, opts LayersPredictOptions, temp float64, rng *rand.Rand) (LayersPrediction, error) {
	cfg := m.Config
	base := m.Base
	hidden := cfg.HiddenDim
	embDim := cfg.EmbedDim
	maxLen := opts.MaxLen
	if maxLen <= 0 {
		maxLen = 60
	}
	out := LayersPrediction{Top: -1, Act: -1, Tone: -1, ReplyTop: -1}
	if len(inputIDs) == 0 {
		return out, nil
	}
	if hasTop, hasSlots := m.frameParts(opts); hasTop || hasSlots {
		return out, fmt.Errorf("PredictLayersSampled: a frame half is live (top=%v slots=%v); the sampled render supports frameless models only", hasTop, hasSlots)
	}

	// ---- Encoder (3D values pass, as in training) — identical
	// to PredictLayers.
	B, T := 1, len(inputIDs)
	tokRows := make([]float32, T*embDim)
	for t, id := range inputIDs {
		off := id * embDim
		copy(tokRows[t*embDim:(t+1)*embDim], base.Encoder.Embedding.Weight.Data[off:off+embDim])
	}
	comb := m.combineRows(tokRows, [][]int{inputIDs}, [][]string{surfaces}, B, T)
	encEmb := tensor.NewTensor([]int{B, T, embDim}, comb, false)
	h0 := tensor.NewTensor([]int{B, hidden}, make([]float32, hidden), false)
	c0 := tensor.NewTensor([]int{B, hidden}, make([]float32, hidden), false)
	encOut, encCell, err := base.Encoder.LSTM.Forward(encEmb, h0, c0)
	if err != nil {
		return out, err
	}
	encFinal := make([]float32, hidden)
	copy(encFinal, encOut.Data[(T-1)*hidden:T*hidden])

	// No frame parts are live (checked above), so the frame the
	// render runs under is the empty one PredictLayers settles
	// on for a frameless model.
	frame := FrameState{Top: -1, Act: -1, Tone: -1}
	out.TopicWords = ContentWordsOf(surfaces)
	ids, text, rerr := m.renderReplySampled(encOut, encCell, encFinal, frame, inputIDs, opts, maxLen, temp, rng)
	if rerr != nil {
		return out, rerr
	}
	out.TokenIDs, out.Text = ids, text
	out.Top, out.Act, out.Tone = frame.Top, frame.Act, frame.Tone
	if frame.topDist != nil {
		out.TopDist = append([]float32{}, frame.topDist...)
	}
	if rt, ok := m.DeriveReplyTop(ids); ok {
		out.ReplyTop = rt
	}
	return out, nil
}

// renderReplySampled is renderReply with the token choice
// sampled: everything else — seed, top-prefix step, copy
// encoder-state setup, stop rules, unk handling, bigram
// bookkeeping, fedSurface — is the greedy render's own code
// path, unchanged.
func (m *LayersModel) renderReplySampled(encOut, encCell *tensor.Tensor, encFinal []float32, frame FrameState, inputIDs []int, opts LayersPredictOptions, maxLen int, temp float64, rng *rand.Rand) ([]int, string, error) {
	cfg := m.Config
	base := m.Base
	vocab := base.OutputVocab
	hidden := cfg.HiddenDim
	embDim := cfg.EmbedDim
	T := len(inputIDs)

	seed := make([]float32, hidden)
	copy(seed, encFinal)
	if frame.Act >= 0 && m.ActEmb != nil {
		m.addEmbRow(seed, m.ActEmb, frame.Act)
	}
	if frame.Tone >= 0 && m.ToneEmb != nil {
		m.addEmbRow(seed, m.ToneEmb, frame.Tone)
	}
	if m.layerOn("turn", opts) {
		if opts.PrevReplyIDs != nil {
			if prevTop, ok := m.DeriveReplyTop(opts.PrevReplyIDs); ok {
				m.addEmbRow(seed, m.TurnEmb, prevTop)
			} else {
				for k := 0; k < hidden; k++ {
					seed[k] += m.NoneTurn.Data[k]
				}
			}
		} else {
			for k := 0; k < hidden; k++ {
				seed[k] += m.NoneTurn.Data[k]
			}
		}
	}
	h := tensor.NewTensor([]int{1, hidden}, seed, false)
	c := encCell

	// Top-code prefix step (inert without a frame: frame.Top is
	// -1 on every model this probe supports).
	if frame.Top >= 0 && m.TopEmb != nil {
		var emb []float32
		if frame.topDist != nil {
			emb = make([]float32, embDim)
			for k, p := range frame.topDist {
				if p != 0 {
					m.addEmbRowScaled(emb, m.TopEmb, k, p)
				}
			}
		} else {
			off := frame.Top * embDim
			emb = append([]float32{}, m.TopEmb.Weight.Data[off:off+embDim]...)
		}
		_, _, nh, nc, err := m.decStep(emb, h, c)
		if err != nil {
			return nil, "", err
		}
		h, c = nh, nc
	}

	// Encoder states for the copy layer.
	var encStates [][]float32
	validLen := 0
	if m.layerOn("copy", opts) {
		encStates = make([][]float32, T)
		for t := 0; t < T; t++ {
			encStates[t] = encOut.Data[t*hidden : (t+1)*hidden]
			if inputIDs[t] != vocab.PaddingTokenID {
				validLen = t + 1
			}
		}
	}

	prevID := -1
	usedBigrams := map[[2]int]bool{}
	fedID := vocab.BosID
	fedSurface := ""
	var tokenIDs []int
	for step := 0; step < maxLen; step++ {
		emb := m.wordStepEmbedding(fedID, fedSurface)
		logits, moeRow, nh, nc, err := m.decStep(emb, h, c)
		if err != nil {
			return nil, "", err
		}
		h, c = nh, nc
		var nextID int
		if m.layerOn("copy", opts) {
			probs, _ := base.Decoder.PtrGen.ptrGenStep(moeRow, encStates, inputIDs, validLen, logits)
			nextID = sampleBannedProbs(probs, prevID, usedBigrams, tokenIDs, temp, rng)
		} else {
			nextID = sampleBannedLogits(logits, prevID, usedBigrams, tokenIDs, temp, rng)
		}
		if (nextID == vocab.EosID && len(tokenIDs) > 0) || nextID == vocab.PaddingTokenID {
			break
		}
		if nextID == vocab.UnkID {
			if step == 0 {
				// Production skips a first-step <unk> without appending.
				fedID = nextID
				fedSurface = ""
				continue
			}
			break
		}
		tokenIDs = append(tokenIDs, nextID)
		if prevID >= 0 {
			usedBigrams[[2]int{prevID, nextID}] = true
		}
		prevID = nextID
		fedID = nextID
		fedSurface = vocab.GetWord(nextID)
	}
	return tokenIDs, vocab.Decode(tokenIDs), nil
}

// bannedAt reports whether index i is banned at this step under
// exactly the conditions argmaxBannedProbs / argmaxBannedLogits
// skip: the previous token, or a continuation that repeats a
// bigram already produced.
func bannedAt(i, prevID int, used map[[2]int]bool, generated []int) bool {
	if i == prevID {
		return true
	}
	if len(generated) > 0 && used[[2]int{generated[len(generated)-1], i}] {
		return true
	}
	return false
}

// sampleBannedProbs draws one token from the copy mixture with
// the banned indices' mass zeroed, the survivors raised to
// 1/temp and renormalized. A zero total (every survivor had no
// mass) falls back to the greedy argmax under the same bans; a
// non-positive temperature is greedy by definition here.
func sampleBannedProbs(probs []float32, prevID int, used map[[2]int]bool, generated []int, temp float64, rng *rand.Rand) int {
	if temp <= 0 {
		return argmaxBannedProbs(probs, prevID, used, generated)
	}
	inv := 1.0 / temp
	weights := make([]float64, len(probs))
	var total float64
	for i, p := range probs {
		if p <= 0 || bannedAt(i, prevID, used, generated) {
			continue
		}
		w := math.Pow(float64(p), inv)
		weights[i] = w
		total += w
	}
	if total <= 0 {
		return argmaxBannedProbs(probs, prevID, used, generated)
	}
	return drawWeighted(weights, total, rng)
}

// sampleBannedLogits draws one token from the generation logits
// with the banned indices masked to -inf and a temperature
// softmax over the survivors. The fallbacks mirror
// sampleBannedProbs.
func sampleBannedLogits(logits []float32, prevID int, used map[[2]int]bool, generated []int, temp float64, rng *rand.Rand) int {
	if temp <= 0 {
		return argmaxBannedLogits(logits, prevID, used, generated, 0)
	}
	mx := math.Inf(-1)
	for i, v := range logits {
		if bannedAt(i, prevID, used, generated) {
			continue
		}
		if s := float64(v) / temp; s > mx {
			mx = s
		}
	}
	if math.IsInf(mx, -1) {
		return argmaxBannedLogits(logits, prevID, used, generated, 0)
	}
	weights := make([]float64, len(logits))
	var total float64
	for i, v := range logits {
		if bannedAt(i, prevID, used, generated) {
			continue
		}
		w := math.Exp(float64(v)/temp - mx)
		weights[i] = w
		total += w
	}
	if total <= 0 {
		return argmaxBannedLogits(logits, prevID, used, generated, 0)
	}
	return drawWeighted(weights, total, rng)
}

// drawWeighted picks an index with probability weights[i]/total.
func drawWeighted(weights []float64, total float64, rng *rand.Rand) int {
	r := rng.Float64() * total
	acc := 0.0
	last := -1
	for i, w := range weights {
		if w <= 0 {
			continue
		}
		acc += w
		last = i
		if r < acc {
			return i
		}
	}
	return last
}
