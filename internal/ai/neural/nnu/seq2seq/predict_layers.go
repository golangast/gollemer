package seq2seq

// Layers pilot: greedy inference for LayersModel, mirroring the
// production PredictWithTrace walk (same anti-degeneration bans)
// with the layers' inference halves: heads predict intent/tone,
// the concept head's codes are fed as the decoder prefix (gold in
// training, predicted here), the seed gains the intent/tone/turn
// rows, and copy steps mix the pointer distribution in. The
// scorer's switches (DESIGN §6) ride on LayersPredictOptions:
// Disable forces trained layers off at inference; ShuffleConcept
// applies the registered fixed permutation to the predicted codes.
// These are diagnostics on a trained model, never ablation verdicts.

import (
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// LayersPredictOptions carries the per-call layer state.
type LayersPredictOptions struct {
	Disable        map[string]bool // layers forced off at inference
	ShuffleConcept bool            // apply c -> c%32+1 (codes >= 1)
	PrevReplyIDs   []int           // previous reply tokens (turn layer); nil = no previous turn
	MaxLen         int             // 0 = 60 (the house scorers' cap)
}

// LayersPrediction is one layers-model reply plus its diagnostics.
type LayersPrediction struct {
	Text         string
	TokenIDs     []int
	ConceptCodes [3]int // codes actually fed to the decoder prefix
	ReplyCodes   [3]int // codes derived from the generated reply (turn threading)
	Intent       int    // predicted intent label (-1 when the layer is off/untrained)
	Tone         int    // predicted tone label (-1 likewise)
}

// layerOn reports whether a layer is both trained into the model
// and not disabled for this call.
func (m *LayersModel) layerOn(name string, opts LayersPredictOptions) bool {
	if opts.Disable != nil && opts.Disable[name] {
		return false
	}
	return m.Config.On(name)
}

// PredictLayers generates one reply for the given input ids and
// their surface forms (aligned; surfaces drive the char layer).
func (m *LayersModel) PredictLayers(inputIDs []int, surfaces []string, opts LayersPredictOptions) (LayersPrediction, error) {
	cfg := m.Config
	base := m.Base
	vocab := base.OutputVocab
	hidden := cfg.HiddenDim
	embDim := cfg.EmbedDim
	maxLen := opts.MaxLen
	if maxLen <= 0 {
		maxLen = 60
	}
	out := LayersPrediction{Intent: -1, Tone: -1}
	if len(inputIDs) == 0 {
		return out, nil
	}

	// ---- Encoder (3D values pass, as in training).
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

	// ---- Heads predict the conditioning labels.
	if m.layerOn("intent", opts) {
		out.Intent = argmaxF(linearRow(m.IntentHead, encFinal))
	}
	if m.layerOn("tone", opts) {
		out.Tone = argmaxF(linearRow(m.ToneHead, encFinal))
	}
	var codes [3]int
	if m.layerOn("concept", opts) {
		logits := linearRow(m.ConceptHead, encFinal)
		per := cfg.NumConcepts + 1
		for s := 0; s < cfg.ConceptLen; s++ {
			codes[s] = argmaxF(logits[s*per : (s+1)*per])
		}
		if opts.ShuffleConcept {
			for s := 0; s < cfg.ConceptLen; s++ {
				if codes[s] >= 1 {
					codes[s] = codes[s]%cfg.NumConcepts + 1
				}
			}
		}
	}
	out.ConceptCodes = codes

	// ---- Seed.
	seed := make([]float32, hidden)
	copy(seed, encFinal)
	if out.Intent >= 0 {
		m.addEmbRow(seed, m.IntentEmb, out.Intent)
	}
	if out.Tone >= 0 {
		m.addEmbRow(seed, m.ToneEmb, out.Tone)
	}
	if m.layerOn("turn", opts) {
		if opts.PrevReplyIDs != nil {
			prev := m.DeriveReplyCodes(opts.PrevReplyIDs)
			for s := 0; s < cfg.ConceptLen; s++ {
				m.addEmbRowScaled(seed, m.TurnEmb, prev[s], 1.0/float32(cfg.ConceptLen))
			}
		} else {
			for k := 0; k < hidden; k++ {
				seed[k] += m.NoneTurn.Data[k]
			}
		}
	}
	// Per-step LSTM state is [1, hidden] (the shape the decoder's
	// own ForwardCopy path carries); encCell from the 3D encoder
	// pass already has exactly that shape at batch 1.
	h := tensor.NewTensor([]int{1, hidden}, seed, false)
	c := encCell

	// ---- Decoder: concept prefix steps first (outputs discarded).
	if m.layerOn("concept", opts) {
		for s := 0; s < cfg.ConceptLen; s++ {
			off := codes[s] * embDim
			emb := append([]float32{}, m.ConceptEmb.Weight.Data[off:off+embDim]...)
			_, _, nh, nc, err := m.decStep(emb, h, c)
			if err != nil {
				return out, err
			}
			h, c = nh, nc
		}
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

	// ---- Greedy word loop with the production bans.
	prevID := -1
	usedBigrams := map[[2]int]bool{}
	fedID := vocab.BosID
	fedSurface := ""
	for step := 0; step < maxLen; step++ {
		emb := m.wordStepEmbedding(fedID, fedSurface)
		logits, moeRow, nh, nc, err := m.decStep(emb, h, c)
		if err != nil {
			return out, err
		}
		h, c = nh, nc
		var nextID int
		if m.layerOn("copy", opts) {
			probs, _ := base.Decoder.PtrGen.ptrGenStep(moeRow, encStates, inputIDs, validLen, logits)
			nextID = argmaxBannedProbs(probs, prevID, usedBigrams, out.TokenIDs)
		} else {
			nextID = argmaxBannedLogits(logits, prevID, usedBigrams, out.TokenIDs, step)
		}
		if (nextID == vocab.EosID && len(out.TokenIDs) > 0) || nextID == vocab.PaddingTokenID {
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
		out.TokenIDs = append(out.TokenIDs, nextID)
		if prevID >= 0 {
			usedBigrams[[2]int{prevID, nextID}] = true
		}
		prevID = nextID
		fedID = nextID
		fedSurface = vocab.GetWord(nextID)
	}
	out.Text = vocab.Decode(out.TokenIDs)
	if cfg.Concept {
		out.ReplyCodes = m.DeriveReplyCodes(out.TokenIDs)
	}
	return out, nil
}

// decStep runs one decoder step on a prepared embedding vector:
// LSTM -> MoE -> Output, returning vocab logits and the MoE row
// (the copy layer's decoder state), plus the new LSTM state.
func (m *LayersModel) decStep(emb []float32, h, c *tensor.Tensor) (logits, moeRow []float32, nh, nc *tensor.Tensor, err error) {
	hidden := m.Config.HiddenDim
	embT := tensor.NewTensor([]int{1, len(emb)}, emb, false)
	outT, newC, err := m.Base.Decoder.LSTM.Forward(embT, h, c)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	out3, err := outT.Reshape([]int{1, 1, hidden})
	if err != nil {
		return nil, nil, nil, nil, err
	}
	moeOut, _, err := m.Base.Decoder.MoE.Forward(out3)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	lg, err := m.Base.Decoder.Output.Forward(moeOut)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	moeRow = append([]float32{}, moeOut.Data[:hidden]...)
	return append([]float32{}, lg.Data...), moeRow, outT, newC, nil
}

// wordStepEmbedding builds the decoder's input vector for one fed
// token: token row + class row + char mean (layers on only).
func (m *LayersModel) wordStepEmbedding(id int, surface string) []float32 {
	embDim := m.Config.EmbedDim
	emb := make([]float32, embDim)
	off := id * embDim
	copy(emb, m.Base.Decoder.Embedding.Weight.Data[off:off+embDim])
	if m.Config.Class && m.ClassTable != nil {
		coff := m.classOf(id) * embDim
		for k := 0; k < embDim; k++ {
			emb[k] += m.ClassTable.Weight.Data[coff+k]
		}
	}
	if m.Config.Char && m.CharTable != nil {
		tmp := make([]float32, embDim)
		m.charVectorInto(surface, m.isSpecialID(id), tmp)
		for k := 0; k < embDim; k++ {
			emb[k] += tmp[k]
		}
	}
	return emb
}

// argmaxBannedLogits mirrors PredictWithTrace's greedy bans:
// max-logit token that is not the previous token and does not
// repeat a bigram already produced.
func argmaxBannedLogits(logits []float32, prevID int, used map[[2]int]bool, generated []int, step int) int {
	best, bestV := 0, float32(-1e30)
	for i, v := range logits {
		if i == prevID {
			continue
		}
		if len(generated) > 0 && used[[2]int{generated[len(generated)-1], i}] {
			continue
		}
		if v > bestV {
			bestV, best = v, i
		}
	}
	return best
}

// argmaxBannedProbs applies the same bans in probability space.
func argmaxBannedProbs(probs []float32, prevID int, used map[[2]int]bool, generated []int) int {
	best, bestV := 0, float32(-1)
	for i, v := range probs {
		if i == prevID {
			continue
		}
		if len(generated) > 0 && used[[2]int{generated[len(generated)-1], i}] {
			continue
		}
		if v > bestV {
			bestV, best = v, i
		}
	}
	return best
}

func argmaxF(xs []float32) int {
	best, bestV := 0, float32(-1e30)
	for i, v := range xs {
		if v > bestV {
			bestV, best = v, i
		}
	}
	return best
}
