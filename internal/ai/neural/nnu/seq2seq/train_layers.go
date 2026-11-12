package seq2seq

// Layers pilot: the training step for LayersModel. The gradient
// walk mirrors TrainBatch exactly (manual layer.Backward calls in
// the production order), extended with the layers' streams:
//
//	combined embeddings (token + class + char, built by values):
//	  the LSTM consumes the combined tensor; after LSTM.Backward
//	  its .Grad is handed to the base embedding's Backward AND
//	  scattered by hand into the class/char tables.
//	seed conditioning (intent/tone/turn): the decoder seed is a
//	  values tensor; GetPrevHiddenGrad's output is scattered into
//	  the seed embeddings and copied on as the encoder-final grad.
//	auxiliary heads (intent/tone/concept): value-space CE; each
//	  head's Backward runs on its own clone of the encoder-final
//	  state so the three gradient contributions sum by hand.
//	concept prefix: the decoder input's first steps are concept
//	  embedding rows; their slice of the input grad is scattered
//	  into the concept table.
//	copy: the mixture NLL replaces the word CE (TrainBatchPtrGen
//	  precedent), its three gradient streams joined at moeOut,
//	  the decoder embeddings, and the encoder output.
//
// The shared-graph rule from the NCM build holds: tensor.Backward
// is never called here; only the layers' own Backward methods.

import (
	"fmt"
	"math"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// LayersBatch is one training batch for the layers trainer. All
// rows are padded to the batch's max input/target length with
// PadID; surfaces align with InputIDs/decoder-fed target ids.
type LayersBatch struct {
	InputIDs      [][]int
	InputSurfaces [][]string // aligned with InputIDs; "" on padding
	TargetIDs     [][]int    // [BOS, reply..., EOS, pad...]
	TargetSurfs   [][]string // aligned with TargetIDs (decoder-fed positions)
	IntentLabels  []int      // valid when intent on
	ToneLabels    []int      // valid when tone on
	ConceptCodes  [][3]int   // gold reply codes, valid when concept on
	PrevCodes     [][3]int   // previous reply's codes, valid when turn on
	HasPrev       []bool
	PadID         int
}

// TrainBatchLayers runs one forward/backward over the batch and
// returns the total loss (word CE or copy mixture NLL, plus the
// MoE aux and the weighted auxiliary head losses). The caller owns
// the optimizer (ZeroGrad before, Step after), exactly as with
// TrainBatch.
func (m *LayersModel) TrainBatchLayers(batch LayersBatch) (float32, error) {
	cfg := m.Config
	base := m.Base
	B := len(batch.InputIDs)
	if B == 0 {
		return 0, fmt.Errorf("layers TrainBatch: empty batch")
	}
	encLen := len(batch.InputIDs[0])
	decLen := len(batch.TargetIDs[0])
	if decLen < 2 {
		return 0, fmt.Errorf("layers TrainBatch: target length %d < 2", decLen)
	}
	steps := decLen - 1
	prefix := 0
	if cfg.Concept {
		prefix = cfg.ConceptLen
	}
	totalSteps := prefix + steps
	vocabSize := base.OutputVocab.Size()
	hidden := cfg.HiddenDim
	embDim := cfg.EmbedDim
	padID := batch.PadID
	m.ensureManualGrads()

	// ---- Encoder: base embedding forward (populates its backward
	// state), then the combined tensor the LSTM actually consumes.
	encTok, err := base.Encoder.Embedding.Forward(idsToTensor(batch.InputIDs))
	if err != nil {
		return 0, fmt.Errorf("layers encoder embedding: %w", err)
	}
	encComb := tensor.NewTensor([]int{B, encLen, embDim},
		m.combineRows(encTok.Data, batch.InputIDs, batch.InputSurfaces, B, encLen), true)
	h0 := tensor.NewTensor([]int{B, hidden}, make([]float32, B*hidden), false)
	c0 := tensor.NewTensor([]int{B, hidden}, make([]float32, B*hidden), false)
	encOut, encCell, err := base.Encoder.LSTM.Forward(encComb, h0, c0)
	if err != nil {
		return 0, fmt.Errorf("layers encoder lstm: %w", err)
	}
	lastSlice, err := encOut.Slice(1, encLen-1, encLen)
	if err != nil {
		return 0, fmt.Errorf("layers encoder final slice: %w", err)
	}
	encFinalH, err := lastSlice.Reshape([]int{B, hidden})
	if err != nil {
		return 0, fmt.Errorf("layers encoder final reshape: %w", err)
	}

	// ---- Auxiliary heads on the encoder final state (values CE).
	var intentLoss, toneLoss, conceptLoss float32
	var dIntent, dTone, dConcept []float32 // dLogits per head, for backward
	if cfg.Intent {
		intentLoss, dIntent = m.headCE(m.IntentHead, encFinalH.Data, batch.IntentLabels, B)
	}
	if cfg.Tone {
		toneLoss, dTone = m.headCE(m.ToneHead, encFinalH.Data, batch.ToneLabels, B)
	}
	if cfg.Concept {
		conceptLoss, dConcept = m.conceptHeadCE(encFinalH.Data, batch.ConceptCodes, B)
	}

	// ---- Decoder seed: encoder final + gold conditioning rows.
	seedData := make([]float32, B*hidden)
	copy(seedData, encFinalH.Data)
	if cfg.Intent {
		for b := 0; b < B; b++ {
			m.addEmbRow(seedData[b*hidden:], m.IntentEmb, batch.IntentLabels[b])
		}
	}
	if cfg.Tone {
		for b := 0; b < B; b++ {
			m.addEmbRow(seedData[b*hidden:], m.ToneEmb, batch.ToneLabels[b])
		}
	}
	if cfg.Turn {
		for b := 0; b < B; b++ {
			if batch.HasPrev[b] {
				for s := 0; s < cfg.ConceptLen; s++ {
					m.addEmbRowScaled(seedData[b*hidden:], m.TurnEmb, batch.PrevCodes[b][s], 1.0/float32(cfg.ConceptLen))
				}
			} else {
				row := seedData[b*hidden:]
				for k := 0; k < hidden; k++ {
					row[k] += m.NoneTurn.Data[k]
				}
			}
		}
	}
	// RequiresGrad must be set: the LSTM records the prev-hidden
	// gradient (the seed's gradient, scattered into the
	// conditioning tables below) only for tracked inputs — in
	// production the seed is a slice of the encoder output, which
	// is tracked; here it is a fresh values tensor.
	seedH := tensor.NewTensor([]int{B, hidden}, seedData, true)

	// ---- Decoder inputs: [concept prefix steps] ++ [word steps].
	decInputIDs := make([][]int, B)
	for b := range batch.TargetIDs {
		decInputIDs[b] = batch.TargetIDs[b][:steps]
	}
	decTok, err := base.Decoder.Embedding.Forward(idsToTensor(decInputIDs))
	if err != nil {
		return 0, fmt.Errorf("layers decoder embedding: %w", err)
	}
	wordComb := m.combineRows(decTok.Data, decInputIDs, targetSurfacesFed(batch), B, steps)
	decCombData := make([]float32, B*totalSteps*embDim)
	for b := 0; b < B; b++ {
		dst := decCombData[b*totalSteps*embDim:]
		if cfg.Concept {
			for s := 0; s < prefix; s++ {
				off := batch.ConceptCodes[b][s] * embDim
				copy(dst[s*embDim:(s+1)*embDim], m.ConceptEmb.Weight.Data[off:off+embDim])
			}
		}
		copy(dst[prefix*embDim:], wordComb[b*steps*embDim:(b+1)*steps*embDim])
	}
	decComb := tensor.NewTensor([]int{B, totalSteps, embDim}, decCombData, true)
	decOut, _, err := base.Decoder.LSTM.Forward(decComb, seedH, encCell)
	if err != nil {
		return 0, fmt.Errorf("layers decoder lstm: %w", err)
	}
	moeOut, moeAux, err := base.Decoder.MoE.Forward(decOut)
	if err != nil {
		return 0, fmt.Errorf("layers decoder moe: %w", err)
	}
	logits, err := base.Decoder.Output.Forward(moeOut)
	if err != nil {
		return 0, fmt.Errorf("layers decoder output: %w", err)
	}

	// ---- Word loss: mixture NLL when copy is on, else CE over the
	// full step axis with prefix positions ignored.
	var wordLoss float32
	var dLogits *tensor.Tensor
	var dMoeCopy []float32   // [B*totalSteps*hidden], word positions only
	var dEncCopy []float32 // [B*encLen*hidden]
	if cfg.Copy {
		pg := base.Decoder.PtrGen
		dLogits = tensor.NewTensor([]int{B, totalSteps, vocabSize}, make([]float32, B*totalSteps*vocabSize), false)
		dMoeCopy = make([]float32, B*totalSteps*hidden)
		dEncCopy = make([]float32, B*encLen*hidden)
		encStates := make([][][]float32, B)
		validLens := make([]int, B)
		for b := 0; b < B; b++ {
			encStates[b] = make([][]float32, encLen)
			for t := 0; t < encLen; t++ {
				off := (b*encLen + t) * hidden
				encStates[b][t] = encOut.Data[off : off+hidden]
				if batch.InputIDs[b][t] != padID {
					validLens[b] = t + 1
				}
			}
		}
		var sum float32
		var count int
		for b := 0; b < B; b++ {
			for t := 0; t < steps; t++ {
				pos := prefix + t
				mOff := (b*totalSteps + pos) * hidden
				lOff := (b*totalSteps + pos) * vocabSize
				final, cache := pg.ptrGenStep(moeOut.Data[mOff:mOff+hidden], encStates[b], batch.InputIDs[b], validLens[b], logits.Data[lOff:lOff+vocabSize])
				label := batch.TargetIDs[b][t+1]
				if label == padID {
					continue
				}
				p := final[label]
				if p < 1e-9 {
					p = 1e-9
				}
				sum += -float32(math.Log(float64(p)))
				count++
				dFinal := -1 / p
				dL, dDec, dEnc := pg.ptrGenBackward(cache, encStates[b], batch.InputIDs[b], label, dFinal)
				for v := 0; v < vocabSize; v++ {
					dLogits.Data[lOff+v] += dL[v]
				}
				for h := 0; h < hidden; h++ {
					dMoeCopy[mOff+h] += dDec[h]
				}
				// dEnc covers only the valid (non-pad) positions:
				// ptrGenBackward sizes it by validLen.
				for tt := 0; tt < len(dEnc); tt++ {
					eOff := (b*encLen + tt) * hidden
					for h := 0; h < hidden; h++ {
						dEncCopy[eOff+h] += dEnc[tt][h]
					}
				}
			}
		}
		if count > 0 {
			wordLoss = sum / float32(count)
			inv := 1 / float32(count)
			for i := range dLogits.Data {
				dLogits.Data[i] *= inv
			}
			for i := range dMoeCopy {
				dMoeCopy[i] *= inv
			}
			for i := range dEncCopy {
				dEncCopy[i] *= inv
			}
		}
	} else {
		labels := make([][]int, B)
		for b := 0; b < B; b++ {
			labels[b] = make([]int, totalSteps)
			for i := 0; i < prefix; i++ {
				labels[b][i] = padID
			}
			copy(labels[b][prefix:], batch.TargetIDs[b][1:])
		}
		var err error
		wordLoss, dLogits, err = softmaxCrossEntropy(logits, labels, padID, vocabSize)
		if err != nil {
			return 0, err
		}
	}
	loss := wordLoss + moeAuxCoeff*moeAux +
		cfg.IntentWeight*intentLoss + cfg.ToneWeight*toneLoss + cfg.ConceptWt*conceptLoss

	// ---- Backward: output -> MoE -> decoder LSTM.
	if err := base.Decoder.Output.Backward(dLogits); err != nil {
		return 0, fmt.Errorf("layers output backward: %w", err)
	}
	if moeOut.Grad == nil {
		return 0, fmt.Errorf("layers: moe grad missing")
	}
	if dMoeCopy != nil {
		for i, g := range dMoeCopy {
			moeOut.Grad.Data[i] += g
		}
	}
	dDecOut, err := base.Decoder.MoE.Backward(moeOut.Grad, moeAuxCoeff)
	if err != nil {
		return 0, fmt.Errorf("layers moe backward: %w", err)
	}
	zeroCell := tensor.NewTensor([]int{B, hidden}, make([]float32, B*hidden), false)
	if err := base.Decoder.LSTM.Backward(dDecOut, zeroCell); err != nil {
		return 0, fmt.Errorf("layers decoder lstm backward: %w", err)
	}
	dSeed := base.Decoder.LSTM.GetPrevHiddenGrad()
	dInitC := base.Decoder.LSTM.GetPrevCellGrad()
	if dSeed == nil || dInitC == nil {
		return 0, fmt.Errorf("layers: decoder init grads missing")
	}
	if decComb.Grad == nil {
		return 0, fmt.Errorf("layers: decoder input grad missing")
	}

	// Decoder input grad: prefix slice -> concept table; word slice
	// -> decoder embedding + class/char scatters.
	wordGradData := make([]float32, B*steps*embDim)
	for b := 0; b < B; b++ {
		src := decComb.Grad.Data[b*totalSteps*embDim:]
		if cfg.Concept {
			for s := 0; s < prefix; s++ {
				m.scatterEmb(m.ConceptEmb, batch.ConceptCodes[b][s], src[s*embDim:(s+1)*embDim], 1)
			}
		}
		copy(wordGradData[b*steps*embDim:], src[prefix*embDim:(prefix+steps)*embDim])
	}
	wordGrad := tensor.NewTensor([]int{B, steps, embDim}, wordGradData, false)
	if err := base.Decoder.Embedding.Backward(wordGrad); err != nil {
		return 0, fmt.Errorf("layers decoder embedding backward: %w", err)
	}
	m.scatterCombined(wordGradData, decInputIDs, targetSurfacesFed(batch), B, steps)

	// Seed grad: scatter into the conditioning tables, then the
	// remainder is the encoder-final grad seed.
	dEncFinal := make([]float32, B*hidden)
	copy(dEncFinal, dSeed.Data)
	for b := 0; b < B; b++ {
		row := dSeed.Data[b*hidden : (b+1)*hidden]
		if cfg.Intent {
			m.scatterEmb(m.IntentEmb, batch.IntentLabels[b], row, 1)
		}
		if cfg.Tone {
			m.scatterEmb(m.ToneEmb, batch.ToneLabels[b], row, 1)
		}
		if cfg.Turn {
			if batch.HasPrev[b] {
				for s := 0; s < cfg.ConceptLen; s++ {
					m.scatterEmb(m.TurnEmb, batch.PrevCodes[b][s], row, 1.0/float32(cfg.ConceptLen))
				}
			} else {
				for k := 0; k < hidden; k++ {
					m.NoneTurn.Grad.Data[k] += row[k]
				}
			}
		}
	}
	// Head grads join at the encoder final state.
	if cfg.Intent {
		if err := m.headBackward(m.IntentHead, encFinalH.Data, dIntent, B, dEncFinal); err != nil {
			return 0, err
		}
	}
	if cfg.Tone {
		if err := m.headBackward(m.ToneHead, encFinalH.Data, dTone, B, dEncFinal); err != nil {
			return 0, err
		}
	}
	if cfg.Concept {
		if err := m.conceptHeadBackward(encFinalH.Data, dConcept, B, dEncFinal); err != nil {
			return 0, err
		}
	}

	// ---- Across the boundary into the encoder.
	encGradData := make([]float32, B*encLen*hidden)
	if dEncCopy != nil {
		copy(encGradData, dEncCopy)
	}
	for b := 0; b < B; b++ {
		dst := (b*encLen + (encLen - 1)) * hidden
		for k := 0; k < hidden; k++ {
			encGradData[dst+k] += dEncFinal[b*hidden+k]
		}
	}
	encGradH := tensor.NewTensor([]int{B, encLen, hidden}, encGradData, false)
	if err := base.Encoder.LSTM.Backward(encGradH, dInitC); err != nil {
		return 0, fmt.Errorf("layers encoder lstm backward: %w", err)
	}
	if encComb.Grad == nil {
		return 0, fmt.Errorf("layers: encoder input grad missing")
	}
	if err := base.Encoder.Embedding.Backward(encComb.Grad); err != nil {
		return 0, fmt.Errorf("layers encoder embedding backward: %w", err)
	}
	m.scatterCombined(encComb.Grad.Data, batch.InputIDs, batch.InputSurfaces, B, encLen)
	return loss, nil
}

// combineRows adds the class and char contributions to base
// embedding values, returning a fresh [B*T*embDim] slice. Positions
// whose id is the pad token keep the base values untouched (they
// are masked by the loss; class/char of padding is class 0 anyway).
func (m *LayersModel) combineRows(baseData []float32, ids [][]int, surfaces [][]string, B, T int) []float32 {
	embDim := m.Config.EmbedDim
	out := make([]float32, B*T*embDim)
	copy(out, baseData)
	if !m.Config.Class && !m.Config.Char {
		return out
	}
	tmp := make([]float32, embDim)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			id := ids[b][t]
			off := (b*T + t) * embDim
			if m.Config.Class && m.ClassTable != nil {
				c := m.classOf(id)
				coff := c * embDim
				for k := 0; k < embDim; k++ {
					out[off+k] += m.ClassTable.Weight.Data[coff+k]
				}
			}
			if m.Config.Char && m.CharTable != nil {
				surf := ""
				if surfaces != nil && t < len(surfaces[b]) {
					surf = surfaces[b][t]
				}
				m.charVectorInto(surf, m.isSpecialID(id), tmp)
				for k := 0; k < embDim; k++ {
					out[off+k] += tmp[k]
				}
			}
		}
	}
	return out
}

// scatterCombined distributes a combined-embedding gradient into
// the class and char tables (the base embedding's own Backward
// consumes the same grad tensor separately).
func (m *LayersModel) scatterCombined(grad []float32, ids [][]int, surfaces [][]string, B, T int) {
	embDim := m.Config.EmbedDim
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			id := ids[b][t]
			off := (b*T + t) * embDim
			row := grad[off : off+embDim]
			if m.Config.Class && m.ClassTable != nil {
				m.scatterEmb(m.ClassTable, m.classOf(id), row, 1)
			}
			if m.Config.Char && m.CharTable != nil && !m.isSpecialID(id) {
				surf := ""
				if surfaces != nil && t < len(surfaces[b]) {
					surf = surfaces[b][t]
				}
				if surf == "" {
					continue
				}
				bs := CharBucketsFor(surf, m.Config.CharBuckets)
				inv := 1 / float32(len(bs))
				for _, bk := range bs {
					m.scatterEmb(m.CharTable, bk, row, inv)
				}
			}
		}
	}
}

// addEmbRow adds embedding row idx of emb into dst (length = row width).
func (m *LayersModel) addEmbRow(dst []float32, emb *nn.Embedding, idx int) {
	m.addEmbRowScaled(dst, emb, idx, 1)
}

func (m *LayersModel) addEmbRowScaled(dst []float32, emb *nn.Embedding, idx int, scale float32) {
	// The row width comes from the embedding's own shape, never
	// from len(dst): callers pass row slices of a larger buffer.
	dim := emb.Weight.Shape[1]
	off := idx * dim
	for k := 0; k < dim; k++ {
		dst[k] += scale * emb.Weight.Data[off+k]
	}
}

// scatterEmb adds scale*row into emb's gradient at row idx. The
// row width comes from the embedding's own shape (see
// addEmbRowScaled); callers pass exactly one row of gradient.
func (m *LayersModel) scatterEmb(emb *nn.Embedding, idx int, row []float32, scale float32) {
	if emb == nil || emb.Weight == nil {
		return
	}
	if emb.Weight.Grad == nil {
		emb.Weight.Grad = tensor.NewTensor(emb.Weight.Shape, make([]float32, len(emb.Weight.Data)), false)
	}
	dim := emb.Weight.Shape[1]
	off := idx * dim
	for k := 0; k < dim; k++ {
		emb.Weight.Grad.Data[off+k] += scale * row[k]
	}
}

// headCE computes mean softmax CE of a linear head over rows of x
// ([B*hidden]) against labels, returning the loss and dLogits
// ([B*out], scaled by 1/B) for headBackward.
func (m *LayersModel) headCE(head *nn.Linear, x []float32, labels []int, B int) (float32, []float32) {
	hidden := m.Config.HiddenDim
	out := len(head.Biases.Data)
	dLogits := make([]float32, B*out)
	var sum float64
	for b := 0; b < B; b++ {
		row := x[b*hidden : (b+1)*hidden]
		logits := linearRow(head, row)
		mx := float32(math.Inf(-1))
		for _, v := range logits {
			if v > mx {
				mx = v
			}
		}
		var denom float64
		probs := make([]float32, out)
		for o := 0; o < out; o++ {
			probs[o] = float32(math.Exp(float64(logits[o] - mx)))
			denom += float64(probs[o])
		}
		for o := 0; o < out; o++ {
			probs[o] /= float32(denom)
		}
		y := labels[b]
		p := probs[y]
		if p < 1e-9 {
			p = 1e-9
		}
		sum += -math.Log(float64(p))
		for o := 0; o < out; o++ {
			g := probs[o]
			if o == y {
				g--
			}
			dLogits[b*out+o] = g / float32(B)
		}
	}
	return float32(sum / float64(B)), dLogits
}

// conceptHeadCE is headCE over the concept head's ConceptLen
// positions: mean CE over B*ConceptLen code predictions.
func (m *LayersModel) conceptHeadCE(x []float32, codes [][3]int, B int) (float32, []float32) {
	cfg := m.Config
	hidden := cfg.HiddenDim
	per := cfg.NumConcepts + 1
	out := cfg.ConceptLen * per
	dLogits := make([]float32, B*out)
	var sum float64
	n := float64(B * cfg.ConceptLen)
	for b := 0; b < B; b++ {
		row := x[b*hidden : (b+1)*hidden]
		logits := linearRow(m.ConceptHead, row)
		for s := 0; s < cfg.ConceptLen; s++ {
			seg := logits[s*per : (s+1)*per]
			mx := float32(math.Inf(-1))
			for _, v := range seg {
				if v > mx {
					mx = v
				}
			}
			var denom float64
			probs := make([]float32, per)
			for o := 0; o < per; o++ {
				probs[o] = float32(math.Exp(float64(seg[o] - mx)))
				denom += float64(probs[o])
			}
			for o := 0; o < per; o++ {
				probs[o] /= float32(denom)
			}
			y := codes[b][s]
			p := probs[y]
			if p < 1e-9 {
				p = 1e-9
			}
			sum += -math.Log(float64(p))
			for o := 0; o < per; o++ {
				g := probs[o]
				if o == y {
					g--
				}
				dLogits[b*out+s*per+o] = g / float32(n)
			}
		}
	}
	return float32(sum / n), dLogits
}

// headBackward runs a head's own Backward on a fresh clone of x and
// adds the resulting input grad into dX ([B*hidden]).
func (m *LayersModel) headBackward(head *nn.Linear, x []float32, dLogits []float32, B int, dX []float32) error {
	hidden := m.Config.HiddenDim
	out := len(head.Biases.Data)
	cloneData := make([]float32, B*hidden)
	copy(cloneData, x)
	clone := tensor.NewTensor([]int{B, hidden}, cloneData, true)
	if _, err := head.Forward(clone); err != nil {
		return fmt.Errorf("layers head backward forward: %w", err)
	}
	dT := tensor.NewTensor([]int{B, out}, dLogits, false)
	if err := head.Backward(dT); err != nil {
		return fmt.Errorf("layers head backward: %w", err)
	}
	if clone.Grad == nil {
		return fmt.Errorf("layers: head input grad missing")
	}
	for i := range dX {
		dX[i] += clone.Grad.Data[i]
	}
	return nil
}

// conceptHeadBackward is headBackward for the concept head.
func (m *LayersModel) conceptHeadBackward(x []float32, dLogits []float32, B int, dX []float32) error {
	cfg := m.Config
	hidden := cfg.HiddenDim
	out := cfg.ConceptLen * (cfg.NumConcepts + 1)
	cloneData := make([]float32, B*hidden)
	copy(cloneData, x)
	clone := tensor.NewTensor([]int{B, hidden}, cloneData, true)
	if _, err := m.ConceptHead.Forward(clone); err != nil {
		return fmt.Errorf("layers concept head backward forward: %w", err)
	}
	dT := tensor.NewTensor([]int{B, out}, dLogits, false)
	if err := m.ConceptHead.Backward(dT); err != nil {
		return fmt.Errorf("layers concept head backward: %w", err)
	}
	if clone.Grad == nil {
		return fmt.Errorf("layers: concept head input grad missing")
	}
	for i := range dX {
		dX[i] += clone.Grad.Data[i]
	}
	return nil
}

// linearRow evaluates a Linear head on one row by values:
// out[o] = sum_i x[i]*W[i*out+o] + Bias[o] (Weights are [in, out]).
func linearRow(head *nn.Linear, x []float32) []float32 {
	out := len(head.Biases.Data)
	res := make([]float32, out)
	copy(res, head.Biases.Data)
	for i, xv := range x {
		if xv == 0 {
			continue
		}
		off := i * out
		for o := 0; o < out; o++ {
			res[o] += xv * head.Weights.Data[off+o]
		}
	}
	return res
}

// targetSurfacesFed returns the surfaces of the decoder-fed target
// positions (TargetIDs minus the final position).
func targetSurfacesFed(batch LayersBatch) [][]string {
	out := make([][]string, len(batch.TargetSurfs))
	for i, row := range batch.TargetSurfs {
		if len(row) > 0 {
			out[i] = row[:len(row)-1]
		}
	}
	return out
}
