package seq2seq

// BPTT training for the pointer-generator model.
//
// TrainBatchPtrGen mirrors TrainBatch's structure exactly — single
// 3D encoder and decoder LSTM passes so the BPTT caches populate,
// then a manual walk back — with three changes:
//
//  1. No MoE: the decoder state feeds the output projection directly.
//  2. The loss is the negative log of the MIXTURE probability of each
//     target token (see ptrgen.go), not a softmax cross-entropy over
//     logits. The mixture's backward is closed-form per step and
//     produces three gradient streams: into the logits (routed through
//     the output projection as usual), into the decoder state, and
//     into every encoder state (through the attention and the gate's
//     context term).
//  3. The encoder receives gradient at EVERY position (attention),
//     not only at the final step that seeds the decoder.
//
// The caller owns the optimizer, exactly as with TrainBatch.

import (
	"fmt"
	"math"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// TrainBatchPtrGen runs one teacher-forcing forward/backward pass
// over a batch and accumulates gradients into the model's parameters
// (which must include the PointerGen head — see Decoder.Parameters).
// Returns the mean mixture NLL per non-padding target token.
func TrainBatchPtrGen(m *Seq2Seq, inputIDs, targetIDs [][]int, padID int) (float32, error) {
	if m.Decoder == nil || m.Decoder.PtrGen == nil {
		return 0, fmt.Errorf("seq2seq TrainBatchPtrGen: model has no pointer-generator head")
	}
	pg := m.Decoder.PtrGen
	batch := len(inputIDs)
	if batch == 0 {
		return 0, fmt.Errorf("seq2seq TrainBatchPtrGen: empty batch")
	}
	encLen := len(inputIDs[0])
	decLen := len(targetIDs[0])
	if decLen < 2 {
		return 0, fmt.Errorf("seq2seq TrainBatchPtrGen: target length %d < 2 (need BOS ... EOS)", decLen)
	}
	steps := decLen - 1
	vocabSize := m.OutputVocab.Size()
	hiddenSize := m.HiddenDim

	// ---- Encoder: single 3D pass (BPTT caches populated). ----
	encEmb, err := m.Encoder.Embedding.Forward(idsToTensor(inputIDs))
	if err != nil {
		return 0, fmt.Errorf("encoder embedding forward: %w", err)
	}
	h0 := tensor.NewTensor([]int{batch, hiddenSize}, make([]float32, batch*hiddenSize), false)
	c0 := tensor.NewTensor([]int{batch, hiddenSize}, make([]float32, batch*hiddenSize), false)
	encOut, encCell, err := m.Encoder.LSTM.Forward(encEmb, h0, c0)
	if err != nil {
		return 0, fmt.Errorf("encoder lstm forward: %w", err)
	}
	lastSlice, err := encOut.Slice(1, encLen-1, encLen)
	if err != nil {
		return 0, fmt.Errorf("encoder final hidden slice: %w", err)
	}
	encFinalH, err := lastSlice.Reshape([]int{batch, hiddenSize})
	if err != nil {
		return 0, fmt.Errorf("encoder final hidden reshape: %w", err)
	}

	// ---- Decoder: teacher forcing in one 3D pass. ----
	decInputIDs := make([][]int, batch)
	labels := make([][]int, batch)
	for b := range targetIDs {
		decInputIDs[b] = targetIDs[b][:steps]
		labels[b] = targetIDs[b][1:]
	}
	decEmb, err := m.Decoder.Embedding.Forward(idsToTensor(decInputIDs))
	if err != nil {
		return 0, fmt.Errorf("decoder embedding forward: %w", err)
	}
	decOut, _, err := m.Decoder.LSTM.Forward(decEmb, encFinalH, encCell)
	if err != nil {
		return 0, fmt.Errorf("decoder lstm forward: %w", err)
	}
	logits, err := m.Decoder.Output.Forward(decOut) // [B, steps, V]
	if err != nil {
		return 0, fmt.Errorf("decoder output forward: %w", err)
	}

	// Per-row valid encoder lengths (padding is trailing).
	validLen := make([]int, batch)
	for b := 0; b < batch; b++ {
		n := 0
		for n < encLen && inputIDs[b][n] != padID {
			n++
		}
		validLen[b] = n
	}
	// Encoder states as per-position views into encOut.
	encStates := make([][][]float32, batch)
	for b := 0; b < batch; b++ {
		encStates[b] = make([][]float32, encLen)
		for i := 0; i < encLen; i++ {
			off := (b*encLen + i) * hiddenSize
			encStates[b][i] = encOut.Data[off : off+hiddenSize]
		}
	}

	// ---- Mixture forward: cache every step, count valid labels. ----
	type stepRef struct {
		cache *ptrGenCache
		final []float32
	}
	stepRefs := make([][]stepRef, batch)
	valid := 0
	for b := 0; b < batch; b++ {
		stepRefs[b] = make([]stepRef, steps)
		for t := 0; t < steps; t++ {
			if labels[b][t] != padID {
				valid++
			}
			dOff := (b*steps + t) * hiddenSize
			lOff := (b*steps + t) * vocabSize
			final, cache := pg.ptrGenStep(decOut.Data[dOff:dOff+hiddenSize], encStates[b], inputIDs[b], validLen[b], logits.Data[lOff:lOff+vocabSize])
			stepRefs[b][t] = stepRef{cache: cache, final: final}
		}
	}
	if valid == 0 {
		return 0, fmt.Errorf("seq2seq TrainBatchPtrGen: no valid (non-pad) target tokens in batch")
	}
	scale := float32(1) / float32(valid)

	// ---- Mixture backward (pure slice math; no layer forwards, so
	// the LSTM BPTT caches stay valid). Produces the three gradient
	// streams described in the file header. ----
	dLogits := tensor.NewTensor([]int{batch, steps, vocabSize}, make([]float32, batch*steps*vocabSize), false)
	dDecOut := tensor.NewTensor([]int{batch, steps, hiddenSize}, make([]float32, batch*steps*hiddenSize), false)
	dEncOut := tensor.NewTensor([]int{batch, encLen, hiddenSize}, make([]float32, batch*encLen*hiddenSize), false)
	var totalLoss float64
	for b := 0; b < batch; b++ {
		for t := 0; t < steps; t++ {
			label := labels[b][t]
			if label == padID {
				continue
			}
			if label < 0 || label >= vocabSize {
				return 0, fmt.Errorf("label %d out of vocab range [0,%d)", label, vocabSize)
			}
			sr := stepRefs[b][t]
			p := float64(sr.final[label])
			if p < ptrGenProbFloor {
				p = ptrGenProbFloor
			}
			totalLoss += -math.Log(p)
			dFinal := -scale / float32(p)
			dl, dDec, dEnc := pg.ptrGenBackward(sr.cache, encStates[b], inputIDs[b], label, dFinal)
			lOff := (b*steps + t) * vocabSize
			copy(dLogits.Data[lOff:lOff+vocabSize], dl)
			dOff := (b*steps + t) * hiddenSize
			for h := 0; h < hiddenSize; h++ {
				dDecOut.Data[dOff+h] += dDec[h]
			}
			for i := 0; i < validLen[b]; i++ {
				eOff := (b*encLen + i) * hiddenSize
				for h := 0; h < hiddenSize; h++ {
					dEncOut.Data[eOff+h] += dEnc[i][h]
				}
			}
		}
	}
	loss := float32(totalLoss / float64(valid))

	// ---- Backward through the layers: output -> decoder LSTM ->
	// decoder embedding, then across the boundary into the encoder. ----
	if err := m.Decoder.Output.Backward(dLogits); err != nil {
		return 0, fmt.Errorf("decoder output backward: %w", err)
	}
	if decOut.Grad == nil {
		return 0, fmt.Errorf("decoder state grad missing after output backward")
	}
	for i, g := range dDecOut.Data {
		decOut.Grad.Data[i] += g
	}
	zeroCell := tensor.NewTensor([]int{batch, hiddenSize}, make([]float32, batch*hiddenSize), false)
	if err := m.Decoder.LSTM.Backward(decOut.Grad, zeroCell); err != nil {
		return 0, fmt.Errorf("decoder lstm backward: %w", err)
	}
	dInitH := m.Decoder.LSTM.GetPrevHiddenGrad()
	dInitC := m.Decoder.LSTM.GetPrevCellGrad()
	if dInitH == nil || dInitC == nil {
		return 0, fmt.Errorf("decoder initial state grads missing")
	}
	if decEmb.Grad == nil {
		return 0, fmt.Errorf("decoder embedding grad missing after lstm backward")
	}
	if err := m.Decoder.Embedding.Backward(decEmb.Grad); err != nil {
		return 0, fmt.Errorf("decoder embedding backward: %w", err)
	}

	// Encoder grad: attention contributes at every position; the
	// decoder-seeding path contributes at the final position.
	if len(dInitH.Data) != batch*hiddenSize {
		return 0, fmt.Errorf("decoder init hidden grad has %d elements, want %d", len(dInitH.Data), batch*hiddenSize)
	}
	for b := 0; b < batch; b++ {
		dst := (b*encLen + (encLen - 1)) * hiddenSize
		src := b * hiddenSize
		for h := 0; h < hiddenSize; h++ {
			dEncOut.Data[dst+h] += dInitH.Data[src+h]
		}
	}
	if err := m.Encoder.LSTM.Backward(dEncOut, dInitC); err != nil {
		return 0, fmt.Errorf("encoder lstm backward: %w", err)
	}
	if encEmb.Grad == nil {
		return 0, fmt.Errorf("encoder embedding grad missing after lstm backward")
	}
	if err := m.Encoder.Embedding.Backward(encEmb.Grad); err != nil {
		return 0, fmt.Errorf("encoder embedding backward: %w", err)
	}
	return loss, nil
}
