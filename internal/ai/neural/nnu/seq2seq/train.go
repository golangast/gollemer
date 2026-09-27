package seq2seq

// Real backpropagation-through-time training for the Seq2Seq model.
//
// The pre-existing Encoder.Forward / Decoder.Forward helpers run the LSTM one
// 2D step at a time, and every 2D call clears the LSTM's timeStepCells cache —
// so they cannot be used for training. TrainBatch instead drives the layers
// directly with single 3D forward passes (which populate the BPTT caches) and
// then walks the gradient back: output layer -> decoder LSTM -> decoder
// embedding -> encoder LSTM -> encoder embedding.
//
// The caller owns the optimizer: ZeroGrad() before the batch,
// ClipGradients() + Step() after.

import (
	"fmt"
	"math"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// moeAuxCoeff scales the mixture-of-experts load-balancing auxiliary loss.
// Standard value from Switch Transformer; keeps the router using all experts.
const moeAuxCoeff = 0.01

// TrainBatch runs one teacher-forcing forward/backward pass over a batch and
// accumulates gradients into the model parameters.
//
// inputIDs  : [batch][encLen] encoder token ids, padded with padID.
// targetIDs : [batch][decLen] decoder token ids, padded with padID, where
//
//	position 0 holds BOS and the sequence ends with EOS.
//
// padID     : padding token id, excluded from the loss.
//
// Returns the mean cross-entropy loss per non-padding target token.
func TrainBatch(m *Seq2Seq, inputIDs, targetIDs [][]int, padID int) (float32, error) {
	batch := len(inputIDs)
	if batch == 0 {
		return 0, fmt.Errorf("seq2seq TrainBatch: empty batch")
	}
	encLen := len(inputIDs[0])
	decLen := len(targetIDs[0])
	if decLen < 2 {
		return 0, fmt.Errorf("seq2seq TrainBatch: target length %d < 2 (need BOS ... EOS)", decLen)
	}
	steps := decLen - 1 // decoder input = targets[:steps], labels = targets[1:]
	vocabSize := m.OutputVocab.Size()
	hiddenSize := m.HiddenDim

	// ---- Encoder: single 3D pass so BPTT caches are populated. ----
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
	// The decoder is seeded with the encoder's final hidden and cell state.
	lastSlice, err := encOut.Slice(1, encLen-1, encLen) // [B,1,H]
	if err != nil {
		return 0, fmt.Errorf("encoder final hidden slice: %w", err)
	}
	encFinalH, err := lastSlice.Reshape([]int{batch, hiddenSize})
	if err != nil {
		return 0, fmt.Errorf("encoder final hidden reshape: %w", err)
	}

	// ---- Decoder: teacher forcing over the whole target in one 3D pass. ----
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
	// Mixture-of-experts refines the decoder state before vocab projection.
	moeOut, moeAux, err := m.Decoder.MoE.Forward(decOut)
	if err != nil {
		return 0, fmt.Errorf("decoder moe forward: %w", err)
	}
	logits, err := m.Decoder.Output.Forward(moeOut) // [B, steps, V]
	if err != nil {
		return 0, fmt.Errorf("decoder output forward: %w", err)
	}

	// ---- Loss (mean cross-entropy over non-padding target tokens). ----
	loss, dLogits, err := softmaxCrossEntropy(logits, labels, padID, vocabSize)
	if err != nil {
		return 0, err
	}
	// Add the MoE load-balancing auxiliary loss (keeps the router from
	// collapsing onto one expert).
	loss += moeAuxCoeff * moeAux

	// ---- Backward: decoder output -> MoE -> decoder LSTM -> decoder embedding ... ----
	if err := m.Decoder.Output.Backward(dLogits); err != nil {
		return 0, fmt.Errorf("decoder output backward: %w", err)
	}
	if moeOut.Grad == nil {
		return 0, fmt.Errorf("moe output grad missing after linear backward")
	}
	dDecOut, err := m.Decoder.MoE.Backward(moeOut.Grad, moeAuxCoeff)
	if err != nil {
		return 0, fmt.Errorf("decoder moe backward: %w", err)
	}
	zeroCell := tensor.NewTensor([]int{batch, hiddenSize}, make([]float32, batch*hiddenSize), false)
	if err := m.Decoder.LSTM.Backward(dDecOut, zeroCell); err != nil {
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

	// ---- ... across the boundary into the encoder.
	// Only the encoder's final step seeds the decoder, so only that step
	// receives gradient. Layout is batch-major: element (b,t) lives at
	// ((b*encLen)+t)*hiddenSize.
	if len(dInitH.Data) != batch*hiddenSize {
		return 0, fmt.Errorf("decoder init hidden grad has %d elements, want %d", len(dInitH.Data), batch*hiddenSize)
	}
	encGradH := tensor.NewTensor([]int{batch, encLen, hiddenSize}, make([]float32, batch*encLen*hiddenSize), false)
	for b := 0; b < batch; b++ {
		dst := (b*encLen + (encLen - 1)) * hiddenSize
		src := b * hiddenSize
		copy(encGradH.Data[dst:dst+hiddenSize], dInitH.Data[src:src+hiddenSize])
	}
	if err := m.Encoder.LSTM.Backward(encGradH, dInitC); err != nil {
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

// idsToTensor converts a padded batch of token-id rows into a 2D float tensor.
func idsToTensor(ids [][]int) *tensor.Tensor {
	batch := len(ids)
	seq := len(ids[0])
	data := make([]float32, batch*seq)
	for b := range ids {
		for t := 0; t < seq && t < len(ids[b]); t++ {
			data[b*seq+t] = float32(ids[b][t])
		}
	}
	return tensor.NewTensor([]int{batch, seq}, data, false)
}

// softmaxCrossEntropy computes the mean cross-entropy loss over non-padding
// positions and the corresponding dL/dLogits tensor.
func softmaxCrossEntropy(logits *tensor.Tensor, labels [][]int, padID, vocabSize int) (float32, *tensor.Tensor, error) {
	batch := len(labels)
	if batch == 0 {
		return 0, nil, fmt.Errorf("softmaxCrossEntropy: no labels")
	}
	steps := len(labels[0])
	dLogits := tensor.NewTensor(
		[]int{batch, steps, vocabSize},
		make([]float32, batch*steps*vocabSize),
		false,
	)
	var totalLoss float64
	valid := 0
	for b := 0; b < batch; b++ {
		for t := 0; t < steps; t++ {
			label := labels[b][t]
			if label == padID {
				continue
			}
			if label < 0 || label >= vocabSize {
				return 0, nil, fmt.Errorf("label %d out of vocab range [0,%d)", label, vocabSize)
			}
			base := (b*steps + t) * vocabSize
			mx := logits.Data[base]
			for v := 1; v < vocabSize; v++ {
				if logits.Data[base+v] > mx {
					mx = logits.Data[base+v]
				}
			}
			var sum float64
			for v := 0; v < vocabSize; v++ {
				e := math.Exp(float64(logits.Data[base+v] - mx))
				dLogits.Data[base+v] = float32(e)
				sum += e
			}
			// loss = -log(p[label]) = log(sum) - (logit[label]-mx)
			totalLoss += math.Log(sum) - float64(logits.Data[base+label]-mx)
			inv := float32(1 / sum)
			for v := 0; v < vocabSize; v++ {
				dLogits.Data[base+v] *= inv
			}
			dLogits.Data[base+label] -= 1
			valid++
		}
	}
	if valid == 0 {
		return 0, nil, fmt.Errorf("softmaxCrossEntropy: no valid (non-pad) target tokens in batch")
	}
	scale := float32(1) / float32(valid)
	for i := range dLogits.Data {
		dLogits.Data[i] *= scale
	}
	return float32(totalLoss / float64(valid)), dLogits, nil
}
