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

// copyAuxCoeff scales the pointer-generator auxiliary loss that trains the
// encoder-side copy attention to concentrate on the target token's input
// positions. Codeintent-only (the EncoderCopy gate is nil for other
// domains, so the loss is zero there). Variable (not const) so tests can
// ablate it.
var copyAuxCoeff = 0.5

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
	// Cross-attention: decoder attends to encoder's per-timestep outputs.
	// This lets the model READ the input (e.g. code) when generating each
	// output token, instead of relying solely on the compressed final state.
	// Query=decoder states, Key/Value=encoder outputs.
	// Residual: the attended context is ADDED to the decoder state (not
	// replacing it), so the model keeps its learned recurrent signal and
	// gains grounding on top. The 1-head PoC replaced the state and
	// scored worse than the no-attention baseline.
	moeIn := decOut
	if m.Decoder.CrossAttn != nil {
		attnOut, err := m.Decoder.CrossAttn.Forward(decOut, encOut, encOut)
		if err != nil {
			return 0, fmt.Errorf("decoder cross-attention forward: %w", err)
		}
		moeIn, err = decOut.Add(attnOut)
		if err != nil {
			return 0, fmt.Errorf("decoder cross-attention residual add failed: %w", err)
		}
	}
	// Mixture-of-experts refines the (residually attended) decoder state
	// before vocab projection.
	moeOut, moeAux, err := m.Decoder.MoE.Forward(moeIn)
	if err != nil {
		return 0, fmt.Errorf("decoder moe forward: %w", err)
	}
	logits, err := m.Decoder.Output.Forward(moeOut) // [B, steps, V]
	if err != nil {
		return 0, fmt.Errorf("decoder output forward: %w", err)
	}

	// Copy mechanism (gocode only; nil gate for other domains): additive
	// copy bias in logit space, applied per (batch, step) over the
	// decoder's own past states. The generation path below is untouched;
	// the bias only adds gradient sources. Caches are kept for the
	// backward pass.
	var copyCaches [][]*copyCache
	if m.Decoder.Copy != nil {
		copyCaches = make([][]*copyCache, batch)
		for b := 0; b < batch; b++ {
			copyCaches[b] = make([]*copyCache, steps)
			for t := 0; t < steps; t++ {
				qOff := (b*steps + t) * hiddenSize
				query := moeOut.Data[qOff : qOff+hiddenSize]
				lOff := (b*steps + t) * vocabSize
				genLogits := logits.Data[lOff : lOff+vocabSize]
				hist, toks := stepHistory(decOut, decInputIDs, b, t, steps, hiddenSize)
				copyCaches[b][t] = m.Decoder.Copy.forwardCopyStep(query, genLogits, hist, toks)
			}
		}
	}

	// Encoder-side copy (codeintent only; nil gate for other domains):
	// same additive-bias machinery, but the attended "history" is the
	// encoder's per-timestep states and the tokens are the encoder's input
	// ids. This biases the logits toward input tokens (file paths,
	// signatures, identifiers) so they can be reproduced verbatim.
	var encCopyCaches [][]*copyCache
	if m.Decoder.EncoderCopy != nil {
		encCopyCaches = make([][]*copyCache, batch)
		for b := 0; b < batch; b++ {
			encCopyCaches[b] = make([]*copyCache, steps)
			// Encoder states for batch row b: [encLen][H] views into encOut.
			encStates := make([][]float32, encLen)
			for e := 0; e < encLen; e++ {
				off := (b*encLen + e) * hiddenSize
				encStates[e] = encOut.Data[off : off+hiddenSize]
			}
			for t := 0; t < steps; t++ {
				qOff := (b*steps + t) * hiddenSize
				query := moeOut.Data[qOff : qOff+hiddenSize]
				lOff := (b*steps + t) * vocabSize
				genLogits := logits.Data[lOff : lOff+vocabSize]
				encCopyCaches[b][t] = m.Decoder.EncoderCopy.forwardCopyStep(query, genLogits, encStates, inputIDs[b])
			}
		}
	}

	// ---- Loss (mean cross-entropy over non-padding target tokens). ----
	loss, dLogits, err := softmaxCrossEntropy(logits, labels, padID, vocabSize)
	if err != nil {
		return 0, err
	}
	// Add the MoE load-balancing auxiliary loss (keeps the router from
	// collapsing onto one expert).
	loss += moeAuxCoeff * moeAux

	// ---- Pointer-generator auxiliary loss (codeintent only): when the
	// target token appears in the encoder input, reward the copy attention
	// for concentrating mass on its positions. Diagnosed 2026-10-15: the
	// gate opens (mean 0.39, max 1.0) but attention stays diffuse
	// (maxW ~0.03), so the copy bias is uniform suppression rather than
	// pointing. The main LM loss does not pressure attention to focus
	// because memorized generation already achieves low loss. This loss
	// directly trains the pointing behavior: L_aux = -log(sum_{i:tok_i==target} w_i).
	var copyAuxLoss float64
	var copyAuxCount int
	// Per-(b,t) target positions in the input, for the backward pass.
	type auxPos struct{ b, t int }
	var auxPositions []auxPos
	if m.Decoder.EncoderCopy != nil {
		for b := 0; b < batch; b++ {
			for t := 0; t < steps; t++ {
				target := labels[b][t]
				if target == padID {
					continue
				}
				cache := encCopyCaches[b][t]
				var mass float32
				for i, tok := range inputIDs[b] {
					if tok == target {
						mass += cache.w[i]
					}
				}
				if mass > 0 {
					copyAuxLoss += -math.Log(float64(mass) + 1e-6)
					copyAuxCount++
					auxPositions = append(auxPositions, auxPos{b, t})
				}
			}
		}
		if copyAuxCount > 0 {
			loss += float32(copyAuxCoeff * copyAuxLoss / float64(copyAuxCount))
		}
	}

	// ---- Copy backward: propagate dLogits through the copy bias into the
	// gate parameters, the query states (moeOut) and the history states
	// (decOut). Computed before Output.Backward; both only read dLogits.
	var dMoeOutCopy, dDecOutCopy *tensor.Tensor
	if m.Decoder.Copy != nil {
		dMoeOutCopy = tensor.NewTensor([]int{batch, steps, hiddenSize}, make([]float32, batch*steps*hiddenSize), false)
		dDecOutCopy = tensor.NewTensor([]int{batch, steps, hiddenSize}, make([]float32, batch*steps*hiddenSize), false)
		for b := 0; b < batch; b++ {
			for t := 0; t < steps; t++ {
				lOff := (b*steps + t) * vocabSize
				dStep := dLogits.Data[lOff : lOff+vocabSize]
				hist, toks := stepHistory(decOut, decInputIDs, b, t, steps, hiddenSize)
				dQuery, dHist := m.Decoder.Copy.backwardCopyStep(dStep, copyCaches[b][t], hist, toks)
				qOff := (b*steps + t) * hiddenSize
				copy(dMoeOutCopy.Data[qOff:qOff+hiddenSize], dQuery)
				for i, dh := range dHist {
					hOff := (b*steps + i) * hiddenSize
					for h := 0; h < hiddenSize; h++ {
						dDecOutCopy.Data[hOff+h] += dh[h]
					}
				}
			}
		}
	}

	// ---- Encoder-side copy backward: dQuery accumulates into the moeOut
	// gradient (same query states as above); dHist accumulates into the
	// encoder-output gradient (added to encOut.Grad alongside the
	// cross-attention key/value gradients below).
	var dMoeOutEncCopy, dEncOutCopy *tensor.Tensor
	if m.Decoder.EncoderCopy != nil {
		dMoeOutEncCopy = tensor.NewTensor([]int{batch, steps, hiddenSize}, make([]float32, batch*steps*hiddenSize), false)
		dEncOutCopy = tensor.NewTensor([]int{batch, encLen, hiddenSize}, make([]float32, batch*encLen*hiddenSize), false)
		for b := 0; b < batch; b++ {
			encStates := make([][]float32, encLen)
			for e := 0; e < encLen; e++ {
				off := (b*encLen + e) * hiddenSize
				encStates[e] = encOut.Data[off : off+hiddenSize]
			}
			for t := 0; t < steps; t++ {
				lOff := (b*steps + t) * vocabSize
				dStep := dLogits.Data[lOff : lOff+vocabSize]
				dQuery, dHist := m.Decoder.EncoderCopy.backwardCopyStep(dStep, encCopyCaches[b][t], encStates, inputIDs[b])
				qOff := (b*steps + t) * hiddenSize
				for h := 0; h < hiddenSize; h++ {
					dMoeOutEncCopy.Data[qOff+h] += dQuery[h]
				}
				for e, dh := range dHist {
					hOff := (b*encLen + e) * hiddenSize
					for h := 0; h < hiddenSize; h++ {
						dEncOutCopy.Data[hOff+h] += dh[h]
					}
				}
			}
		}
		// ---- Pointer-generator aux backward: gradients of
		// -log(sum_{i:tok_i==target} w_i) through the copy softmax.
		// Accumulates into the same dMoeOutEncCopy / dEncOutCopy tensors.
		if copyAuxCount > 0 {
			auxScale := float32(copyAuxCoeff / float64(copyAuxCount))
			invSqrtH := float32(1.0 / math.Sqrt(float64(hiddenSize)))
			for _, ap := range auxPositions {
				b, t := ap.b, ap.t
				target := labels[b][t]
				cache := encCopyCaches[b][t]
				nh := len(cache.w)
				if nh == 0 {
					continue
				}
				// mass = sum of attention on target positions
				var mass float32
				for i, tok := range inputIDs[b] {
					if tok == target {
						mass += cache.w[i]
					}
				}
				if mass <= 0 {
					continue
				}
				// dw_i = dL/dw_i = -auxScale/(mass+eps) for i on target, else 0
				inv := -auxScale / (mass + 1e-6)
				// softmax backward: ds_i = w_i * (dw_i - sum_j dw_j w_j)
				var sDot float32
				for i, tok := range inputIDs[b] {
					if tok == target {
						sDot += inv * cache.w[i]
					}
				}
				qOff := (b*steps + t) * hiddenSize
				query := moeOut.Data[qOff : qOff+hiddenSize]
				encStates := make([][]float32, encLen)
				for e := 0; e < encLen; e++ {
					off := (b*encLen + e) * hiddenSize
					encStates[e] = encOut.Data[off : off+hiddenSize]
				}
				for i, tok := range inputIDs[b] {
					var dw float32
					if tok == target {
						dw = inv
					}
					ds := cache.w[i] * (dw - sDot)
					if ds == 0 {
						continue
					}
					// Use augmented keys (pos enc) for the query-gradient
					// path, matching the forward pass.
					hs := encStates[i]
					if cache.keys != nil {
						hs = cache.keys[i]
					}
					for h := 0; h < hiddenSize; h++ {
						dMoeOutEncCopy.Data[qOff+h] += ds * hs[h] * invSqrtH
						dEncOutCopy.Data[(b*encLen+i)*hiddenSize+h] += ds * query[h] * invSqrtH
					}
				}
			}
		}
	}

	// ---- Backward: decoder output -> MoE -> cross-attn -> decoder LSTM -> decoder embedding ... ----
	if err := m.Decoder.Output.Backward(dLogits); err != nil {
		return 0, fmt.Errorf("decoder output backward: %w", err)
	}
	if moeOut.Grad == nil {
		return 0, fmt.Errorf("moe output grad missing after linear backward")
	}
	if dMoeOutCopy != nil {
		for i, g := range dMoeOutCopy.Data {
			moeOut.Grad.Data[i] += g
		}
	}
	if dMoeOutEncCopy != nil {
		for i, g := range dMoeOutEncCopy.Data {
			moeOut.Grad.Data[i] += g
		}
	}
	dMoeIn, err := m.Decoder.MoE.Backward(moeOut.Grad, moeAuxCoeff)
	if err != nil {
		return 0, fmt.Errorf("decoder moe backward: %w", err)
	}
	// Cross-attention backward: with the residual, the gradient flows to
	// decOut through TWO paths: directly (dL/d(moeIn) * 1) and through the
	// attention query. Both accumulate into decOut.Grad.
	var dDecOut *tensor.Tensor
	if m.Decoder.CrossAttn != nil {
		// Direct residual path.
		if decOut.Grad == nil {
			decOut.Grad = tensor.NewTensor(decOut.Shape, make([]float32, len(decOut.Data)), false)
		}
		for i, g := range dMoeIn.Data {
			decOut.Grad.Data[i] += g
		}
		// Attention query path (accumulates into the existing decOut.Grad).
		if err := m.Decoder.CrossAttn.Backward(dMoeIn); err != nil {
			return 0, fmt.Errorf("decoder cross-attention backward: %w", err)
		}
		if decOut.Grad == nil {
			return 0, fmt.Errorf("decoder lstm output grad missing after cross-attention backward")
		}
		dDecOut = decOut.Grad
		// encOut.Grad now holds the key/value gradients from attention;
		// it will be combined with the final-state gradient below.
	} else {
		dDecOut = dMoeIn
	}
	if dDecOutCopy != nil {
		for i, g := range dDecOutCopy.Data {
			dDecOut.Data[i] += g
		}
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
	// The encoder's final step seeds the decoder (dInitH), AND (with
	// cross-attention) all steps receive gradient via the attention
	// key/value path (encOut.Grad). Layout is batch-major: element (b,t)
	// lives at ((b*encLen)+t)*hiddenSize.
	if len(dInitH.Data) != batch*hiddenSize {
		return 0, fmt.Errorf("decoder init hidden grad has %d elements, want %d", len(dInitH.Data), batch*hiddenSize)
	}
	encGradH := tensor.NewTensor([]int{batch, encLen, hiddenSize}, make([]float32, batch*encLen*hiddenSize), false)
	for b := 0; b < batch; b++ {
		dst := (b*encLen + (encLen - 1)) * hiddenSize
		src := b * hiddenSize
		copy(encGradH.Data[dst:dst+hiddenSize], dInitH.Data[src:src+hiddenSize])
	}
	// Add cross-attention key/value gradients (all encoder timesteps).
	if m.Decoder.CrossAttn != nil && encOut.Grad != nil {
		if len(encOut.Grad.Data) != len(encGradH.Data) {
			return 0, fmt.Errorf("encoder output grad size mismatch: got %d, want %d", len(encOut.Grad.Data), len(encGradH.Data))
		}
		for i, g := range encOut.Grad.Data {
			encGradH.Data[i] += g
		}
	}
	// Add encoder-side copy gradients (all encoder timesteps).
	if dEncOutCopy != nil {
		if len(dEncOutCopy.Data) != len(encGradH.Data) {
			return 0, fmt.Errorf("encoder copy grad size mismatch: got %d, want %d", len(dEncOutCopy.Data), len(encGradH.Data))
		}
		for i, g := range dEncOutCopy.Data {
			encGradH.Data[i] += g
		}
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
