package moe

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"runtime"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func init() {
	gob.Register(&RNNDecoder{})
}

// RNNDecoder is a simple RNN-based decoder for sequence generation.
type RNNDecoder struct {
	// LSTM layer for recurrent processing
	LSTM *nn.LSTM
	// Layer normalization after LSTM
	LayerNorm *nn.LayerNorm
	// Linear layer to project LSTM output to vocabulary size (used if not using MoE)
	OutputLayer *nn.Linear
	// MoE Layer for output projection (optional)
	OutputMoE *MoELayer
	// Output vocabulary size
	OutputVocabSize int
	// Embedding layer for the decoder input
	Embedding         *nn.Embedding
	MaxAttentionHeads int
	Attention         *nn.MultiHeadCrossAttention
	// Initial hidden and cell states for the LSTM
	InitialHiddenState *Tensor
	InitialCellState   *Tensor

	InputNorm  *nn.LayerNorm // Normalize embeddings before LSTM
	HiddenNorm *nn.LayerNorm // Normalize hidden states before Attention

	// Intermediate states for BPTT (not serialized)
	hiddenStates     []*Tensor // Hidden state at each timestep
	cellStates       []*Tensor // Cell state at each timestep
	embeddedInputs   []*Tensor // Embedded inputs at each timestep
	attentionOutputs []*Tensor // Attention outputs at each timestep
	combinedInputs   []*Tensor // Combined inputs to LSTM at each timestep
	decoderInputs    []*Tensor // Decoder inputs at each timestep
	contextVector    *Tensor   // Context vector from encoder (saved for backward pass)
	attentionMask    *Tensor   // Attention mask from forward (saved for backward pass)

	ContextMultiplier      float32 // Scale for reinforced context injection
	ContextMultiplierDecay float32 // Decay factor per step (e.g. 0.7)
}

// NewRNNDecoder creates a new RNNDecoder.
func NewRNNDecoder(inputDim, outputVocabSize, hiddenSize, maxAttentionHeads, numLayers int, dropoutRate float32, numExperts int) (*RNNDecoder, error) {
	// LSTM input dimension will be embeddingDim (context comes in via cross-attention after LSTM)
	lstmInputDim := inputDim

	// Create multi-layer LSTM with dropout
	lstm, err := nn.NewLSTM(lstmInputDim, hiddenSize, numLayers)
	if err != nil {
		return nil, fmt.Errorf("failed to create LSTM for decoder: %w", err)
	}
	lstm.DropoutRate = dropoutRate
	lstm.Training = true

	// Create layer normalization for different parts of the decoder
	inputNorm := nn.NewLayerNorm(inputDim)
	hiddenNorm := nn.NewLayerNorm(hiddenSize)
	layerNorm := nn.NewLayerNorm(hiddenSize + inputDim)

	// Create output layer
	var outputLayer *nn.Linear
	var outputMoE *MoELayer
	
	if numExperts > 1 {
		expertBuilder := func(expertIdx int) (Expert, error) {
			return NewInternalExpert(expertIdx, hiddenSize+inputDim, (hiddenSize+inputDim)*2, outputVocabSize)
		}
		moeLayer, err := NewMoELayer(hiddenSize+inputDim, outputVocabSize, numExperts, 2, expertBuilder)
		if err != nil {
			return nil, fmt.Errorf("failed to create MoE output layer: %w", err)
		}
		outputMoE = moeLayer
	} else {
		outputLayer, err = nn.NewLinear(hiddenSize+inputDim, outputVocabSize)
		if err != nil {
			return nil, fmt.Errorf("failed to create output linear layer: %w", err)
		}
	}

	embedding := nn.NewEmbedding(outputVocabSize, inputDim)

	attention, err := nn.NewMultiHeadCrossAttention(hiddenSize, inputDim, inputDim, maxAttentionHeads, maxAttentionHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create multi-head attention for decoder: %w", err)
	}

	return &RNNDecoder{
		LSTM:              lstm,
		InputNorm:         inputNorm,
		HiddenNorm:        hiddenNorm,
		LayerNorm:         layerNorm,
		OutputLayer:       outputLayer,
		OutputMoE:         outputMoE,
		OutputVocabSize:   outputVocabSize,
		Embedding:         embedding,
		MaxAttentionHeads: maxAttentionHeads,
		Attention:         attention,
		ContextMultiplier:      10.0,
		ContextMultiplierDecay: 1.0, // Default: no decay
	}, nil
}

// Forward performs the forward pass of the RNNDecoder.
func (d *RNNDecoder) Forward(contextVector, targetSequence *Tensor, scheduledSamplingProb float32, mask ...*Tensor) ([]*Tensor, error) {
	var attentionMask *Tensor
	if len(mask) > 0 {
		attentionMask = mask[0]
	}
	if len(contextVector.Shape) != 3 {
		return nil, fmt.Errorf("contextVector must be 3D tensor [batchSize, sequenceLength, embeddingDim], got shape %v", contextVector.Shape)
	}
	if len(targetSequence.Shape) != 2 {
		return nil, fmt.Errorf("targetSequence must be 2D tensor [batchSize, sequenceLength], got shape %v", targetSequence.Shape)
	}

	batchSize := targetSequence.Shape[0]
	maxSequenceLength := targetSequence.Shape[1]
	var err error
	hiddenSize := d.LSTM.HiddenSize
	d.contextVector = contextVector
	d.attentionMask = attentionMask

	// Calculate initial hidden state from context.
	// This ensures the decoder doesn't start with "diluted" context from padding.
	initialHidden, err := contextVector.Mean(1)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate initial hidden state: %w", err)
	}
	dim := contextVector.Shape[2]
	initialHidden, _ = initialHidden.Reshape([]int{batchSize, dim})

	if initialHidden.Shape[1] != hiddenSize {
		if initialHidden.Shape[1] > hiddenSize {
			initialHidden, _ = initialHidden.Slice(1, 0, hiddenSize)
		} else {
			padding := NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float32, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, _ = Concat([]*Tensor{initialHidden, padding}, 1)
		}
	}

	hiddenState := initialHidden
	cellState, _ := initialHidden.Slice(0, 0, batchSize) // Start cell state with initial hidden too
	if cellState.Shape[1] != hiddenSize {
		// Padding/truncation for cell state if needed
		if cellState.Shape[1] > hiddenSize {
			cellState, _ = cellState.Slice(1, 0, hiddenSize)
		} else {
			padding := NewTensor([]int{batchSize, hiddenSize - cellState.Shape[1]}, make([]float32, batchSize*(hiddenSize-cellState.Shape[1])), false)
			cellState, _ = Concat([]*Tensor{cellState, padding}, 1)
		}
	}

	// Reinforced Context Injection (Normalized)
	ctxMean, _ := d.contextVector.Mean(1)
	ctxMeanNorm := ctxMean.L2Norm()
	// 🛡️ NUMERICAL STABILITY: Use larger epsilon and cap the scaling factor
	// to prevent signal explosion when the context is weak (near zero).
	const ctxEpsilon = 1e-5
	if ctxMeanNorm > ctxEpsilon {
		scale := 1.0 / ctxMeanNorm
		if scale > 10.0 { scale = 10.0 } // Cap scaling factor
		ctxMean = ctxMean.Scale(scale)
	}
	ctxMeanReshaped, _ := ctxMean.Reshape([]int{batchSize, 1, d.contextVector.Shape[2]})
	
	if scheduledSamplingProb == 0.0 {
		fullInput, _ := targetSequence.Slice(1, 0, maxSequenceLength-1)
		allEmbedded, _ := d.Embedding.Forward(fullInput)
		normedEmbedded, _ := d.InputNorm.Forward(allEmbedded)
		
		// Apply Decaying Context Injection
		if d.ContextMultiplierDecay > 0 && d.ContextMultiplierDecay < 1.0 {
			decayData := make([]float32, maxSequenceLength-1)
			scale := d.ContextMultiplier
			for t := 0; t < maxSequenceLength-1; t++ {
				decayData[t] = scale
				scale *= d.ContextMultiplierDecay
			}
			decayTensor := NewTensor([]int{1, maxSequenceLength - 1, 1}, decayData, false)
			scaledCtx, _ := ctxMeanReshaped.MulWithBroadcast(decayTensor)
			normedEmbedded, _ = normedEmbedded.AddWithBroadcast(scaledCtx)
			scaledCtx.Release()
			decayTensor.Release()
		} else {
			normedEmbedded, _ = normedEmbedded.AddWithBroadcast(ctxMeanReshaped.Scale(d.ContextMultiplier))
		}

		// 1. LSTM first
		allHidden, lastCell, err := d.LSTM.Forward(normedEmbedded, initialHidden, cellState)
		if err != nil {
			return nil, fmt.Errorf("vectorized LSTM failed: %w", err)
		}
		cellState = lastCell
		
		// Apply HiddenNorm
		normedHidden, err := d.HiddenNorm.Forward(allHidden)
		if err != nil {
			return nil, err
		}

		// 2. Attention using LSTM Hidden states as Queries
		allAttention, err := d.Attention.Forward(normedHidden, contextVector, contextVector, attentionMask)
		if err != nil {
			return nil, fmt.Errorf("vectorized attention failed: %w", err)
		}

		// 3. Concat Hidden and Attention
		combined, err := Concat([]*Tensor{allHidden, allAttention}, 2)
		if err != nil {
			return nil, err
		}

		// Apply LayerNorm
		normed, err := d.LayerNorm.Forward(combined)
		if err != nil {
			return nil, err
		}

		var allLogits *Tensor
		if d.OutputMoE != nil {
			allLogits, err = d.OutputMoE.Forward(normed)
		} else {
			allLogits, err = d.OutputLayer.Forward(normed)
		}
		if err != nil {
			return nil, err
		}

		// Store states for backward pass
		d.hiddenStates = []*Tensor{allHidden}
		d.InitialHiddenState = initialHidden
		d.InitialCellState = cellState
		d.attentionOutputs = []*Tensor{allAttention}
		d.combinedInputs = []*Tensor{combined}
		
		return []*Tensor{allLogits}, nil
	}

	var outputs []*Tensor
	d.hiddenStates = make([]*Tensor, 0, maxSequenceLength-1)
	d.cellStates = make([]*Tensor, 0, maxSequenceLength-1)
	d.embeddedInputs = make([]*Tensor, 0, maxSequenceLength-1)
	d.attentionOutputs = make([]*Tensor, 0, maxSequenceLength-1)
	d.combinedInputs = make([]*Tensor, 0, maxSequenceLength-1)
	d.decoderInputs = make([]*Tensor, 0, maxSequenceLength-1)

	decoderInput, _ := targetSequence.Slice(1, 0, 1)

	for t := 0; t < maxSequenceLength-1; t++ {
		d.decoderInputs = append(d.decoderInputs, decoderInput)
		embeddedInput, _ := d.Embedding.Forward(decoderInput)
		d.embeddedInputs = append(d.embeddedInputs, embeddedInput)

		// Apply InputNorm first, then inject context as post-norm residual with decay.
		normedIn, _ := d.InputNorm.Forward(embeddedInput)
		stepScale := d.ContextMultiplier * float32(math.Pow(float64(d.ContextMultiplierDecay), float64(t)))
		normedIn, _ = normedIn.AddWithBroadcast(ctxMeanReshaped.Scale(stepScale))

		// 1. LSTM
		reshapedIn, _ := normedIn.Reshape([]int{batchSize, embeddedInput.Shape[2]})
		
		// 🛡️ NUMERICAL SAFETY: Check for NaNs before LSTM
		if len(reshapedIn.Data) > 0 && math.IsNaN(float64(reshapedIn.Data[0])) {
			fmt.Printf("⚠️ [Decoder] NaNs detected in LSTM input at step %d! Resetting to zero.\n", t)
			for i := range reshapedIn.Data { reshapedIn.Data[i] = 0 }
		}
		
		hiddenState, cellState, err = d.LSTM.Forward(reshapedIn, hiddenState, cellState)
		if err != nil {
			return nil, err
		}
		d.hiddenStates = append(d.hiddenStates, hiddenState)
		d.cellStates = append(d.cellStates, cellState)
		
		// Apply HiddenNorm
		hiddenQuery, _ := hiddenState.Reshape([]int{batchSize, 1, hiddenSize})
		normedHidden, _ := d.HiddenNorm.Forward(hiddenQuery)

		// 2. Attention (Query is current hidden state)
		attentionOutput, err := d.Attention.Forward(normedHidden, contextVector, contextVector, attentionMask)
		if err != nil {
			return nil, err
		}
		d.attentionOutputs = append(d.attentionOutputs, attentionOutput)

		// 3. Combined Output
		combined, _ := Concat([]*Tensor{hiddenQuery, attentionOutput}, 2)
		d.combinedInputs = append(d.combinedInputs, combined)

		normed, _ := d.LayerNorm.Forward(combined)
		var outputLogits *Tensor
		if d.OutputMoE != nil {
			outputLogits, _ = d.OutputMoE.Forward(normed)
		} else {
			outputLogits, _ = d.OutputLayer.Forward(normed)
		}
		resLogits, _ := outputLogits.Reshape([]int{batchSize, d.OutputVocabSize})
		outputs = append(outputs, resLogits)

		if t < maxSequenceLength-2 {
			if rand.Float32() < scheduledSamplingProb {
				// Use Nucleus (Top-P) Sampling instead of Argmax to avoid "garbage" noise
				// Sampling is done per batch item.
				nextTokens := make([]float32, batchSize)
				for b := 0; b < batchSize; b++ {
					// Slice out logits for this batch item
					itemLogits, _ := resLogits.Slice(0, b, b+1)
					// Use 0.8 temperature and 0.9 top-P for high-quality diversity
					sampledID, err := SampleFromLogits(itemLogits, 0.8, 0, 0.9)
					if err != nil {
						// Fallback to argmax if sampling fails
						argmax, _ := itemLogits.Argmax(1)
						sampledID = int(argmax.Data[0])
					}
					nextTokens[b] = float32(sampledID)
				}
				decoderInput = NewTensor([]int{batchSize, 1}, nextTokens, false)
			} else {
				decoderInput, _ = targetSequence.Slice(1, t+1, t+2)
			}
		}
	}

	d.InitialHiddenState = initialHidden
	d.InitialCellState = cellState
	ctxMean.Release()
	ctxMeanReshaped.Release()
	return outputs, nil
}

// ClearState clears the intermediate states of the decoder to free memory.
func (d *RNNDecoder) ClearState() {
	for _, t := range d.hiddenStates {
		if t != nil { t.Release() }
	}
	for _, t := range d.cellStates {
		if t != nil { t.Release() }
	}
	for _, t := range d.embeddedInputs {
		if t != nil { t.Release() }
	}
	for _, t := range d.attentionOutputs {
		if t != nil { t.Release() }
	}
	for _, t := range d.combinedInputs {
		if t != nil { t.Release() }
	}
	for _, t := range d.decoderInputs {
		if t != nil { t.Release() }
	}
	if d.InitialHiddenState != nil { d.InitialHiddenState.Release() }
	if d.InitialCellState != nil { d.InitialCellState.Release() }
	if d.contextVector != nil { d.contextVector.Release() }

	d.hiddenStates = nil
	d.cellStates = nil
	d.embeddedInputs = nil
	d.attentionOutputs = nil
	d.combinedInputs = nil
	d.decoderInputs = nil
	d.InitialHiddenState = nil
	d.InitialCellState = nil
	
	if d.LSTM != nil {
		d.LSTM.ClearState()
	}
	if d.Attention != nil {
		d.Attention.ClearState()
	}
	if d.OutputLayer != nil {
		d.OutputLayer.ClearState()
	}
	if d.OutputMoE != nil {
		d.OutputMoE.ClearState()
	}
	if d.Embedding != nil {
		d.Embedding.ClearState()
	}
	d.contextVector = nil
}

// Backward performs the backward pass of the RNNDecoder with proper BPTT.
func (d *RNNDecoder) Backward(grads []*Tensor) error {
	if len(grads) == 0 {
		return nil
	}

	batchSize := grads[0].Shape[0]
	hiddenSize := d.LSTM.HiddenSize
	embeddingDim := d.Embedding.DimModel

	var allGrads *Tensor
	if len(grads) == 1 && len(grads[0].Shape) == 3 {
		allGrads = grads[0]
	} else {
		// --- Re-vectorization path for scheduled sampling or step-by-step grads ---
		// 1. Stack gradients into [batchSize, seqLen, vocabSize]
		reshapedGrads := make([]*Tensor, len(grads))
		for i, g := range grads {
			reshapedGrads[i], _ = g.Reshape([]int{batchSize, 1, d.OutputVocabSize})
		}
		var err error
		allGrads, err = Concat(reshapedGrads, 1)
		if err != nil {
			return fmt.Errorf("failed to concat gradients for vectorized backward: %w", err)
		}

		// 2. Re-construct the sequence logic to set up vectorized states
		// This is MUCH faster than the step-by-step backward loop.
		
		// 2a. Re-construct decoder inputs sequence
		allInputs, err := Concat(d.decoderInputs, 1)
		if err != nil {
			return fmt.Errorf("failed to concat decoder inputs: %w", err)
		}
		allEmbedded, _ := d.Embedding.Forward(allInputs)
		
		// 2a. Re-apply Reinforced Context Injection to match Forward pass for correct BPTT
		ctxMean, _ := d.contextVector.Mean(1)
		ctxMeanReshaped, _ := ctxMean.Reshape([]int{batchSize, 1, d.contextVector.Shape[2]})
		
        // Re-run InputNorm FIRST
        allEmbedded, _ = d.InputNorm.Forward(allEmbedded)
		// Then add context as post-norm residual with decay
		if d.ContextMultiplierDecay > 0 && d.ContextMultiplierDecay < 1.0 {
			seqLen := allEmbedded.Shape[1]
			decayData := make([]float32, seqLen)
			scale := d.ContextMultiplier
			for t := 0; t < seqLen; t++ {
				decayData[t] = scale
				scale *= d.ContextMultiplierDecay
			}
			decayTensor := NewTensor([]int{1, seqLen, 1}, decayData, false)
			scaledCtx, _ := ctxMeanReshaped.MulWithBroadcast(decayTensor)
			allEmbedded, _ = allEmbedded.AddWithBroadcast(scaledCtx)
			scaledCtx.Release()
			decayTensor.Release()
		} else {
			ctxScaled := ctxMeanReshaped.Scale(d.ContextMultiplier)
			allEmbedded, _ = allEmbedded.AddWithBroadcast(ctxScaled)
			ctxScaled.Release()
		}

		// 2b. Re-run LSTM sequence forward to populate timeStepCells for BPTT
		allHidden, _, err := d.LSTM.Forward(allEmbedded, d.InitialHiddenState, initialCell(batchSize, hiddenSize))
		if err != nil {
			return fmt.Errorf("LSTM forward failed during backward re-vectorization: %w", err)
		}
        
        // Re-run HiddenNorm
        allHidden, _ = d.HiddenNorm.Forward(allHidden)
		
		// 2c. Re-run Attention forward to populate query/key/value states for the whole sequence
		allAttention, err := d.Attention.Forward(allHidden, d.contextVector, d.contextVector, d.attentionMask)
		if err != nil {
			return fmt.Errorf("attention forward failed during backward re-vectorization: %w", err)
		}

		// 2d. Re-run Concat, LayerNorm, and OutputLayer forward to populate their internal states
		// This is CRITICAL to ensure that d.OutputLayer.input and d.LayerNorm.inputTensor
		// match the current sequence length, avoiding 'zombie' tensors that cause MatMul panics.
		combined, err := Concat([]*Tensor{allHidden, allAttention}, 2)
		if err != nil {
			return fmt.Errorf("concat failed during backward re-vectorization: %w", err)
		}
		normed, err := d.LayerNorm.Forward(combined)
		if err != nil {
			return fmt.Errorf("layer norm failed during backward re-vectorization: %w", err)
		}
		if d.OutputMoE != nil {
			_, err = d.OutputMoE.Forward(normed)
		} else {
			_, err = d.OutputLayer.Forward(normed)
		}
		if err != nil {
			return fmt.Errorf("output layer forward failed during backward re-vectorization: %w", err)
		}
	}

	// --- Optimized Vectorized Backward Path ---
	// 1. Output Layer Backward
	var normedGrad *Tensor
	if d.OutputMoE != nil {
		if err := d.OutputMoE.Backward(allGrads); err != nil {
			return fmt.Errorf("output MoE backward failed: %w", err)
		}
		if len(d.OutputMoE.Inputs()) > 0 {
			normedGrad = d.OutputMoE.Inputs()[0].Grad
		}
	} else if d.OutputLayer != nil {
		if err := d.OutputLayer.Backward(allGrads); err != nil {
			return fmt.Errorf("output layer backward failed: %w", err)
		}
		normedGrad = d.OutputLayer.Input().Grad
	}

	// 2. LayerNorm Backward
	if err := d.LayerNorm.Backward(normedGrad); err != nil {
		return fmt.Errorf("layer norm backward failed: %w", err)
	}
	combinedGrad := d.LayerNorm.Input().Grad

	// 3. Split Combined [Hidden, Attention]
	splits, err := Split(combinedGrad, 2, []int{hiddenSize, embeddingDim})
	if err != nil {
		return fmt.Errorf("combined grad split failed: %w", err)
	}
	hiddenGradFromOutput := splits[0]
	attentionGrad := splits[1]

	// 4. Attention Backward (Query is LSTM Hidden)
	if err := d.Attention.Backward(attentionGrad); err != nil {
		return fmt.Errorf("attention backward failed: %w", err)
	}
	normedHiddenGrad := d.Attention.Query().Grad
	
	// 5. HiddenNorm Backward
	if err := d.HiddenNorm.Backward(normedHiddenGrad); err != nil {
		return fmt.Errorf("hidden norm backward failed: %w", err)
	}
	hiddenGradFromAttention := d.HiddenNorm.Input().Grad

	// 6. Combine Hidden Gradients
	totalHiddenGrad := hiddenGradFromOutput
	if hiddenGradFromAttention != nil {
		totalHiddenGrad, err = hiddenGradFromOutput.Add(hiddenGradFromAttention)
		if err != nil {
			return fmt.Errorf("failed to add hidden gradients: %w", err)
		}
	}

	// 7. LSTM Backward (Sequence Path)
	// No next hidden/cell grad from future here as the decoder is the end of the chain.
	zeroCellGrad := NewTensor(initialCell(batchSize, hiddenSize).Shape, make([]float32, batchSize*hiddenSize), false)
	if err := d.LSTM.Backward(totalHiddenGrad, zeroCellGrad); err != nil {
		return fmt.Errorf("LSTM backward failed: %w", err)
	}

	// 8. InputNorm Backward (flows from LSTM input grad)
	lstmInputGrad := d.LSTM.GetInputGrad()
	if err := d.InputNorm.Backward(lstmInputGrad); err != nil {
		return fmt.Errorf("input norm backward failed: %w", err)
	}
	
	// 9. Embedding and Context Vector Gradients
	if lstmInputGrad != nil {
		// Branch A: To Embedding (flows through InputNorm)
		if err := d.Embedding.Backward(d.InputNorm.Input().Grad); err != nil {
			return err
		}

		// 2. Scale by multiplier (decayed)
		// Since context injection is now decayed, we must apply the same decay in backward.
		if d.ContextMultiplierDecay > 0 && d.ContextMultiplierDecay < 1.0 {
			seqLen := lstmInputGrad.Shape[1]
			decayData := make([]float32, seqLen)
			scale := d.ContextMultiplier
			for t := 0; t < seqLen; t++ {
				decayData[t] = scale
				scale *= d.ContextMultiplierDecay
			}
			decayTensor := NewTensor([]int{1, seqLen, 1}, decayData, false)
			// Apply decay to gradients before summing over time
			decayedGrads, _ := lstmInputGrad.MulWithBroadcast(decayTensor)
			gradCtxMeanFromInputs, _ := decayedGrads.Sum(1)
			gradCtxMean := gradCtxMeanFromInputs.Scale(1.0) // multiplier already in decayTensor
			
			// Continue with standard flow... (simplified for this edit)
			initialHiddenGrad := d.LSTM.GetPrevHiddenGrad()
			if initialHiddenGrad != nil {
				oldGrad := gradCtxMean
				gradCtxMean, _ = gradCtxMean.Add(initialHiddenGrad)
				oldGrad.Release()
			}
			
			encSeqLen := d.contextVector.Shape[1]
			distGrad := gradCtxMean.Scale(1.0 / float32(encSeqLen))
			expandedGrad := distGrad.Expand([]int{batchSize, encSeqLen, embeddingDim})
			
			if d.contextVector.Grad == nil {
				d.contextVector.Grad = expandedGrad
			} else {
				oldGrad := d.contextVector.Grad
				d.contextVector.Grad, _ = d.contextVector.Grad.Add(expandedGrad)
				oldGrad.Release()
				expandedGrad.Release()
			}
			
			decayedGrads.Release()
			decayTensor.Release()
			gradCtxMeanFromInputs.Release()
			gradCtxMean.Release()
			distGrad.Release()
		} else {
			// Legacy path
			gradCtxMeanFromInputs, _ := lstmInputGrad.Sum(1)
			gradCtxMean := gradCtxMeanFromInputs.Scale(d.ContextMultiplier)
			
			initialHiddenGrad := d.LSTM.GetPrevHiddenGrad()
			if initialHiddenGrad != nil {
				oldGrad := gradCtxMean
				gradCtxMean, _ = gradCtxMean.Add(initialHiddenGrad)
				oldGrad.Release()
			}
			
			encSeqLen := d.contextVector.Shape[1]
			distGrad := gradCtxMean.Scale(1.0 / float32(encSeqLen))
			expandedGrad := distGrad.Expand([]int{batchSize, encSeqLen, embeddingDim})
			
			if d.contextVector.Grad == nil {
				d.contextVector.Grad = expandedGrad
			} else {
				oldGrad := d.contextVector.Grad
				d.contextVector.Grad, _ = d.contextVector.Grad.Add(expandedGrad)
				oldGrad.Release()
				expandedGrad.Release()
			}
			
			gradCtxMeanFromInputs.Release()
			gradCtxMean.Release()
			distGrad.Release()
		}
	}

	// Release local sequence tensors created for re-vectorization
	if allGrads != nil && len(grads) > 1 {
		allGrads.Release()
	}

	// 🛡️ Proactive Release of intermediate backprop tensors
	if zeroCellGrad != nil { zeroCellGrad.Release() }
	if totalHiddenGrad != nil && totalHiddenGrad != hiddenGradFromOutput {
		totalHiddenGrad.Release()
	}
	// Note: normedGrad and combinedGrad are handled by their creators' ClearState()
	// or are references to previous layer grads.

	return nil
}

func initialCell(batchSize, hiddenSize int) *Tensor {
	return NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)
}

// DecodeStep performs a single decoding step.
func (d *RNNDecoder) DecodeStep(inputToken *Tensor, prevHiddenState, prevCellState, contextVector *Tensor, stepIndex int, mask ...*Tensor) (*Tensor, *Tensor, *Tensor, error) {
	var attentionMask *Tensor
	if len(mask) > 0 {
		attentionMask = mask[0]
	}
	batchSize := inputToken.Shape[0]
	hiddenSize := d.LSTM.HiddenSize

	if prevHiddenState == nil {
		prevHiddenState = NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)
	}
	if prevCellState == nil {
		prevCellState = NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)
	}

	embeddedInput, err := d.Embedding.Forward(inputToken)
	if err != nil {
		return nil, nil, nil, err
	}

	// Apply InputNorm first
	normedIn, _ := d.InputNorm.Forward(embeddedInput)
	// Reinforced Context Injection (Normalized Mean)
	ctxMean, _ := contextVector.Mean(1)
	ctxMeanNorm := ctxMean.L2Norm()
	const ctxEpsilon = 1e-5
	if ctxMeanNorm > ctxEpsilon {
		scale := 1.0 / ctxMeanNorm
		if scale > 10.0 { scale = 10.0 } // Cap scaling factor
		ctxMean = ctxMean.Scale(scale)
	}
	ctxMeanReshaped, _ := ctxMean.Reshape([]int{batchSize, 1, contextVector.Shape[2]})
	// 🆕 Context Decay: ContextMultiplier * decay^step so context cools as generation progresses.
	// Step 0 gets full multiplier (topic-setting); later steps let grammar experts drive structure.
	decayedMultiplier := d.ContextMultiplier
	if d.ContextMultiplierDecay > 0 && d.ContextMultiplierDecay < 1.0 && stepIndex > 0 {
		decayedMultiplier *= float32(math.Pow(float64(d.ContextMultiplierDecay), float64(stepIndex)))
		if decayedMultiplier < 0.5 {
			decayedMultiplier = 0.5 // Floor so context never disappears entirely
		}
	}
	normedIn, _ = normedIn.AddWithBroadcast(ctxMeanReshaped.Scale(decayedMultiplier))

	// 2. LSTM
	reshapedIn, _ := normedIn.Reshape([]int{batchSize, embeddedInput.Shape[2]})
	hiddenState, cellState, err := d.LSTM.Forward(reshapedIn, prevHiddenState, prevCellState)
	if err != nil {
		return nil, nil, nil, err
	}

	// Apply HiddenNorm
	hiddenQuery, _ := hiddenState.Reshape([]int{batchSize, 1, hiddenSize})
	normedHidden, _ := d.HiddenNorm.Forward(hiddenQuery)

	// 3. Attention
	attentionOutput, err := d.Attention.Forward(normedHidden, contextVector, contextVector, attentionMask)
	if err != nil {
		return nil, nil, nil, err
	}

	// 4. Combined
	combined, _ := Concat([]*Tensor{hiddenQuery, attentionOutput}, 2)
	normed, _ := d.LayerNorm.Forward(combined)
	
	var outputLogits *Tensor
	if d.OutputMoE != nil {
		// Set step index for step-aware routing bias
		d.OutputMoE.CurrentStepIndex = stepIndex
		outputLogits, err = d.OutputMoE.Forward(normed)
	} else {
		outputLogits, err = d.OutputLayer.Forward(normed)
	}
	if err != nil {
		return nil, nil, nil, err
	}

	resLogits, _ := outputLogits.Reshape([]int{batchSize, d.OutputVocabSize})
	return resLogits, hiddenState, cellState, nil
}

// DecodeStepWithExpert is like DecodeStep but also returns the ID of the top expert used.
func (d *RNNDecoder) DecodeStepWithExpert(input *Tensor, prevHiddenState, prevCellState, contextVector *Tensor, stepIndex int) (*Tensor, *Tensor, *Tensor, []int, error) {
	logits, h, c, err := d.DecodeStep(input, prevHiddenState, prevCellState, contextVector, stepIndex)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	
	expertIDs := []int{}
	if d.OutputMoE != nil && len(d.OutputMoE.SelectedExperts) > 0 {
		expertIDs = d.OutputMoE.SelectedExperts[0]
	} else if d.OutputLayer != nil {
		expertIDs = []int{0}
	}
	
	return logits, h, c, expertIDs, nil
}

// Parameters returns all learnable parameters of the RNNDecoder.
func (d *RNNDecoder) Parameters() []*Tensor {
	params := []*Tensor{}
	params = append(params, d.Embedding.Parameters()...)
	params = append(params, d.LSTM.Parameters()...)
	if d.InputNorm != nil {
		params = append(params, d.InputNorm.Parameters()...)
	}
	if d.HiddenNorm != nil {
		params = append(params, d.HiddenNorm.Parameters()...)
	}
	if d.LayerNorm != nil {
		params = append(params, d.LayerNorm.Parameters()...)
	}
	if d.OutputLayer != nil {
		params = append(params, d.OutputLayer.Parameters()...)
	}
	if d.OutputMoE != nil {
		params = append(params, d.OutputMoE.Parameters()...)
	}
	params = append(params, d.Attention.Parameters()...)
	return params
}

// ResizeOutputLayer resizes the output layer and embedding layer to match a new vocabulary size, while preserving existing weights.
// Memory strategy: old data slices are explicitly zeroed/nilled before allocating new ones so the
// old and new buffers don't coexist at peak, reducing peak RSS by ~50% during resize.
func (d *RNNDecoder) ResizeOutputLayer(newSize int) {
	oldVocabSize := d.OutputVocabSize
	d.OutputVocabSize = newSize
	inputDim := d.LSTM.HiddenSize + d.Embedding.DimModel
	dimModel := d.Embedding.DimModel

	// --- 1. Resize Embedding (memory-safe) ---
	copyLimit := oldVocabSize
	if newSize < copyLimit {
		copyLimit = newSize
	}

	// Build new embedding weight data inline before creating Tensor.
	newEmbData := make([]float32, newSize*dimModel)
	oldEmbData := d.Embedding.Weight.Data
	for i := 0; i < copyLimit; i++ {
		start := i * dimModel
		copy(newEmbData[start:start+dimModel], oldEmbData[start:start+dimModel])
	}

	// Free old embedding data before creating the new embedding.
	d.Embedding.Weight.Data = nil
	d.Embedding.Weight = nil
	d.Embedding = nil
	runtime.GC()

	newEmb := nn.NewEmbedding(newSize, dimModel)
	newEmb.Weight.Data = newEmbData
	d.Embedding = newEmb
	newEmbData = nil // allow GC after handing ownership to Embedding

	// --- 2. Resize Output Layer ---
	if d.OutputLayer != nil {
		// Capture old data inline.
		oldLayerWeights := d.OutputLayer.Weights.Data
		oldLayerBiases := d.OutputLayer.Biases.Data

		newWeights := make([]float32, inputDim*newSize)
		for i := 0; i < inputDim; i++ {
			oldStart := i * oldVocabSize
			newStart := i * newSize
			copy(newWeights[newStart:newStart+copyLimit], oldLayerWeights[oldStart:oldStart+copyLimit])
		}

		newBiases := make([]float32, newSize)
		for j := 0; j < copyLimit && j < len(oldLayerBiases); j++ {
			newBiases[j] = oldLayerBiases[j]
		}

		// Free old layer data before creating the new one.
		d.OutputLayer.Weights.Data = nil
		d.OutputLayer.Biases.Data = nil
		d.OutputLayer.Weights = nil
		d.OutputLayer.Biases = nil
		d.OutputLayer = nil
		runtime.GC()

		newLinear, _ := nn.NewLinear(inputDim, newSize)
		newLinear.Weights.Data = newWeights
		newLinear.Biases.Data = newBiases
		d.OutputLayer = newLinear

	} else if d.OutputMoE != nil {
		// ResizeExperts already performs one-expert-at-a-time GC.
		d.OutputMoE.ResizeExperts(newSize)
		fmt.Printf("✅ Resized all %d Decoder Experts to %d\n", len(d.OutputMoE.Experts), newSize)
	}

	// --- 3. Resize LayerNorm only if dimensions changed ---
	if d.LayerNorm == nil || d.LayerNorm.NormalizedShape != inputDim {
		d.LayerNorm = nn.NewLayerNorm(inputDim)
	}
}

// ToGPU moves the decoder's parameters to the GPU.
func (d *RNNDecoder) ToGPU() {
	if d.LSTM != nil {
		d.LSTM.ToGPU()
	}
	if d.InputNorm != nil {
		d.InputNorm.ToGPU()
	}
	if d.HiddenNorm != nil {
		d.HiddenNorm.ToGPU()
	}
	if d.LayerNorm != nil {
		d.LayerNorm.ToGPU()
	}
	if d.OutputLayer != nil {
		d.OutputLayer.ToGPU()
	}
	if d.OutputMoE != nil {
		d.OutputMoE.ToGPU()
	}
	if d.Embedding != nil {
		d.Embedding.ToGPU()
	}
	if d.Attention != nil {
		d.Attention.ToGPU()
	}
}

func (d *RNNDecoder) SyncParameters() error {
	if d.OutputMoE != nil {
		return d.OutputMoE.SyncParameters()
	}
	return nil
}
// SetMode sets the decoder to training or inference mode.
func (d *RNNDecoder) SetMode(training bool) {
	if d.LSTM != nil {
		d.LSTM.Training = training
	}
	if d.OutputMoE != nil {
		d.OutputMoE.SetMode(training)
	}
	if d.Attention != nil {
		d.Attention.SetMode(training)
	}
}

func (d *RNNDecoder) RepairArchitecture() {
	if d.InputNorm == nil {
		d.InputNorm = nn.NewLayerNorm(d.Embedding.DimModel)
	}
	if d.HiddenNorm == nil {
		d.HiddenNorm = nn.NewLayerNorm(d.LSTM.HiddenSize)
	}
	if d.OutputMoE != nil {
		d.OutputMoE.RepairArchitecture()
	}
}
// Step performs a single step of decoding (used for inference and diagnostics)
func (d *RNNDecoder) Step(tokenID int, contextVector, prevHidden, prevCell *Tensor) (*Tensor, *Tensor, *Tensor, error) {
	batchSize := 1
	if prevHidden != nil {
		batchSize = prevHidden.Shape[0]
	}

	if prevHidden == nil {
		prevHidden = NewTensor([]int{batchSize, d.LSTM.HiddenSize}, make([]float32, batchSize*d.LSTM.HiddenSize), false)
	}
	if prevCell == nil {
		prevCell = NewTensor([]int{batchSize, d.LSTM.HiddenSize}, make([]float32, batchSize*d.LSTM.HiddenSize), false)
	}

	// 1. Embedding
	inputT := NewTensor([]int{batchSize, 1}, []float32{float32(tokenID)}, false)
	embedded, err := d.Embedding.Forward(inputT)
	if err != nil { return nil, nil, nil, err }

	// 3. LSTM
	normedIn, _ := d.InputNorm.Forward(embedded)
	
	// Reinforced Context Injection (Normalized Mean)
	// Match Forward() and DecodeStep() logic: post-norm residual.
	ctxMean, _ := contextVector.Mean(1)
	ctxMeanNorm := ctxMean.L2Norm()
	if ctxMeanNorm > 1e-8 {
		ctxMean = ctxMean.Scale(1.0 / ctxMeanNorm)
	}
	ctxMeanReshaped, _ := ctxMean.Reshape([]int{batchSize, 1, contextVector.Shape[2]})
	normedIn, _ = normedIn.AddWithBroadcast(ctxMeanReshaped.Scale(d.ContextMultiplier))

	reshapedIn, _ := normedIn.Reshape([]int{batchSize, embedded.Shape[2]})
	hidden, cell, err := d.LSTM.Forward(reshapedIn, prevHidden, prevCell)
	if err != nil { return nil, nil, nil, err }

	// 4. Attention
	hiddenQuery, _ := hidden.Reshape([]int{batchSize, 1, d.LSTM.HiddenSize})
	normedHidden, _ := d.HiddenNorm.Forward(hiddenQuery)
	// Passing nil for attention mask during Step (single token inference)
	attnOut, err := d.Attention.Forward(normedHidden, contextVector, contextVector, nil)
	if err != nil { return nil, nil, nil, err }

	// 5. Combined & Output
	combined, _ := Concat([]*Tensor{hiddenQuery, attnOut}, 2)
	normedOut, _ := d.LayerNorm.Forward(combined)
	
	var logits *Tensor
	if d.OutputMoE != nil {
		logits, err = d.OutputMoE.Forward(normedOut)
	} else {
		logits, err = d.OutputLayer.Forward(normedOut)
	}
	if err != nil { return nil, nil, nil, err }

	// Reshape logits to 2D [batch, vocab]
	flatLogits, _ := logits.Reshape([]int{batchSize, d.OutputVocabSize})

	return flatLogits, hidden, cell, nil
}
