package moe

import (
	"encoding/gob"
	"fmt"
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

	// Intermediate states for BPTT (not serialized)
	hiddenStates     []*Tensor // Hidden state at each timestep
	cellStates       []*Tensor // Cell state at each timestep
	embeddedInputs   []*Tensor // Embedded inputs at each timestep
	attentionOutputs []*Tensor // Attention outputs at each timestep
	combinedInputs   []*Tensor // Combined inputs to LSTM at each timestep
	decoderInputs    []*Tensor // Decoder inputs at each timestep
	contextVector    *Tensor   // Context vector from encoder (saved for backward pass)
	attentionMask    *Tensor   // Attention mask from forward (saved for backward pass)
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

	// Create layer normalization for the combined [hidden + attention] vector
	layerNorm := nn.NewLayerNorm(hiddenSize + inputDim)

	// Create output layer
	var outputLayer *nn.Linear
	var outputMoE *MoELayer
	
	if numExperts > 1 {
		expertBuilder := func(expertIdx int) (Expert, error) {
			return NewBornExpert(hiddenSize+inputDim, (hiddenSize+inputDim)*2, outputVocabSize)
		}
		moeLayer, err := NewMoELayer(hiddenSize+inputDim, outputVocabSize, numExperts, 1, expertBuilder)
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
		LayerNorm:         layerNorm,
		OutputLayer:       outputLayer,
		OutputMoE:         outputMoE,
		OutputVocabSize:   outputVocabSize,
		Embedding:         embedding,
		MaxAttentionHeads: maxAttentionHeads,
		Attention:         attention,
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
	hiddenSize := d.LSTM.HiddenSize
	d.contextVector = contextVector
	d.attentionMask = attentionMask

	// Calculate initial hidden state from context (using Mean over sequence dimension)
	// This ensures autograd connectivity back to the contextVector.
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
	cellState := NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	// Context Injection Multiplier
	const contextMultiplier = 2.0
	ctxMean, _ := contextVector.Mean(1)
	ctxMeanReshaped, _ := ctxMean.Reshape([]int{batchSize, 1, contextVector.Shape[2]})

	if scheduledSamplingProb == 0.0 {
		fullInput, _ := targetSequence.Slice(1, 0, maxSequenceLength-1)
		allEmbedded, _ := d.Embedding.Forward(fullInput)

		// Reinforced Context Injection
		allEmbedded, _ = allEmbedded.AddWithBroadcast(ctxMeanReshaped.Scale(contextMultiplier))

		// 1. LSTM first
		allHidden, lastCell, err := d.LSTM.Forward(allEmbedded, initialHidden, cellState)
		if err != nil {
			return nil, fmt.Errorf("vectorized LSTM failed: %w", err)
		}
		cellState = lastCell

		// 2. Attention using LSTM Hidden states as Queries
		allAttention, err := d.Attention.Forward(allHidden, contextVector, contextVector, attentionMask)
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
		
		// Reinforced Context Injection
		embeddedInput, _ = embeddedInput.Add(ctxMeanReshaped.Scale(float32(contextMultiplier)))
		
		d.embeddedInputs = append(d.embeddedInputs, embeddedInput)

		// 1. LSTM
		reshapedIn, _ := embeddedInput.Reshape([]int{batchSize, embeddedInput.Shape[2]})
		hiddenState, cellState, err = d.LSTM.Forward(reshapedIn, hiddenState, cellState)
		if err != nil {
			return nil, err
		}
		d.hiddenStates = append(d.hiddenStates, hiddenState)
		d.cellStates = append(d.cellStates, cellState)

		// 2. Attention (Query is current hidden state)
		hiddenQuery, _ := hiddenState.Reshape([]int{batchSize, 1, hiddenSize})
		attentionOutput, err := d.Attention.Forward(hiddenQuery, contextVector, contextVector, attentionMask)
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
	return outputs, nil
}

// ClearState clears the intermediate states of the decoder to free memory.
func (d *RNNDecoder) ClearState() {
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
	if d.OutputMoE != nil {
		d.OutputMoE.ClearState()
	}
	if d.Attention != nil {
		d.Attention.ClearState()
	}
	if d.OutputLayer != nil {
		d.OutputLayer.ClearState()
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
		const contextMultiplier = 2.0
		ctxMean, _ := d.contextVector.Mean(1)
		ctxMeanReshaped, _ := ctxMean.Reshape([]int{batchSize, 1, d.contextVector.Shape[2]})
		allEmbedded, _ = allEmbedded.AddWithBroadcast(ctxMeanReshaped.Scale(float32(contextMultiplier)))

		// 2b. Re-run LSTM sequence forward to populate timeStepCells for BPTT
		allHidden, _, err := d.LSTM.Forward(allEmbedded, d.InitialHiddenState, initialCell(batchSize, hiddenSize))
		if err != nil {
			return fmt.Errorf("LSTM forward failed during backward re-vectorization: %w", err)
		}
		
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
	hiddenGradFromAttention := d.Attention.Query().Grad

	// 5. Combine Hidden Gradients
	totalHiddenGrad := hiddenGradFromOutput
	if hiddenGradFromAttention != nil {
		totalHiddenGrad, err = hiddenGradFromOutput.Add(hiddenGradFromAttention)
		if err != nil {
			return fmt.Errorf("failed to add hidden gradients: %w", err)
		}
	}

	// 6. LSTM Backward (Sequence Path)
	// No next hidden/cell grad from future here as the decoder is the end of the chain.
	zeroCellGrad := NewTensor(initialCell(batchSize, hiddenSize).Shape, make([]float32, batchSize*hiddenSize), false)
	if err := d.LSTM.Backward(totalHiddenGrad, zeroCellGrad); err != nil {
		return fmt.Errorf("LSTM backward failed: %w", err)
	}

	// 7. Embedding Backward
	inputGrad := d.LSTM.GetInputGrad() // This is the grad for (Embedded + ContextMean)
if inputGrad != nil {
    // Since we used Add(ctxMean), the gradient flows 1:1 to both branches
    
    // Branch A: To Embedding
    if err := d.Embedding.Backward(inputGrad); err != nil {
        return err
    }
    
    // Branch B: To Context Vector (via Mean and Multiplier)
    const contextMultiplier = 2.0
    // 1. Sum over decoder sequence dimension to get grad for ctxMean from the LSTM inputs
    gradCtxMeanFromInputs, _ := inputGrad.Sum(1) 
    
    // 2. Scale by multiplier (y = x + k*z => dz = k*dy)
    gradCtxMean := gradCtxMeanFromInputs.Scale(float32(contextMultiplier))
 
    // 3. Add gradients from the initial hidden state (which came from context mean but NO multiplier)
    initialHiddenGrad := d.LSTM.GetPrevHiddenGrad()
    if initialHiddenGrad != nil {
        gradCtxMean, _ = gradCtxMean.Add(initialHiddenGrad)
    }

    // 4. Distribute back to encoder sequence dimension
    // ctxMean = sum(contextVector) / S, so dL/dv_i = (dL/dctxMean) / S
    encSeqLen := d.contextVector.Shape[1]
    distGrad := gradCtxMean.Scale(1.0 / float32(encSeqLen))
    
    // expandedGrad shape [Batch, encSeqLen, Dim]
    expandedGrad := distGrad.Expand([]int{batchSize, encSeqLen, embeddingDim})
    
    if d.contextVector.Grad == nil {
        d.contextVector.Grad = expandedGrad
    } else {
        d.contextVector.Grad, _ = d.contextVector.Grad.Add(expandedGrad)
    }
}

	return nil
}

func initialCell(batchSize, hiddenSize int) *Tensor {
	return NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)
}

// DecodeStep performs a single decoding step.
func (d *RNNDecoder) DecodeStep(inputToken *Tensor, prevHiddenState, prevCellState, contextVector *Tensor, mask ...*Tensor) (*Tensor, *Tensor, *Tensor, error) {
	var attentionMask *Tensor
	if len(mask) > 0 {
		attentionMask = mask[0]
	}
	batchSize := inputToken.Shape[0]
	hiddenSize := d.LSTM.HiddenSize

	embeddedInput, err := d.Embedding.Forward(inputToken)
	if err != nil {
		return nil, nil, nil, err
	}

	// Reinforced Context Injection
	const contextMultiplier = 2.0
	ctxMean, _ := contextVector.Mean(1)
	ctxMeanReshaped, _ := ctxMean.Reshape([]int{batchSize, 1, contextVector.Shape[2]})
	embeddedInput, _ = embeddedInput.AddWithBroadcast(ctxMeanReshaped.Scale(contextMultiplier))

	// 2. LSTM
	reshapedIn, _ := embeddedInput.Reshape([]int{batchSize, embeddedInput.Shape[2]})
	hiddenState, cellState, err := d.LSTM.Forward(reshapedIn, prevHiddenState, prevCellState)
	if err != nil {
		return nil, nil, nil, err
	}

	// --- [Diagnostic Probe] ---
	// Check for Hidden State signal collapse using SIMD dot product (magnitude squared)
	hMag := DotProduct(hiddenState.Data, hiddenState.Data)
	if hMag < 1e-6 {
		fmt.Printf("⚠️ [Decoder Diagnostic] Signal Collapse! Hidden State Magnitude: %.8f\n", float64(hMag))
	}
	// --- [/Diagnostic Probe] ---

	// 3. Attention
	hiddenQuery, _ := hiddenState.Reshape([]int{batchSize, 1, hiddenSize})
	attentionOutput, err := d.Attention.Forward(hiddenQuery, contextVector, contextVector, attentionMask)
	if err != nil {
		return nil, nil, nil, err
	}

	// 4. Combined
	combined, _ := Concat([]*Tensor{hiddenQuery, attentionOutput}, 2)
	normed, _ := d.LayerNorm.Forward(combined)
	
	var outputLogits *Tensor
	if d.OutputMoE != nil {
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
func (d *RNNDecoder) DecodeStepWithExpert(input *Tensor, prevHiddenState, prevCellState, contextVector *Tensor) (*Tensor, *Tensor, *Tensor, int, error) {
	logits, h, c, err := d.DecodeStep(input, prevHiddenState, prevCellState, contextVector)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	
	expertID := -1 // Use -1 as default to detect if MoE was skipped or failed
	if d.OutputMoE != nil && len(d.OutputMoE.TopExpertIDs) > 0 {
		expertID = d.OutputMoE.TopExpertIDs[0]
	} else if d.OutputLayer != nil {
		expertID = 0 // Standard layer acts as Expert 0
	}
	
	return logits, h, c, expertID, nil
}

// Parameters returns all learnable parameters of the RNNDecoder.
func (d *RNNDecoder) Parameters() []*Tensor {
	params := []*Tensor{}
	params = append(params, d.Embedding.Parameters()...)
	params = append(params, d.LSTM.Parameters()...)
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
