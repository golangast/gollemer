package moe

import (
	"encoding/gob"
	"fmt"
	"math/rand"

	"github.com/golangast/gollemer/neural/nn"
	. "github.com/golangast/gollemer/neural/tensor"
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
	// Linear layer to project LSTM output to vocabulary size
	OutputLayer *nn.Linear
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
func NewRNNDecoder(inputDim, outputVocabSize, hiddenSize, maxAttentionHeads, numLayers int, dropoutRate float64) (*RNNDecoder, error) {
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

	// Combine hidden (hiddenSize) and attention (inputDim)
	outputLayer, err := nn.NewLinear(hiddenSize+inputDim, outputVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create output linear layer for decoder: %w", err)
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
			OutputVocabSize:   outputVocabSize,
			Embedding:         embedding,
			MaxAttentionHeads: maxAttentionHeads,
			Attention:         attention,
		},
		nil
}

// Forward performs the forward pass of the RNNDecoder.
func (d *RNNDecoder) Forward(contextVector, targetSequence *Tensor, scheduledSamplingProb float64, mask ...*Tensor) ([]*Tensor, error) {
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
			padding := NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float64, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, _ = Concat([]*Tensor{initialHidden, padding}, 1)
		}
	}

	hiddenState := initialHidden
	cellState := NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)

	if scheduledSamplingProb == 0.0 {
		fullInput, _ := targetSequence.Slice(1, 0, maxSequenceLength-1)
		allEmbedded, _ := d.Embedding.Forward(fullInput)

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

		allLogits, err := d.OutputLayer.Forward(normed)
		if err != nil {
			return nil, err
		}

		d.hiddenStates = make([]*Tensor, maxSequenceLength-1)
		d.InitialHiddenState = initialHidden
		d.InitialCellState = cellState
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
		outputLogits, _ := d.OutputLayer.Forward(normed)
		resLogits, _ := outputLogits.Reshape([]int{batchSize, d.OutputVocabSize})
		outputs = append(outputs, resLogits)

		if t < maxSequenceLength-2 {
			if rand.Float64() < scheduledSamplingProb {
				argmax, _ := resLogits.Argmax(1)
				decoderInput, _ = argmax.Reshape([]int{batchSize, 1})
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
		
		// 2b. Re-run LSTM sequence forward to populate timeStepCells for BPTT
		// Use the already-stored initial hidden state and a zero cell state.
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
		_, err = d.OutputLayer.Forward(normed)
		if err != nil {
			return fmt.Errorf("output layer failed during backward re-vectorization: %w", err)
		}
	}

	// --- Optimized Vectorized Backward Path ---
	// 1. Output Layer Backward
	if err := d.OutputLayer.Backward(allGrads); err != nil {
		return fmt.Errorf("output layer backward failed: %w", err)
	}
	normedGrad := d.OutputLayer.Input().Grad

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
	zeroCellGrad := NewTensor(initialCell(batchSize, hiddenSize).Shape, make([]float64, batchSize*hiddenSize), false)
	if err := d.LSTM.Backward(totalHiddenGrad, zeroCellGrad); err != nil {
		return fmt.Errorf("LSTM backward failed: %w", err)
	}

	// 7. Embedding Backward
	inputGrad := d.LSTM.GetInputGrad()
	if inputGrad != nil {
		return d.Embedding.Backward(inputGrad)
	}

	return nil
}

func initialCell(batchSize, hiddenSize int) *Tensor {
	return NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)
}

// DecodeStep performs a single decoding step.
func (d *RNNDecoder) DecodeStep(inputToken *Tensor, prevHiddenState, prevCellState, contextVector *Tensor, mask ...*Tensor) (*Tensor, *Tensor, *Tensor, error) {
	var attentionMask *Tensor
	if len(mask) > 0 {
		attentionMask = mask[0]
	}
	batchSize := inputToken.Shape[0]
	hiddenSize := d.LSTM.HiddenSize

	// 1. Embed
	embeddedInput, err := d.Embedding.Forward(inputToken)
	if err != nil {
		return nil, nil, nil, err
	}

	// 2. LSTM
	reshapedIn, _ := embeddedInput.Reshape([]int{batchSize, embeddedInput.Shape[2]})
	hiddenState, cellState, err := d.LSTM.Forward(reshapedIn, prevHiddenState, prevCellState)
	if err != nil {
		return nil, nil, nil, err
	}

	// 3. Attention
	hiddenQuery, _ := hiddenState.Reshape([]int{batchSize, 1, hiddenSize})
	attentionOutput, err := d.Attention.Forward(hiddenQuery, contextVector, contextVector, attentionMask)
	if err != nil {
		return nil, nil, nil, err
	}

	// 4. Combined
	combined, _ := Concat([]*Tensor{hiddenQuery, attentionOutput}, 2)
	normed, _ := d.LayerNorm.Forward(combined)
	outputLogits, err := d.OutputLayer.Forward(normed)
	if err != nil {
		return nil, nil, nil, err
	}

	resLogits, _ := outputLogits.Reshape([]int{batchSize, d.OutputVocabSize})
	return resLogits, hiddenState, cellState, nil
}

// Parameters returns all learnable parameters of the RNNDecoder.
func (d *RNNDecoder) Parameters() []*Tensor {
	params := []*Tensor{}
	params = append(params, d.Embedding.Parameters()...)
	params = append(params, d.LSTM.Parameters()...)
	if d.LayerNorm != nil {
		params = append(params, d.LayerNorm.Parameters()...)
	}
	params = append(params, d.OutputLayer.Parameters()...)
	params = append(params, d.Attention.Parameters()...)
	return params
}

// ResizeOutputLayer resizes the output layer and embedding layer to match a new vocabulary size, while preserving existing weights.
func (d *RNNDecoder) ResizeOutputLayer(newSize int) {
	oldVocabSize := d.OutputVocabSize
	d.OutputVocabSize = newSize
	inputDim := d.LSTM.HiddenSize + d.Embedding.DimModel

	// 1. Resize Embedding
	oldEmb := d.Embedding
	d.Embedding = nn.NewEmbedding(newSize, oldEmb.DimModel)
	// Copy old weights
	copyLimit := oldVocabSize
	if newSize < copyLimit {
		copyLimit = newSize
	}
	for i := 0; i < copyLimit; i++ {
		start := i * oldEmb.DimModel
		end := start + oldEmb.DimModel
		if end <= len(oldEmb.Weight.Data) && end <= len(d.Embedding.Weight.Data) {
			copy(d.Embedding.Weight.Data[start:end], oldEmb.Weight.Data[start:end])
		}
	}

	// 2. Resize Output Layer (Linear)
	oldOutput := d.OutputLayer
	d.OutputLayer, _ = nn.NewLinear(inputDim, newSize)
	// Copy old weights [InputDim, VocabSize]
	// In row-major format, weights are [inputDim * newSize]
	// Column i represents weights for token i.
	for i := 0; i < inputDim; i++ {
		for j := 0; j < copyLimit; j++ {
			oldIdx := i*oldVocabSize + j
			newIdx := i*newSize + j
			if oldIdx < len(oldOutput.Weights.Data) && newIdx < len(d.OutputLayer.Weights.Data) {
				d.OutputLayer.Weights.Data[newIdx] = oldOutput.Weights.Data[oldIdx]
			}
		}
	}
	// Copy old biases
	if oldOutput.Biases != nil && d.OutputLayer.Biases != nil {
		for j := 0; j < copyLimit; j++ {
			d.OutputLayer.Biases.Data[j] = oldOutput.Biases.Data[j]
		}
	}

	// 3. Resize LayerNorm only if dimensions changed
	if d.LayerNorm == nil || d.LayerNorm.NormalizedShape != inputDim {
		d.LayerNorm = nn.NewLayerNorm(inputDim)
	}
}
