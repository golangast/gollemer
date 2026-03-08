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
}

// NewRNNDecoder creates a new RNNDecoder.
// inputDim is the dimension of the context vector from the encoder.
// outputVocabSize is the size of the target vocabulary.
// hiddenSize is the hidden dimension of the LSTM.
// numLayers is the number of LSTM layers.
// dropoutRate is the dropout rate between LSTM layers.
func NewRNNDecoder(inputDim, outputVocabSize, hiddenSize, maxAttentionHeads, numLayers int, dropoutRate float64) (*RNNDecoder, error) {
	// LSTM input dimension will be embeddingDim + attentionOutputDim
	// Assuming attentionOutputDim is also inputDim for simplicity in this setup
	lstmInputDim := inputDim + inputDim // embeddedInput + attentionOutput

	// Create multi-layer LSTM with dropout
	lstm, err := nn.NewLSTM(lstmInputDim, hiddenSize, numLayers)
	if err != nil {
		return nil, fmt.Errorf("failed to create LSTM for decoder: %w", err)
	}
	lstm.DropoutRate = dropoutRate
	lstm.Training = true // Will be set to false during inference

	// Create layer normalization
	layerNorm := nn.NewLayerNorm(hiddenSize)

	outputLayer, err := nn.NewLinear(hiddenSize, outputVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create output linear layer for decoder: %w", err)
	}

	embedding := nn.NewEmbedding(outputVocabSize, inputDim)

	attention, err := nn.NewMultiHeadCrossAttention(inputDim, inputDim, hiddenSize, maxAttentionHeads, maxAttentionHeads)
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
// It takes the context vector from the encoder and the target sequence (for teacher forcing) and generates a sequence of tokens.
// scheduledSamplingProb: probability of using model's own prediction instead of ground truth (0.0 = pure teacher forcing, 1.0 = pure sampling)
func (d *RNNDecoder) Forward(contextVector, targetSequence *Tensor, scheduledSamplingProb float64) ([]*Tensor, error) {
	// Validate input shapes
	if len(contextVector.Shape) != 3 {
		return nil, fmt.Errorf("contextVector must be 3D tensor [batchSize, sequenceLength, embeddingDim], got shape %v", contextVector.Shape)
	}
	if len(targetSequence.Shape) != 2 {
		return nil, fmt.Errorf("targetSequence must be 2D tensor [batchSize, sequenceLength], got shape %v", targetSequence.Shape)
	}

	batchSize := targetSequence.Shape[0]
	maxSequenceLength := targetSequence.Shape[1]
	hiddenSize := d.LSTM.HiddenSize
	d.contextVector = contextVector // Save context vector for backward pass

	// Initialize hidden and cell states for the LSTM
	// Use the contextVector to initialize the hidden state
	// The contextVector is [batchSize, sequenceLength, embeddingDim]. We need [batchSize, hiddenSize]
	// For now, let's take the mean of the contextVector along the sequence length dimension
	initialHidden, err := contextVector.Mean(1)
	if err != nil {
		return nil, fmt.Errorf("failed to get mean of context vector for initial hidden state: %w", err)
	}

	// If the hiddenSize of LSTM is different from embeddingDim, we need a linear projection
	if initialHidden.Shape[1] != hiddenSize {
		// This case needs a projection layer, which is not currently in the decoder.
		// For simplicity, let's assume hiddenSize == embeddingDim for now, or handle this with a linear layer.
		// For now, we'll just resize if possible, or error if not compatible.
		if initialHidden.Shape[1] > hiddenSize {
			initialHidden, err = initialHidden.Slice(1, 0, hiddenSize)
			if err != nil {
				return nil, fmt.Errorf("failed to slice initial hidden state: %w", err)
			}
		} else if initialHidden.Shape[1] < hiddenSize {
			// Pad with zeros if hiddenSize is larger
			padding := NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float64, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, err = Concat([]*Tensor{initialHidden, padding}, 1)
			if err != nil {
				return nil, fmt.Errorf("failed to pad initial hidden state: %w", err)
			}
		}
	}

	hiddenState := initialHidden
	cellState := NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false) // Initialize cell state to zeros

	// Create a tensor to hold the decoder outputs
	var outputs []*Tensor

	// Initialize intermediate state storage for BPTT
	d.hiddenStates = make([]*Tensor, 0, maxSequenceLength-1)
	d.cellStates = make([]*Tensor, 0, maxSequenceLength-1)
	d.embeddedInputs = make([]*Tensor, 0, maxSequenceLength-1)
	d.attentionOutputs = make([]*Tensor, 0, maxSequenceLength-1)
	d.combinedInputs = make([]*Tensor, 0, maxSequenceLength-1)
	d.decoderInputs = make([]*Tensor, 0, maxSequenceLength-1)

	// Start with the first token of the target sequence (teacher forcing)
	// We assume targetSequence starts with SOS.
	// Optimization: If pure teacher forcing, use vectorized operations
	if scheduledSamplingProb == 0.0 {
		// Prepare sequence of inputs (all but the last one)
		fullInput, err := targetSequence.Slice(1, 0, maxSequenceLength-1)
		if err != nil {
			return nil, fmt.Errorf("failed to slice full target sequence: %w", err)
		}

		// Embed all inputs at once
		allEmbedded, err := d.Embedding.Forward(fullInput)
		if err != nil {
			return nil, fmt.Errorf("vectorized embedding failed: %w", err)
		}

		// Apply attention to the whole sequence
		allAttention, err := d.Attention.Forward(allEmbedded, contextVector, contextVector)
		if err != nil {
			return nil, fmt.Errorf("vectorized attention failed: %w", err)
		}

		// Concatenate (no reshape needed if axis 2 used)
		reshapedCombined, err := Concat([]*Tensor{allEmbedded, allAttention}, 2)
		if err != nil {
			return nil, fmt.Errorf("vectorized concat failed: %w", err)
		}

		// Run LSTM through the whole sequence
		allHidden, lastCell, err := d.LSTM.Forward(reshapedCombined, initialHidden, cellState)
		if err != nil {
			return nil, fmt.Errorf("vectorized LSTM failed: %w", err)
		}
		cellState = lastCell

		// Final projection
		var finalOutput *Tensor
		if d.LayerNorm != nil {
			reshapedHidden, err := allHidden.Reshape([]int{batchSize * (maxSequenceLength - 1), hiddenSize})
			if err != nil {
				return nil, err
			}
			normalizedHidden, err := d.LayerNorm.Forward(reshapedHidden)
			if err != nil {
				return nil, err
			}
			finalOutput, _ = normalizedHidden.Reshape([]int{batchSize, maxSequenceLength - 1, hiddenSize})
		} else {
			finalOutput = allHidden
		}

		allLogits, err := d.OutputLayer.Forward(finalOutput)
		if err != nil {
			return nil, err
		}

		// In vectorized mode, we return the 3D tensor directly to avoid expensive slicing.
		// The training loop can detect this and use a more efficient loss path.
		outputs = []*Tensor{allLogits}
		
		// For the vectorized Backward to work, we just need to ensure hiddenStates has the right length
		// so the initial check in Backward passes.
		d.hiddenStates = make([]*Tensor, maxSequenceLength-1)

		d.InitialHiddenState = initialHidden
		d.InitialCellState = cellState
		return outputs, nil
	}

	// Start with the first token of the target sequence (teacher forcing)
	// We assume targetSequence starts with SOS.
	decoderInput, err := targetSequence.Slice(1, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to slice initial decoder input: %w", err)
	}

	for t := 0; t < maxSequenceLength-1; t++ {
		// Scheduled sampling: decide between ground truth and model prediction for NEXT iteration
		d.decoderInputs = append(d.decoderInputs, decoderInput)

		// Embed the decoder input
		embeddedInput, err := d.Embedding.Forward(decoderInput)
		if err != nil {
			return nil, fmt.Errorf("decoder embedding failed: %w", err)
		}
		d.embeddedInputs = append(d.embeddedInputs, embeddedInput)

		// Apply attention
		attentionOutput, err := d.Attention.Forward(embeddedInput, contextVector, contextVector)
		if err != nil {
			return nil, fmt.Errorf("decoder attention failed: %w", err)
		}
		d.attentionOutputs = append(d.attentionOutputs, attentionOutput)

		// Concatenate embedded input and attention output
		combinedInput, err := Concat([]*Tensor{embeddedInput, attentionOutput}, 1)
		if err != nil {
			return nil, fmt.Errorf("decoder concat failed: %w", err)
		}

		// Reshape combinedInput from [batchSize, 2, embeddingDim] to [batchSize, 2 * embeddingDim]
		reshapedCombinedInput, err := combinedInput.Reshape([]int{batchSize, combinedInput.Shape[1] * combinedInput.Shape[2]})
		if err != nil {
			return nil, fmt.Errorf("failed to reshape combined input for LSTM: %w", err)
		}
		d.combinedInputs = append(d.combinedInputs, reshapedCombinedInput)

		// Pass through LSTM
		hiddenState, cellState, err = d.LSTM.Forward(reshapedCombinedInput, hiddenState, cellState)
		if err != nil {
			return nil, fmt.Errorf("decoder LSTM forward failed: %w", err)
		}

		// Store hidden and cell states for this timestep
		d.hiddenStates = append(d.hiddenStates, hiddenState)
		d.cellStates = append(d.cellStates, cellState)

		// Apply layer normalization (if available - for backward compatibility)
		var normalizedHidden *Tensor
		if d.LayerNorm != nil {
			var err error
			normalizedHidden, err = d.LayerNorm.Forward(hiddenState)
			if err != nil {
				return nil, fmt.Errorf("decoder layer norm forward failed: %w", err)
			}
		} else {
			// For backward compatibility with models saved before LayerNorm
			normalizedHidden = hiddenState
		}

		// Hidden state to output logits
		outputLogits, err := d.OutputLayer.Forward(normalizedHidden)
		if err != nil {
			return nil, fmt.Errorf("decoder output layer forward failed: %w", err)
		}

		outputs = append(outputs, outputLogits)

		// Prepare input for NEXT timestep (scheduled sampling decision)
		if t < maxSequenceLength-2 { // Don't need next input for last timestep
			useModelPrediction := rand.Float64() < scheduledSamplingProb

			if useModelPrediction {
				// Use model's own prediction (scheduled sampling)
				argmax, err := outputLogits.Argmax(1)
				if err != nil {
					return nil, fmt.Errorf("argmax failed during scheduled sampling: %w", err)
				}
				// Use the predicted token IDs as the next input
				decoderInput, err = argmax.Reshape([]int{batchSize, 1})
				if err != nil {
					return nil, fmt.Errorf("failed to reshape argmax: %w", err)
				}
			} else {
				// Use ground truth (teacher forcing)
				slicedTensor, err := targetSequence.Slice(1, t+1, t+2)
				if err != nil {
					return nil, fmt.Errorf("error slicing target sequence: %w", err)
				}
				decoderInput = slicedTensor
			}
		}
	}
	// fmt.Println("Finished decoder loop")

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

	// LSTM cells are cleared by the main DetachModel loop, but we can do it here too for safety
	if d.LSTM != nil {
		d.LSTM.ClearState()
		for _, layer := range d.LSTM.Cells {
			for _, cell := range layer {
				cell.InputTensor = nil
				cell.PrevHidden = nil
				cell.PrevCell = nil
			}
		}
	}
	d.contextVector = nil
}

// Backward performs the backward pass of the RNNDecoder with proper BPTT.
func (d *RNNDecoder) Backward(grads []*Tensor) error {
	// If it's a single 3D tensor, we don't need to check length against hiddenStates here,
	// because we'll check it inside the vectorized block.
	if len(grads) != len(d.hiddenStates) && !(len(grads) == 1 && len(grads[0].Shape) == 3) {
		return fmt.Errorf("gradient length (%d) doesn't match number of timesteps (%d)", len(grads), len(d.hiddenStates))
	}

	// Check if we can use vectorized backward
	// If the OutputLayer.input is a whole sequence (from vectorized Forward)
	if d.OutputLayer.Input() != nil && len(d.OutputLayer.Input().Shape) == 3 {
		var allGrads *Tensor
		numTimesteps := len(d.hiddenStates)
		batchSize := grads[0].Shape[0]
		hiddenSize := d.LSTM.HiddenSize
		
		if len(grads) == 1 && len(grads[0].Shape) == 3 {
			allGrads = grads[0]
			numTimesteps = grads[0].Shape[1]
		} else {
			vocabSize := grads[0].Shape[1]
			allGradsData := make([]float64, batchSize*numTimesteps*vocabSize)
			for t, grad := range grads {
				for b := 0; b < batchSize; b++ {
					copy(allGradsData[(b*numTimesteps+t)*vocabSize:(b*numTimesteps+t+1)*vocabSize], grad.Data[b*vocabSize:(b+1)*vocabSize])
				}
			}
			allGrads = NewTensor([]int{batchSize, numTimesteps, vocabSize}, allGradsData, false)
		}
		
		err := d.OutputLayer.Backward(allGrads)
		if err != nil {
			return err
		}
		
		hiddenGrad := d.OutputLayer.Input().Grad
		
		// 1.5 Backprop through layer norm if exists
		if d.LayerNorm != nil {
			err = d.LayerNorm.Backward(hiddenGrad)
			if err != nil {
				return err
			}
			// Reshape gradient back to 3D
			hiddenGrad, err = d.LayerNorm.Input().Reshape([]int{batchSize, numTimesteps, hiddenSize})
			if err != nil {
				return err
			}
		}
		
		// 2. Vectorized backprop through LSTM (includes BPTT)
		// We need a final cell gradient (usually zeros)
		zeroCellGrad := NewTensor(d.InitialCellState.Shape, make([]float64, len(d.InitialCellState.Data)), false)
		err = d.LSTM.Backward(hiddenGrad, zeroCellGrad)
		if err != nil {
			return err
		}
		
		// LSTM input grad is already computed and stored in its input tensor's Grad field
		inputGrad := d.LSTM.GetInputGrad()
		if inputGrad == nil {
			return fmt.Errorf("failed to get LSTM input gradient in vectorized backward")
		}
		
		// 3. Backprop through concat - split the gradient
		embeddingDim := d.Embedding.DimModel
		splitGrads, err := Split(inputGrad, 2, []int{embeddingDim, embeddingDim})
		if err != nil {
			return err
		}
		embeddedGrad := splitGrads[0]
		attentionGrad := splitGrads[1]
		
		// 4. Vectorized backprop through attention
		err = d.Attention.Backward(attentionGrad)
		if err != nil {
			return err
		}
		
		// 5. Vectorized backprop through embedding
		// Embedding.Backward supports 3D
		err = d.Embedding.Backward(embeddedGrad)
		if err != nil {
			return err
		}
		
		if d.InitialHiddenState != nil {
			// Initial hidden state grad comes from the start of the LSTM BPTT
			// LSTM doesn't explicitly return it yet but it's in d.InitialHiddenState.Grad (if connected)
			if d.InitialHiddenState.Creator != nil && d.InitialHiddenState.Grad != nil {
				_ = d.InitialHiddenState.Creator.Backward(d.InitialHiddenState.Grad)
			}
		}
		
		return nil
	}

	numTimesteps := len(grads)
	batchSize := grads[0].Shape[0]
	hiddenSize := d.LSTM.HiddenSize
	if len(d.embeddedInputs) > 0 && d.embeddedInputs[0].Shape[0] != batchSize {
		return fmt.Errorf("decoder backward: batch size mismatch: expected %d, got %d from embeddedInputs", batchSize, d.embeddedInputs[0].Shape[0])
	}

	// Initialize gradients for hidden and cell states (will accumulate from future timesteps)
	nextHiddenGrad := NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)
	nextCellGrad := NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)

	// Backpropagate through time (from last timestep to first)
	for t := numTimesteps - 1; t >= 0; t-- {
		// 1. Backprop through output layer
		err := d.OutputLayer.Backward(grads[t])
		if err != nil {
			return fmt.Errorf("decoder output layer backward at t=%d failed: %w", t, err)
		}

		// Get gradient w.r.t. hidden state from output layer
		hiddenGrad := d.OutputLayer.Input().Grad

		// Backprop through layer norm
		if d.LayerNorm != nil {
			err = d.LayerNorm.Backward(hiddenGrad)
			if err != nil {
				return fmt.Errorf("decoder layer norm backward at t=%d failed: %w", t, err)
			}
			hiddenGrad = d.LayerNorm.Input().Grad
		}

		// Add gradient from future timestep
		AddAccumulate(hiddenGrad.Data, nextHiddenGrad.Data)

		// 2. Backprop through LSTM
		// We need to set up the LSTM cell state to match this timestep
		// The LSTM.Backward expects the current cell's state to be set
		if t > 0 {
			d.LSTM.Cells[0][0].PrevHidden = d.hiddenStates[t-1]
			d.LSTM.Cells[0][0].PrevCell = d.cellStates[t-1]
		} else {
			d.LSTM.Cells[0][0].PrevHidden = d.InitialHiddenState
			d.LSTM.Cells[0][0].PrevCell = d.InitialCellState
		}
		d.LSTM.Cells[0][0].InputTensor = d.combinedInputs[t]

		// Ensure gradient is initialized for the input tensor
		if d.LSTM.Cells[0][0].InputTensor.RequiresGrad && d.LSTM.Cells[0][0].InputTensor.Grad == nil {
			d.LSTM.Cells[0][0].InputTensor.Grad = NewTensor(d.LSTM.Cells[0][0].InputTensor.Shape, make([]float64, len(d.LSTM.Cells[0][0].InputTensor.Data)), false)
		}

		err = d.LSTM.Backward(hiddenGrad, nextCellGrad)
		if err != nil {
			return fmt.Errorf("decoder LSTM backward at t=%d failed: %w", t, err)
		}

		// Get gradients for next iteration
		inputGrad := d.LSTM.Cells[0][0].InputTensor.Grad
		if t > 0 {
			if d.LSTM.Cells[0][0].PrevHidden != nil && d.LSTM.Cells[0][0].PrevHidden.Grad != nil {
				nextHiddenGrad = d.LSTM.Cells[0][0].PrevHidden.Grad
			} else {
				nextHiddenGrad = NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)
			}
			if d.LSTM.Cells[0][0].PrevCell != nil && d.LSTM.Cells[0][0].PrevCell.Grad != nil {
				nextCellGrad = d.LSTM.Cells[0][0].PrevCell.Grad
			} else {
				nextCellGrad = NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)
			}
		}

		// 3. Backprop through reshape (gradient flows straight through)
		// inputGrad is already the right shape [batchSize, 2*embeddingDim]

		// 4. Backprop through concat - split the gradient
		embeddingDim := d.Embedding.DimModel
		splitGrads, err := Split(inputGrad, 1, []int{embeddingDim, embeddingDim})
		if err != nil {
			return fmt.Errorf("decoder split at t=%d failed: %w", t, err)
		}
		embeddedGrad := splitGrads[0]
		attentionGrad := splitGrads[1]

		// 5. Backprop through attention
		// Re-run Attention Forward to restore state for this timestep
		// We use the stored embedded input for this timestep and the saved context vector
		if _, err := d.Attention.Forward(d.embeddedInputs[t], d.contextVector, d.contextVector); err != nil {
			return fmt.Errorf("failed to re-run attention forward at t=%d: %w", t, err)
		}

		// Reshape attention gradient to 3D
		reshapedAttentionGrad, err := attentionGrad.Reshape([]int{batchSize, 1, embeddingDim})
		if err != nil {
			return fmt.Errorf("failed to reshape attention gradient at t=%d: %w", t, err)
		}

		err = d.Attention.Backward(reshapedAttentionGrad)
		if err != nil {
			return fmt.Errorf("decoder attention backward at t=%d failed: %w", t, err)
		}

		// 6. Backprop through embedding
		// Manually set the input for this timestep because Embedding only remembers the last one
		d.Embedding.SetInput(d.decoderInputs[t])
		// Reshape embeddedGrad to [batch, 1, dim] to match embedding output shape
		reshapedEmbeddedGrad, err := embeddedGrad.Reshape([]int{batchSize, 1, embeddingDim})
		if err != nil {
			return fmt.Errorf("failed to reshape embedded gradient at t=%d: %w", t, err)
		}
		err = d.Embedding.Backward(reshapedEmbeddedGrad)
		if err != nil {
			return fmt.Errorf("decoder embedding backward at t=%d failed: %w", t, err)
		}
	}

	// Store gradient for initial hidden state (gradient w.r.t. context vector)
	if d.InitialHiddenState != nil {
		d.InitialHiddenState.Grad = nextHiddenGrad
		if d.InitialHiddenState.Creator != nil {
			_ = d.InitialHiddenState.Creator.Backward(nextHiddenGrad)
		}
	}

	return nil
}

// DecodeStep performs a single decoding step.
// It takes the current input token, the previous hidden and cell states, and the encoder's context vector.
// It returns the output logits for the current step, and the new hidden and cell states.
func (d *RNNDecoder) DecodeStep(inputToken *Tensor, prevHiddenState, prevCellState, contextVector *Tensor) (*Tensor, *Tensor, *Tensor, error) {
	// Embed the decoder input
	embeddedInput, err := d.Embedding.Forward(inputToken)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder embedding failed: %w", err)
	}

	// Apply attention
	attentionOutput, err := d.Attention.Forward(embeddedInput, contextVector, contextVector)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder attention failed: %w", err)
	}

	// Concatenate embedded input and attention output
	combinedInput, err := Concat([]*Tensor{embeddedInput, attentionOutput}, 1)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder concat failed: %w", err)
	}

	// Reshape combinedInput from [batchSize, 2, embeddingDim] to [batchSize, 2 * embeddingDim]
	batchSize := combinedInput.Shape[0]
	reshapedCombinedInput, err := combinedInput.Reshape([]int{batchSize, combinedInput.Shape[1] * combinedInput.Shape[2]})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to reshape combined input for LSTM: %w", err)
	}

	// Pass through LSTM
	hiddenState, cellState, err := d.LSTM.Forward(reshapedCombinedInput, prevHiddenState, prevCellState)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder LSTM forward failed: %w", err)
	}

	// Hidden state to output logits
	outputLogits, err := d.OutputLayer.Forward(hiddenState)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder output layer forward failed: %w", err)
	}

	return outputLogits, hiddenState, cellState, nil
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
