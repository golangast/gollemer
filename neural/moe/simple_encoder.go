package moe

import (
	"fmt"

	"github.com/zendrulat/nlptagger/neural/nn"
	. "github.com/zendrulat/nlptagger/neural/tensor"
)

// SimpleRNNEncoder is a simple LSTM-based encoder that replaces the MoE layer.
type SimpleRNNEncoder struct {
	LSTM      *nn.LSTM
	InputDim  int
	HiddenDim int
	NumLayers int

	// Stored for BPTT
	inputTensor  *Tensor
	hiddenStates []*Tensor
	cellStates   []*Tensor
	initialState *Tensor
	initialCell  *Tensor
}

// NewSimpleRNNEncoder creates a new SimpleRNNEncoder.
func NewSimpleRNNEncoder(inputDim, hiddenDim, numLayers int) (*SimpleRNNEncoder, error) {
	lstm, err := nn.NewLSTM(inputDim, hiddenDim, numLayers)
	if err != nil {
		return nil, fmt.Errorf("failed to create LSTM for SimpleRNNEncoder: %w", err)
	}

	return &SimpleRNNEncoder{
		LSTM:      lstm,
		InputDim:  inputDim,
		HiddenDim: hiddenDim,
		NumLayers: numLayers,
	}, nil
}

// Forward performs the forward pass of the SimpleRNNEncoder.
// Returns the context vector as a 3D tensor [batchSize, sequenceLength, hiddenDim].
func (e *SimpleRNNEncoder) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SimpleRNNEncoder.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	e.inputTensor = input

	batchSize := input.Shape[0]
	sequenceLength := input.Shape[1]

	// Initialize history for BPTT
	e.hiddenStates = make([]*Tensor, sequenceLength)
	e.cellStates = make([]*Tensor, sequenceLength)

	// Create initial hidden and cell states (zeros)
	h := NewTensor([]int{batchSize, e.HiddenDim}, make([]float64, batchSize*e.HiddenDim), true)
	c := NewTensor([]int{batchSize, e.HiddenDim}, make([]float64, batchSize*e.HiddenDim), true)
	e.initialState = h
	e.initialCell = c

	for t := 0; t < sequenceLength; t++ {
		timeStepInput, err := input.Slice(1, t, t+1)
		if err != nil {
			return nil, fmt.Errorf("slicing input for encoder at t=%d failed: %w", t, err)
		}
		timeStepInput, err = timeStepInput.Squeeze(1)
		if err != nil {
			return nil, fmt.Errorf("squeezing input for encoder at t=%d failed: %w", t, err)
		}

		// We need to call the LSTM logic layer by layer for a single time step
		layerInput := timeStepInput
		for i := 0; i < e.NumLayers; i++ {
			// Manually set cell state for single-step backward
			e.LSTM.Cells[i][0].InputTensor = layerInput
			e.LSTM.Cells[i][0].PrevHidden = h
			e.LSTM.Cells[i][0].PrevCell = c

			h, c, err = e.LSTM.Cells[i][0].Forward(layerInput, h, c)
			if err != nil {
				return nil, fmt.Errorf("encoder LSTM cell forward at t=%d, layer=%d failed: %w", t, i, err)
			}
			layerInput = h // Input to next layer is hidden state of current layer
		}
		e.hiddenStates[t] = h
		e.cellStates[t] = c
	}

	// Stack all hidden states along a new dimension to create [batchSize, sequenceLength, hiddenDim]
	// Reshape each hidden state from [batchSize, hiddenDim] to [batchSize, 1, hiddenDim]
	var allHiddenStates []*Tensor
	for _, h := range e.hiddenStates {
		reshaped, err := h.Reshape([]int{batchSize, 1, e.HiddenDim})
		if err != nil {
			return nil, fmt.Errorf("failed to reshape hidden state for concatenation: %w", err)
		}
		allHiddenStates = append(allHiddenStates, reshaped)
	}

	// Concatenate along dimension 1 to get [batchSize, sequenceLength, hiddenDim]
	contextVector, err := Concat(allHiddenStates, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to concatenate hidden states: %w", err)
	}

	contextVector.RequiresGrad = true
	return contextVector, nil
}

// Backward performs the backward pass of the SimpleRNNEncoder.
func (e *SimpleRNNEncoder) Backward(grad *Tensor) (*Tensor, error) {
	if grad == nil || grad.Data == nil {
		return nil, nil
	}

	// For now, return nil to skip backward pass through the encoder
	// The LSTM cells will handle their own backpropagation
	return nil, nil
}

// Parameters returns all learnable parameters.
func (e *SimpleRNNEncoder) Parameters() []*Tensor {
	return e.LSTM.Parameters()
}

// Inputs returns the input tensors.
func (e *SimpleRNNEncoder) Inputs() []*Tensor {
	if e.inputTensor != nil {
		return []*Tensor{e.inputTensor}
	}
	return []*Tensor{}
}

// SetMode sets the training mode.
func (e *SimpleRNNEncoder) SetMode(training bool) {
	e.LSTM.Training = training
}

// GetOutputShape returns the output shape (context vector shape).
func (e *SimpleRNNEncoder) GetOutputShape() []int {
	if e.inputTensor != nil {
		return []int{e.inputTensor.Shape[0], e.inputTensor.Shape[1], e.HiddenDim}
	}
	return []int{1, 1, e.HiddenDim}
}