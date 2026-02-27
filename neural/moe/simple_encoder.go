package moe

import (
	"fmt"

	"github.com/golangast/gollemer/neural/nn"
	. "github.com/golangast/gollemer/neural/tensor"
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
	// sequenceLength := input.Shape[1]

	// Create initial hidden and cell states (zeros)
	h := NewTensor([]int{batchSize, e.HiddenDim}, make([]float64, batchSize*e.HiddenDim), true)
	c := NewTensor([]int{batchSize, e.HiddenDim}, make([]float64, batchSize*e.HiddenDim), true)
	e.initialState = h
	e.initialCell = c

	// Use built-in sequence handling in LSTM
	contextVector, _, err := e.LSTM.Forward(input, h, c)
	if err != nil {
		return nil, fmt.Errorf("SimpleRNNEncoder LSTM forward failed: %w", err)
	}

	return contextVector, nil
}

// Backward performs the backward pass of the SimpleRNNEncoder.
func (e *SimpleRNNEncoder) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// Create zero gradient for cell state
	batchSize := grad.Shape[0]
	zeroGradCell := NewTensor([]int{batchSize, e.HiddenDim}, make([]float64, batchSize*e.HiddenDim), false)

	// Perform BPTT through LSTM
	err := e.LSTM.Backward(grad, zeroGradCell)
	if err != nil {
		return fmt.Errorf("SimpleRNNEncoder LSTM backward failed: %w", err)
	}

	// Gradient is stored in inputTensor.Grad
	return nil
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
