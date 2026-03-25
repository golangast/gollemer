package moe

import (
	"fmt"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// MoEEncoder implements an encoder using Mixture of Experts.
type MoEEncoder struct {
	InputSize  int
	HiddenSize int
	NumLayers  int
	NumExperts int

	// Underlying MoE Layer
	Layer *MoELayer
}

// NewMoEEncoder creates a new Mixture of Experts Encoder.
func NewMoEEncoder(inputSize, hiddenSize, numLayers, numExperts int) (*MoEEncoder, error) {
	// Define expert builder using nn.Linear wrapped in LinearExpert
	expertBuilder := func(i int) (Expert, error) {
		lin, err := nn.NewLinear(inputSize, hiddenSize)
		if err != nil {
			return nil, err
		}
		return &LinearExpert{Linear: lin}, nil
	}

	// Create MoELayer with Top-K=2 (standard for MoE)
	layer, err := NewMoELayer(inputSize, hiddenSize, numExperts, 2, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE layer: %w", err)
	}

	return &MoEEncoder{
		InputSize:     inputSize,
		HiddenSize:    hiddenSize,
		NumLayers:     numLayers,
		NumExperts:    numExperts,
		Layer:         layer,
	}, nil
}

// Forward performs the forward pass for the MoE Encoder.
func (m *MoEEncoder) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("MoEEncoder.Forward expects at least 1 input")
	}
	// MoELayer handles reshaping internally if needed, or expects [batch, seq, dim]
	return m.Layer.Forward(inputs[0])
}

// Parameters returns the parameters of the model.
func (m *MoEEncoder) Parameters() []*tensor.Tensor {
	return m.Layer.Parameters()
}

// Backward performs the backward pass for the MoE Encoder.
func (m *MoEEncoder) Backward(grad *tensor.Tensor) error {
	return m.Layer.Backward(grad)
}

// Inputs returns the input tensors of the encoder.
func (m *MoEEncoder) Inputs() []*tensor.Tensor {
	return m.Layer.Inputs()
}

// SetMode sets the training mode.
func (m *MoEEncoder) SetMode(training bool) {
	m.Layer.SetMode(training)
}

// ClearState clears the internal state of the encoder (input/output tensors) to free memory.
func (m *MoEEncoder) ClearState() {
	m.Layer.ClearState()
}

// GetMoELayers returns all MoE layers used by the encoder.
func (m *MoEEncoder) GetMoELayers() []*MoELayer {
	return []*MoELayer{m.Layer}
}

// LinearExpert wraps nn.Linear to satisfy the Expert interface.
type LinearExpert struct {
	*nn.Linear
}

// SetMode implements the Expert interface.
func (l *LinearExpert) SetMode(training bool) {
	// nn.Linear doesn't have mode-specific behavior
}

// Inputs implements the Expert interface (if required by Operation).
func (l *LinearExpert) Inputs() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

// Description implements the Expert interface.
func (l *LinearExpert) Description() string {
	return "LinearExpert"
}

// Forward implements the Expert interface.
func (l *LinearExpert) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return l.Linear.Forward(input)
}

// ClearState implements the Expert interface.
func (l *LinearExpert) ClearState() {
	if l.Linear != nil {
		l.Linear.ClearState()
	}
}
// ClipWeights implements the Expert interface.
func (l *LinearExpert) ClipWeights(maxVal float64) {
	if l.Linear == nil { return }
	if l.Linear.Weights != nil {
		tensor.ClipWeights(l.Linear.Weights.Data, maxVal)
	}
	if l.Linear.Biases != nil {
		tensor.ClipWeights(l.Linear.Biases.Data, maxVal)
	}
}

// EvolutionaryReset performs a "Genetic Mutation" on the LinearExpert.
func (l *LinearExpert) EvolutionaryReset(winner Expert, jitterScale float64) {
	wExpert, ok := winner.(*LinearExpert)
	if !ok {
		return // Cannot mutate from different expert type
	}

	// 1. Copy weights from the winner
	l.Linear.Weights.CopyFrom(wExpert.Linear.Weights)
	if l.Linear.Biases != nil && wExpert.Linear.Biases != nil {
		l.Linear.Biases.CopyFrom(wExpert.Linear.Biases)
	}

	// 2. Apply Jitter to "mutate"
	l.Linear.Weights.ApplyJitter(jitterScale)

	// 3. Normalized Re-Centering (Ensure non-zero signal)
	if l.Linear.Weights.L2Norm() < 1e-4 {
		l.Linear.Weights.Scale(1.2)
	}

	// 4. Reset gradients
	l.Linear.Weights.ZeroGrad()
}
