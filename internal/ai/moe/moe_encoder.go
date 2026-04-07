package moe

import (
	"fmt"

	"math/rand"
	"time"

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
	expertBuilder := func(int) (Expert, error) {
		return NewBornExpert(inputSize, hiddenSize, hiddenSize)
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

func (m *MoEEncoder) SetGateTemperature(temp float32) {
	if m.Layer != nil {
		m.Layer.SetGateTemperature(temp)
	}
}

// ToGPU moves the encoder's parameters to the GPU.
func (m *MoEEncoder) ToGPU() {
	if m.Layer != nil {
		m.Layer.ToGPU()
	}
}

// LinearExpert wraps nn.Linear to satisfy the Expert interface.
type LinearExpert struct {
	*nn.Linear
	ActivationEMA float32
	Decay         float32
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
func (l *LinearExpert) ClipWeights(maxVal float32) {
	if l.Linear == nil { return }
	if l.Linear.Weights != nil {
		tensor.ClipWeights(l.Linear.Weights.Data, maxVal)
	}
	if l.Linear.Biases != nil {
		tensor.ClipWeights(l.Linear.Biases.Data, maxVal)
	}
}

// ToGPU moves the expert's parameters to the GPU.
func (l *LinearExpert) ToGPU() {
	if l.Linear != nil {
		l.Linear.ToGPU()
	}
}

// EvolutionaryReset performs a "Genetic Mutation" on the LinearExpert.
func (l *LinearExpert) EvolutionaryReset(winner Expert, jitterScale float32) {
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

// Shake performs an in-place noise injection to all weights of the expert.
func (l *LinearExpert) Shake(intensity float32) {
	if l.Linear == nil || l.Linear.Weights == nil {
		return
	}
	// Use a local seed to avoid global mutex contention in rand
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)

	weights := l.Linear.Weights.Data
	for i := range weights {
		noise := (r.Float32() - 0.5) * intensity
		weights[i] += noise
	}
}

// IsStagnant returns true if the expert's relevance is below a minimal threshold.
func (l *LinearExpert) IsStagnant() bool {
	return l.ActivationEMA < 0.01
}

// UpdateHealth updates the ActivationEMA based on usage.
func (l *LinearExpert) UpdateHealth(wasUsed bool) {
	var current float32 = 0.0
	if wasUsed {
		current = 1.0
	}
	// EMA = (Current * (1 - Decay)) + (Previous * Decay)
	l.ActivationEMA = (current * (1.0 - l.Decay)) + (l.ActivationEMA * l.Decay)
}

// Resize updates the output dimension of the expert.
func (l *LinearExpert) Resize(newOutputDim int) {
	if l.Linear == nil {
		return
	}

	oldWeightsData := l.Linear.Weights.Data
	oldBiasData := l.Linear.Biases.Data
	oldVocabSize := l.Linear.Weights.Shape[1]
	inputDim := l.Linear.Weights.Shape[0]

	copyLimit := oldVocabSize
	if newOutputDim < copyLimit {
		copyLimit = newOutputDim
	}

	newWeightsData := make([]float32, inputDim*newOutputDim)
	for row := 0; row < inputDim; row++ {
		oldStart := row * oldVocabSize
		newStart := row * newOutputDim
		copy(newWeightsData[newStart:newStart+copyLimit], oldWeightsData[oldStart:oldStart+copyLimit])
	}

	newBiasData := make([]float32, newOutputDim)
	if oldBiasData != nil {
		copy(newBiasData, oldBiasData[:min(len(oldBiasData), newOutputDim)])
	}

	// Replace linear layer with new dimensions
	newLinear, _ := nn.NewLinear(inputDim, newOutputDim)
	newLinear.Weights.Data = newWeightsData
	if newLinear.Biases != nil {
		newLinear.Biases.Data = newBiasData
	}
	l.Linear = newLinear
}
