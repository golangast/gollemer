package moe

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// FeedForwardExpert is a simple feed-forward neural network that implements the Expert interface.
type FeedForwardExpert struct {
	Layer1             *nn.Linear
	Layer2             *nn.Linear
	inputTensor        *tensor.Tensor
	activationOutput   *tensor.Tensor
	intermediateOutput *tensor.Tensor
	ActivationEMA      float64
	Decay              float64
}

// NewFeedForwardExpert creates a new FeedForwardExpert.
// inputDim is the dimension of the input to the expert.
// hiddenDim is the dimension of the hidden layer.
// outputDim is the dimension of the output from the expert.
func NewFeedForwardExpert(inputDim, hiddenDim, outputDim int) (*FeedForwardExpert, error) {
	layer1, err := nn.NewLinear(inputDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create first linear layer for expert: %w", err)
	}
	layer2, err := nn.NewLinear(hiddenDim, outputDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create second linear layer for expert: %w", err)
	}

	return &FeedForwardExpert{
		Layer1: layer1,
		Layer2: layer2,
		ActivationEMA: 0.125, // Initial health (1/num_experts if 8)
		Decay: 0.99,
	}, nil
}

// Forward performs the forward pass of the FeedForwardExpert.
func (e *FeedForwardExpert) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	e.inputTensor = input

	// Layer 1: Linear -> ReLU
	output1, err := e.Layer1.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("expert layer 1 forward failed: %w", err)
	}
	e.intermediateOutput = output1 // Store for backward pass

	activationOutput, err := output1.ReLU()
	if err != nil {
		return nil, fmt.Errorf("expert activation ReLU failed: %w", err)
	}
	e.activationOutput = activationOutput

	// Layer 2: Linear
	output2, err := e.Layer2.Forward(activationOutput)
	if err != nil {
		return nil, fmt.Errorf("expert layer 2 forward failed: %w", err)
	}

	return output2, nil
}

// Backward performs the backward pass of the FeedForwardExpert.
func (e *FeedForwardExpert) Backward(grad *tensor.Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// Backpropagate through Layer2
	err := e.Layer2.Backward(grad)
	if err != nil {
		return fmt.Errorf("expert layer 2 backward failed: %w", err)
	}

	// Backpropagate through ReLU
	if e.activationOutput == nil || e.activationOutput.Grad == nil {
		return fmt.Errorf("expert activation output or its gradient is nil in backward")
	}
	
	// Explicit check for "Dead ReLU" during backprop
	if e.intermediateOutput != nil {
		if e.intermediateOutput.Grad == nil {
			e.intermediateOutput.Grad = tensor.NewTensor(e.intermediateOutput.Shape, make([]float64, len(e.intermediateOutput.Data)), false)
		}
		for i := range e.intermediateOutput.Data {
			if e.intermediateOutput.Data[i] <= 0 {
				e.intermediateOutput.Grad.Data[i] = 0
			} else if i < len(e.activationOutput.Grad.Data) {
				e.intermediateOutput.Grad.Data[i] += e.activationOutput.Grad.Data[i]
			}
		}
	} else {
		err = e.activationOutput.Creator.Backward(e.activationOutput.Grad)
		if err != nil {
			return fmt.Errorf("expert activation backward failed: %w", err)
		}
	}

	// Backpropagate through Layer1
	if e.intermediateOutput == nil || e.intermediateOutput.Grad == nil {
		return fmt.Errorf("expert intermediate output or its gradient is nil in backward")
	}
	err = e.Layer1.Backward(e.intermediateOutput.Grad)
	if err != nil {
		return fmt.Errorf("expert layer 1 backward failed: %w", err)
	}

	return nil
}

// Parameters returns all learnable parameters of the FeedForwardExpert.
func (e *FeedForwardExpert) Parameters() []*tensor.Tensor {
	params := e.Layer1.Parameters()
	params = append(params, e.Layer2.Parameters()...)
	return params
}

// Inputs returns the input tensors of the FeedForwardExpert's last forward operation.
func (e *FeedForwardExpert) Inputs() []*tensor.Tensor {
	if e.inputTensor != nil {
		return []*tensor.Tensor{e.inputTensor}
	}
	return []*tensor.Tensor{}
}

// Description returns a string description of the expert.
func (e *FeedForwardExpert) Description() string {
	return "FeedForwardExpert"
}

// SetMode sets the mode for the expert.
func (e *FeedForwardExpert) SetMode(training bool) {
	// No specific behavior for training/inference in this simple expert
}

// ClearState clears the expert's internal states.
func (e *FeedForwardExpert) ClearState() {
	e.inputTensor = nil
	e.activationOutput = nil
	e.intermediateOutput = nil
	if e.Layer1 != nil {
		e.Layer1.ClearState()
	}
	if e.Layer2 != nil {
		e.Layer2.ClearState()
	}
}

// ClipWeights bounds the expert's learnable parameters.
func (e *FeedForwardExpert) ClipWeights(maxVal float64) {
	if e.Layer1 != nil {
		tensor.ClipWeights(e.Layer1.Weights.Data, maxVal)
		if e.Layer1.Biases != nil {
			tensor.ClipWeights(e.Layer1.Biases.Data, maxVal)
		}
	}
	if e.Layer2 != nil {
		tensor.ClipWeights(e.Layer2.Weights.Data, maxVal)
		if e.Layer2.Biases != nil {
			tensor.ClipWeights(e.Layer2.Biases.Data, maxVal)
		}
	}
}

// EvolutionaryReset performs a "Genetic Mutation" on the expert.
func (e *FeedForwardExpert) EvolutionaryReset(winner Expert, jitterScale float64) {
	wExpert, ok := winner.(*FeedForwardExpert)
	if !ok {
		return // Cannot mutate from different expert type
	}

	// 1. Copy weights from the winner
	e.Layer1.Weights.CopyFrom(wExpert.Layer1.Weights)
	e.Layer2.Weights.CopyFrom(wExpert.Layer2.Weights)

	if e.Layer1.Biases != nil && wExpert.Layer1.Biases != nil {
		e.Layer1.Biases.CopyFrom(wExpert.Layer1.Biases)
	}
	if e.Layer2.Biases != nil && wExpert.Layer2.Biases != nil {
		e.Layer2.Biases.CopyFrom(wExpert.Layer2.Biases)
	}

	// 2. Apply Jitter to "mutate"
	e.Layer1.Weights.ApplyJitter(jitterScale)
	e.Layer2.Weights.ApplyJitter(jitterScale)

	// 3. Normalized Re-Centering (Ensure non-zero signal)
	if e.Layer1.Weights.L2Norm() < 1e-4 {
		e.Layer1.Weights.Scale(1.2)
	}

	// 4. Reset optimizer-related data (handled by Trainer usually, but we clear Gradients here)
	e.Layer1.Weights.ZeroGrad()
	e.Layer2.Weights.ZeroGrad()
}

// Shake performs an in-place noise injection to break loops.
func (e *FeedForwardExpert) Shake(intensity float64) {
	// We use a local seed to avoid global mutex contention in rand
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)

	// Apply noise to Layer 1 Weights
	weights1 := e.Layer1.Weights.Data
	for i := range weights1 {
		noise := (r.Float64() - 0.5) * intensity
		weights1[i] += noise
	}

	// Apply noise to Layer 2 Weights
	weights2 := e.Layer2.Weights.Data
	for i := range weights2 {
		noise := (r.Float64() - 0.5) * intensity
		weights2[i] += noise
	}
}

// IsStagnant returns true if the expert's relevance is below a minimal threshold.
func (e *FeedForwardExpert) IsStagnant() bool {
	// If 8 experts, ideal health is 0.125. 
	// Threshold of 0.01 means less than 1% utilization.
	return e.ActivationEMA < 0.01
}

// UpdateHealth updates the ActivationEMA based on usage.
func (e *FeedForwardExpert) UpdateHealth(wasUsed bool) {
	var current float64 = 0.0
	if wasUsed {
		current = 1.0
	}
	// EMA = (Current * (1 - Decay)) + (Previous * Decay)
	e.ActivationEMA = (current * (1.0 - e.Decay)) + (e.ActivationEMA * e.Decay)
}
