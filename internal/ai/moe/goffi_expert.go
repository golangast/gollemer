package moe

import (
	"fmt"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// GoffiExpert implements the Expert interface using high-performance goffi-accelerated tensors.
// It consists of two linear layers with a ReLU activation in between (standard MLP expert).
type GoffiExpert struct {
	ID         int
	inputDim   int
	hiddenDim  int
	outputDim  int
	isTraining bool
	health     float32

	FC1   *nn.Linear
	ReLU  bool // activation flag
	FC2   *nn.Linear

	// Cache for backward pass
	lastInput   *tensor.Tensor
	lastReLUOut *tensor.Tensor // Output after ReLU (which is h1 modified in-place)
}

func NewGoffiExpert(id, inputDim, hiddenDim, outputDim int) (*GoffiExpert, error) {
	fc1, err := nn.NewLinear(inputDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	fc2, err := nn.NewLinear(hiddenDim, outputDim)
	if err != nil {
		return nil, err
	}

	return &GoffiExpert{
		ID:        id,
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		outputDim: outputDim,
		health:    0.125,
		FC1:       fc1,
		ReLU:      true,
		FC2:       fc2,
	}, nil
}

func (e *GoffiExpert) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	e.lastInput = input

	// 1. First Linear Layer (creates new tensor)
	h1, err := e.FC1.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("GoffiExpert FC1 failed: %w", err)
	}

	// 2. ReLU Activation (In-place)
	tensor.ReLUVector(h1.Data)
	e.lastReLUOut = h1

	// 3. Second Linear Layer
	out, err := e.FC2.Forward(h1)
	if err != nil {
		return nil, fmt.Errorf("GoffiExpert FC2 failed: %w", err)
	}

	return out, nil
}

func (e *GoffiExpert) Backward(grad *tensor.Tensor) error {
	if !e.isTraining || e.lastReLUOut == nil {
		return nil
	}

	// 1. Backward through FC2
	err := e.FC2.Backward(grad)
	if err != nil {
		return err
	}

	// 2. Backward through ReLU
	// We use lastReLUOut to determine where gradient flows.
	// grad_input = grad_output if input > 0 else 0
	fc2InputGrad := e.FC2.Input().Grad
	if fc2InputGrad == nil {
		return fmt.Errorf("GoffiExpert: FC2 input grad is nil")
	}

	h1GradData := make([]float32, len(e.lastReLUOut.Data))
	for i, v := range e.lastReLUOut.Data {
		if v > 0 {
			h1GradData[i] = fc2InputGrad.Data[i]
		}
	}
	h1Grad := tensor.NewTensor(e.lastReLUOut.Shape, h1GradData, false)

	// 3. Backward through FC1
	return e.FC1.Backward(h1Grad)
}

func (e *GoffiExpert) Parameters() []*tensor.Tensor {
	params := e.FC1.Parameters()
	params = append(params, e.FC2.Parameters()...)
	return params
}

func (e *GoffiExpert) ToGPU() {
	e.FC1.ToGPU()
	e.FC2.ToGPU()
}

func (e *GoffiExpert) SyncParameters() error {
	return nil
}

func (e *GoffiExpert) Description() string {
	return fmt.Sprintf("GoffiExpert(id=%d, hid=%d)", e.ID, e.hiddenDim)
}

func (e *GoffiExpert) SetMode(training bool) {
	e.isTraining = training
}

func (e *GoffiExpert) ClearState() {
	e.FC1.ClearState()
	e.FC2.ClearState()
	e.lastInput = nil
	e.lastReLUOut = nil
}

func (e *GoffiExpert) UpdateHealth(wasUsed bool) {
	const decay = 0.99
	var current float32 = 0.0
	if wasUsed {
		current = 1.0
	}
	e.health = (current * (1.0 - decay)) + (e.health * decay)
}

func (e *GoffiExpert) IsStagnant() bool {
	return e.health < 0.01
}

func (e *GoffiExpert) ClipWeights(maxVal float32) {
	if e.FC1 != nil && e.FC1.Weights != nil {
		tensor.ClipWeights(e.FC1.Weights.Data, maxVal)
	}
	if e.FC2 != nil && e.FC2.Weights != nil {
		tensor.ClipWeights(e.FC2.Weights.Data, maxVal)
	}
}

func (e *GoffiExpert) EvolutionaryReset(winner Expert, jitterScale float32) {
	w, ok := winner.(*GoffiExpert)
	if !ok { return }
	
	// Simple copy + jitter
	e.FC1.Weights.CopyFrom(w.FC1.Weights)
	e.FC1.Weights.ApplyJitter(jitterScale)
	e.FC2.Weights.CopyFrom(w.FC2.Weights)
	e.FC2.Weights.ApplyJitter(jitterScale)
}

func (e *GoffiExpert) Shake(intensity float32) {
	e.FC1.Weights.ApplyJitter(intensity)
	e.FC2.Weights.ApplyJitter(intensity)
}

func (e *GoffiExpert) Resize(newOutputDim int) {
	// Not implemented for simplicity, but not needed for basic MLM
}

func (e *GoffiExpert) Inputs() []*tensor.Tensor {
	if e.lastInput == nil { return nil }
	return []*tensor.Tensor{e.lastInput}
}
