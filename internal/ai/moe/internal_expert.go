package moe

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func init() {
	gob.Register(&InternalExpert{})
	// Maintain compatibility with BornExpert data structure for seamless migration
	gob.Register(bornExpertData{})
}

type bornExpertData struct {
	ID        int
	InputDim  int
	HiddenDim int
	OutputDim int
	Health    float32
	FC1Weight []float32
	FC1Bias   []float32
	FC2Weight []float32
	FC2Bias   []float32
}

// Add these to your internal expert file inside package moe:

func (e *InternalExpert) GetFC1() *nn.Linear {
	return e.fc1
}

func (e *InternalExpert) GetFC2() *nn.Linear {
	return e.fc2
}

func (e *InternalExpert) GetHealth() float32 {
	return e.health
}

// InternalExpert implements the Expert interface using the native neural framework.
type InternalExpert struct {
	ID         int
	inputDim   int
	hiddenDim  int
	outputDim  int
	isTraining bool
	health     float32 // EMA of usage

	fc1 *nn.Linear
	fc2 *nn.Linear

	// Cache for backward pass
	lastInput  *tensor.Tensor
	lastOutput *tensor.Tensor
}

// NewInternalExpert creates a new InternalExpert using the native neural package.
func NewInternalExpert(id, inputDim, hiddenDim, outputDim int) (*InternalExpert, error) {
	fc1, err := nn.NewLinear(inputDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create fc1: %w", err)
	}
	fc2, err := nn.NewLinear(hiddenDim, outputDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create fc2: %w", err)
	}

	expert := &InternalExpert{
		ID:        id,
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		outputDim: outputDim,
		health:    0.125, // Initial health
		fc1:       fc1,
		fc2:       fc2,
	}

	// Initialize weights using Xavier/Glorot initialization for stability
	expert.resetWeights()

	return expert, nil
}

func (e *InternalExpert) resetWeights() {
	// inputDim -> hiddenDim
	limit1 := float32(math.Sqrt(6.0 / float64(e.inputDim+e.hiddenDim)))
	for i := range e.fc1.Weights.Data {
		e.fc1.Weights.Data[i] = (rand.Float32() * 2 * limit1) - limit1
	}
	// hiddenDim -> outputDim
	limit2 := float32(math.Sqrt(6.0 / float64(e.hiddenDim+e.outputDim)))
	for i := range e.fc2.Weights.Data {
		e.fc2.Weights.Data[i] = (rand.Float32() * 2 * limit2) - limit2
	}
}

// Forward performs the forward pass using native tensors.
func (e *InternalExpert) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	e.lastInput = input

	// h1 = input @ w1 + b1
	h1, err := e.fc1.Forward(input)
	if err != nil {
		return nil, err
	}

	// h2 = ReLU(h1)
	h2, err := h1.ReLU()
	if err != nil {
		return nil, err
	}

	// out = h2 @ w2 + b2
	out, err := e.fc2.Forward(h2)
	if err != nil {
		return nil, err
	}

	e.lastOutput = out
	return out, nil
}

// Backward performs the backward pass.
func (e *InternalExpert) Backward(grad *tensor.Tensor) error {
	if !e.isTraining {
		return nil
	}

	if e.lastOutput == nil {
		return fmt.Errorf("backward called before forward")
	}

	// Native autodiff: propagate gradient back through the expert's layers
	err := e.lastOutput.Backward(grad)
	if err != nil {
		return err
	}

	// 🛡️ LOCAL GRADIENT CLIPPING (EXPERT-LEVEL)
	// Preserve the "Nuclear Option" against exploding experts.
	const expertLocalClip = 2.0
	for _, p := range e.Parameters() {
		if p.Grad != nil {
			p.ClipGrad(expertLocalClip)
		}
	}

	return nil
}

// GobEncode handles serialization, maintaining compatibility with BornExpert data.
func (e *InternalExpert) GobEncode() ([]byte, error) {
	data := bornExpertData{
		ID:        e.ID,
		InputDim:  e.inputDim,
		HiddenDim: e.hiddenDim,
		OutputDim: e.outputDim,
		Health:    e.health,
		FC1Weight: e.fc1.Weights.Data,
		FC2Weight: e.fc2.Weights.Data,
	}

	if e.fc1.Biases != nil {
		data.FC1Bias = e.fc1.Biases.Data
	}
	if e.fc2.Biases != nil {
		data.FC2Bias = e.fc2.Biases.Data
	}

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(data); err != nil {
		return nil, fmt.Errorf("failed to encode InternalExpert data: %w", err)
	}
	return buf.Bytes(), nil
}

// GobDecode handles de-serialization and migration from BornExpert format.
func (e *InternalExpert) GobDecode(data []byte) error {
	var decoded bornExpertData
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&decoded); err != nil {
		return fmt.Errorf("failed to decode InternalExpert data: %w", err)
	}

	// Re-initialize the expert with decoded dimensions
	expert, err := NewInternalExpert(decoded.ID, decoded.InputDim, decoded.HiddenDim, decoded.OutputDim)
	if err != nil {
		return err
	}

	// Restore health
	expert.health = decoded.Health

	// Restore weights
	copy(expert.fc1.Weights.Data, decoded.FC1Weight)
	copy(expert.fc2.Weights.Data, decoded.FC2Weight)

	if len(decoded.FC1Bias) > 0 && expert.fc1.Biases != nil {
		copy(expert.fc1.Biases.Data, decoded.FC1Bias)
	}
	if len(decoded.FC2Bias) > 0 && expert.fc2.Biases != nil {
		copy(expert.fc2.Biases.Data, decoded.FC2Bias)
	}

	// Update the current receiver
	*e = *expert
	return nil
}

func (e *InternalExpert) Parameters() []*tensor.Tensor {
	return append(e.fc1.Parameters(), e.fc2.Parameters()...)
}

func (e *InternalExpert) Inputs() []*tensor.Tensor {
	if e.lastInput == nil {
		return []*tensor.Tensor{}
	}
	return []*tensor.Tensor{e.lastInput}
}

func (e *InternalExpert) Description() string {
	return fmt.Sprintf("InternalExpert(in=%d, hid=%d, out=%d)", e.inputDim, e.hiddenDim, e.outputDim)
}

func (e *InternalExpert) SetMode(training bool) {
	e.isTraining = training
	e.fc1.SetMode(training)
	e.fc2.SetMode(training)
}

func (e *InternalExpert) ClearState() {
	e.lastInput = nil
	e.lastOutput = nil
	e.fc1.ClearState()
	e.fc2.ClearState()
}

func (e *InternalExpert) ClipWeights(maxVal float32) {
	for _, p := range e.Parameters() {
		p.Clip(-maxVal, maxVal)
	}
}

func (e *InternalExpert) EvolutionaryReset(winner Expert, jitterScale float32) {
	w, ok := winner.(*InternalExpert)
	if !ok {
		return
	}

	params := e.Parameters()
	winnerParams := w.Parameters()

	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)

	for i, p := range params {
		winnerData := winnerParams[i].Data
		for j := range p.Data {
			p.Data[j] = winnerData[j] + (r.Float32()*2-1)*jitterScale
		}
	}
}

func (e *InternalExpert) Shake(intensity float32) {
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)

	for _, p := range e.Parameters() {
		for i := range p.Data {
			p.Data[i] += (r.Float32() - 0.5) * intensity
		}
	}
}

func (e *InternalExpert) IsStagnant() bool {
	return e.health < 0.01
}

func (e *InternalExpert) UpdateHealth(wasUsed bool) {
	const decay = 0.99
	var current float32 = 0.0
	if wasUsed {
		current = 1.0
	}
	e.health = (current * (1.0 - decay)) + (e.health * decay)
}

func (e *InternalExpert) ToGPU() {
	e.fc1.ToGPU()
	e.fc2.ToGPU()
}

func (e *InternalExpert) Resize(newOutputDim int) {
	if newOutputDim == e.outputDim {
		return
	}

	// Create new fc2 (automatically initialized)
	newFc2, _ := nn.NewLinear(e.hiddenDim, newOutputDim)

	// Copy old weights where possible
	oldW := e.fc2.Weights.Data
	newW := newFc2.Weights.Data

	copyLimit := e.outputDim
	if newOutputDim < copyLimit {
		copyLimit = newOutputDim
	}

	for i := 0; i < e.hiddenDim; i++ {
		copy(newW[i*newOutputDim:], oldW[i*e.outputDim:i*e.outputDim+copyLimit])
	}

	// Copy old biases
	if e.fc2.Biases != nil && newFc2.Biases != nil {
		copy(newFc2.Biases.Data, e.fc2.Biases.Data[:copyLimit])
	}

	e.fc2 = newFc2
	e.outputDim = newOutputDim
}

func (e *InternalExpert) SyncParameters() error {
	// Native tensors handle their own synchronization during operations or explicit calls.
	// For example, calling ToGPU() or DispatchGPUMatMul handles sync.
	if e.fc1.Weights.Device == tensor.GPU {
		// If we are on GPU, we might want to ensure weights are fresh?
		// But in the internal framework, weights are updated on whatever device they are.
	}
	return nil
}

func (e *InternalExpert) GetID() int {
	return e.ID
}

func (e *InternalExpert) GetContext() []float32 {
	return []float32{e.health}
}

func (e *InternalExpert) RestoreContext(ctx []float32) {
	if len(ctx) > 0 {
		e.health = ctx[0]
	}
}
