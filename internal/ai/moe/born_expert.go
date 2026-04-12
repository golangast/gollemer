package moe

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math/rand"
	"sync"
	"time"
	"unsafe"

	"github.com/born-ml/born/autodiff"
	"github.com/born-ml/born/backend/cpu"
	"github.com/born-ml/born/nn"
	borntensor "github.com/born-ml/born/tensor"
	gtensor "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

var (
	// Singleton GPU backend for all BornExperts to share a single GPU context via libgoffi/gogpu.
	// Sharing context prevents initialization crashes during parallel
	// initialization of multiple experts on the same device.
	globalBornGPULock    sync.Mutex
	globalBornGPUBackend borntensor.Backend

	// Global execution lock to prevent parallel GPU calls that crash the Vulkan driver.
	// This ensures that only one expert is executing a GPU kernel at a time.
	globalBornGPUExecLock sync.Mutex
)

func getBornGPUContext() (borntensor.Backend, error) {
	globalBornGPULock.Lock()
	defer globalBornGPULock.Unlock()

	if globalBornGPUBackend != nil {
		return globalBornGPUBackend, nil
	}

	// Use CPU backend as the default
	// GPU acceleration via libgoffi/gogpu can be integrated in future versions
	cpuBackend := cpu.New()
	globalBornGPUBackend = borntensor.Backend(cpuBackend)
	return globalBornGPUBackend, nil
}

func init() {
	gob.Register(&BornExpert{})
	gob.Register(bornExpertData{})
}

// BornExpert implements the Expert interface using the born-ml/born framework.
type BornExpert struct {
	inputDim   int
	hiddenDim  int
	outputDim  int
	isTraining bool
	health     float32 // EMA of usage

	backend      *autodiff.Backend[borntensor.Backend]
	fc1          *nn.Linear[*autodiff.Backend[borntensor.Backend]]
	relu         *nn.ReLU[*autodiff.Backend[borntensor.Backend]]
	fc2          *nn.Linear[*autodiff.Backend[borntensor.Backend]]
	paramShadows []*gtensor.Tensor // Shadows that share data with Born params

	// Cache for backward pass
	lastInput  *gtensor.Tensor
	lastOutput *borntensor.Tensor[float32, *autodiff.Backend[borntensor.Backend]]
}

type bornExpertData struct {
	InputDim  int
	HiddenDim int
	OutputDim int
	Health    float32
	FC1Weight []float32
	FC1Bias   []float32
	FC2Weight []float32
	FC2Bias   []float32
}

// GobEncode handles custom serialization since born-ml structs contain non-serializable fields.
func (e *BornExpert) GobEncode() ([]byte, error) {
	data := bornExpertData{
		InputDim:  e.inputDim,
		HiddenDim: e.hiddenDim,
		OutputDim: e.outputDim,
		Health:    e.health,
		FC1Weight: e.fc1.Weight().Tensor().Data(),
		FC2Weight: e.fc2.Weight().Tensor().Data(),
	}

	if e.fc1.Bias() != nil {
		data.FC1Bias = e.fc1.Bias().Tensor().Data()
	}
	if e.fc2.Bias() != nil {
		data.FC2Bias = e.fc2.Bias().Tensor().Data()
	}

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(data); err != nil {
		return nil, fmt.Errorf("failed to encode BornExpert data: %w", err)
	}
	return buf.Bytes(), nil
}

// GobDecode handles custom de-serialization.
func (e *BornExpert) GobDecode(data []byte) error {
	var decoded bornExpertData
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&decoded); err != nil {
		return fmt.Errorf("failed to decode BornExpert data: %w", err)
	}

	// Re-initialize the expert with decoded dimensions
	expert, err := NewBornExpert(decoded.InputDim, decoded.HiddenDim, decoded.OutputDim)
	if err != nil {
		return err
	}

	// Restore health
	expert.health = decoded.Health

	// Restore weights
	copy(expert.fc1.Weight().Tensor().Data(), decoded.FC1Weight)
	copy(expert.fc2.Weight().Tensor().Data(), decoded.FC2Weight)

	if len(decoded.FC1Bias) > 0 && expert.fc1.Bias() != nil {
		copy(expert.fc1.Bias().Tensor().Data(), decoded.FC1Bias)
	}
	if len(decoded.FC2Bias) > 0 && expert.fc2.Bias() != nil {
		copy(expert.fc2.Bias().Tensor().Data(), decoded.FC2Bias)
	}

	// Update the current receiver
	*e = *expert
	return nil
}

// NewBornExpert creates a new BornExpert.
func NewBornExpert(inputDim, hiddenDim, outputDim int) (*BornExpert, error) {
	var base borntensor.Backend = cpu.New()
	backend := autodiff.New(base)

	fc1 := nn.NewLinear(inputDim, hiddenDim, backend)
	relu := nn.NewReLU[*autodiff.Backend[borntensor.Backend]]()
	fc2 := nn.NewLinear(hiddenDim, outputDim, backend)

	backend.Tape().StartRecording() // Ensure operations are recorded for training

	expert := &BornExpert{
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		outputDim: outputDim,
		health:    0.125, // Initial health
		backend:   backend,
		fc1:       fc1,
		relu:      relu,
		fc2:       fc2,
	}

	// Initialize param shadows for global optimizer integration
	for _, p := range expert.allParams() {
		bt := p.Tensor()
		shadow := gtensor.NewTensor(bt.Shape(), bt.Data(), true)
		expert.paramShadows = append(expert.paramShadows, shadow)
	}

	return expert, nil
}

// Forward performs the forward pass, bridging Gollemer and Born-ML tensors.
func (e *BornExpert) Forward(input *gtensor.Tensor) (*gtensor.Tensor, error) {
	// Serialization Lock for GPU stability
	globalBornGPUExecLock.Lock()
	defer globalBornGPUExecLock.Unlock()

	e.lastInput = input

	// Bridge Gollemer tensor to Born tensor
	bt, err := borntensor.FromSlice(input.Data, borntensor.Shape(input.Shape), e.backend)
	if err != nil {
		return nil, fmt.Errorf("failed to convert input to born tensor: %w", err)
	}

	if e.isTraining {
		bt.RequireGrad() // Mark input for grad tracking in training
	}

	var out *borntensor.Tensor[float32, *autodiff.Backend[borntensor.Backend]]
	if e.isTraining {
		h1 := e.fc1.Forward(bt)
		h2 := e.relu.Forward(h1)
		out = e.fc2.Forward(h2)
		e.lastOutput = out
	} else {
		autodiff.NoGrad(e.backend, func() {
			h1 := e.fc1.Forward(bt)
			h2 := e.relu.Forward(h1)
			out = e.fc2.Forward(h2)
		})
	}

	// Bridge back to Gollemer tensor
	return gtensor.NewTensor(out.Shape(), out.Data(), e.isTraining), nil
}

// Backward performs the backward pass.
func (e *BornExpert) Backward(grad *gtensor.Tensor) error {
	// Serialization Lock for GPU stability
	globalBornGPUExecLock.Lock()
	defer globalBornGPUExecLock.Unlock()

	if !e.isTraining {
		return nil // No-op if not in training mode
	}

	if e.lastOutput == nil {
		return fmt.Errorf("backward called before forward")
	}

	// Convert Gollemer gradient to Born tensor
	gradBT, err := borntensor.FromSlice(grad.Data, borntensor.Shape(grad.Shape), e.backend)
	if err != nil {
		return fmt.Errorf("failed to convert grad to born tensor: %w", err)
	}

	// Compute gradients using the tape directly
	grads := e.backend.Tape().Backward(gradBT.Raw(), e.backend)

	// Bridge gradients back to Gollemer shadows for the global optimizer
	for i, p := range e.allParams() {
		raw := p.Tensor().Raw()
		if g, ok := grads[raw]; ok {
			shadow := e.paramShadows[i]
			if shadow.Grad == nil {
				shadow.Grad = gtensor.NewTensor(shadow.Shape, make([]float32, len(shadow.Data)), false)
			}
			gtensor.AddAccumulate(shadow.Grad.Data, g.AsFloat32())
		}
	}

	e.backend.Tape().Clear() // IMPORTANT: reset for next batch
	return nil
}

// shadowToBytes reinterprets a []float32 as []byte for GPU upload (zero-copy).
func shadowToBytes(data []float32) []byte {
	if len(data) == 0 {
		return nil
	}
	//nolint:gosec // G103: unsafe.Slice for zero-copy float32→byte, safe here
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
}

// SyncParameters uploads updated weights from CPU shadows back to the GPU backend.
func (e *BornExpert) SyncParameters() error {
	if e.backend == nil {
		return nil
	}
	// Use the inner backend's Update method on the raw tensor
	for i, p := range e.allParams() {
		shadow := e.paramShadows[i]
		raw := p.Tensor().Raw()
		// Try to sync to GPU if the backend supports explicit updates (e.g. libgoffi/gogpu)
		if gpu, ok := e.backend.Inner().(interface {
			Update(*borntensor.RawTensor, []byte) error
		}); ok {
			if err := gpu.Update(raw, shadowToBytes(shadow.Data)); err != nil {
				return fmt.Errorf("failed to sync parameter %d to GPU: %w", i, err)
			}
		}
	}
	return nil
}

func (e *BornExpert) Parameters() []*gtensor.Tensor {
	return e.paramShadows
}

func (e *BornExpert) Inputs() []*gtensor.Tensor {
	if e.lastInput == nil {
		return []*gtensor.Tensor{}
	}
	return []*gtensor.Tensor{e.lastInput}
}

func (e *BornExpert) Description() string {
	return fmt.Sprintf("BornExpert(in=%d, hid=%d, out=%d)", e.inputDim, e.hiddenDim, e.outputDim)
}

func (e *BornExpert) SetMode(training bool) {
	e.isTraining = training
}

func (e *BornExpert) ClearState() {
	e.lastInput = nil
	e.lastOutput = nil
}

func (e *BornExpert) ClipWeights(maxVal float32) {
	for _, p := range append(e.fc1.Parameters(), e.fc2.Parameters()...) {
		data := p.Tensor().Data()
		for i, v := range data {
			if v > maxVal {
				data[i] = maxVal
			} else if v < -maxVal {
				data[i] = -maxVal
			}
		}
	}
}

func (e *BornExpert) EvolutionaryReset(winner Expert, jitterScale float32) {
	w, ok := winner.(*BornExpert)
	if !ok {
		return
	}

	params := e.allParams()
	winnerParams := w.allParams()

	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)

	for i, p := range params {
		data := p.Tensor().Data()
		winnerData := winnerParams[i].Tensor().Data()
		for j := range data {
			data[j] = winnerData[j] + (r.Float32()*2-1)*jitterScale
		}
	}
}

func (e *BornExpert) Shake(intensity float32) {
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)

	for _, p := range e.allParams() {
		data := p.Tensor().Data()
		for i := range data {
			data[i] += (r.Float32() - 0.5) * intensity
		}
	}
}

func (e *BornExpert) IsStagnant() bool {
	return e.health < 0.01
}

func (e *BornExpert) UpdateHealth(wasUsed bool) {
	const decay = 0.99
	var current float32 = 0.0
	if wasUsed {
		current = 1.0
	}
	e.health = (current * (1.0 - decay)) + (e.health * decay)
}

func (e *BornExpert) ToGPU() {
	if e.backend.Inner().Name() == "cpu" {
		// Already on primary backend or GPU using libgoffi/gogpu
		return
	}

	fmt.Println("🚀 Moving BornExpert to shared GPU context (libgoffi/gogpu)...")
	gpu, err := getBornGPUContext()
	if err != nil {
		fmt.Printf("❌ Failed to initialize GPU backend: %v\n", err)
		return
	}

	// Migrate parameters to new backend
	// Note: We create a NEW Autodiff wrapper so this expert has its own private Tape,
	// but it shares the underlying GPU hardware context with other experts.
	newBackend := autodiff.New(gpu)

	// Create new layers on GPU
	newFc1 := nn.NewLinear(e.inputDim, e.hiddenDim, newBackend)
	newFc2 := nn.NewLinear(e.hiddenDim, e.outputDim, newBackend)

	// Copy weights
	copy(newFc1.Weight().Tensor().Data(), e.fc1.Weight().Tensor().Data())
	if e.fc1.Bias() != nil && newFc1.Bias() != nil {
		copy(newFc1.Bias().Tensor().Data(), e.fc1.Bias().Tensor().Data())
	}
	copy(newFc2.Weight().Tensor().Data(), e.fc2.Weight().Tensor().Data())
	if e.fc2.Bias() != nil && newFc2.Bias() != nil {
		copy(newFc2.Bias().Tensor().Data(), e.fc2.Bias().Tensor().Data())
	}

	e.fc1 = newFc1
	e.fc2 = newFc2
	e.backend = newBackend

	// Re-link param shadows
	e.paramShadows = nil
	for _, p := range e.allParams() {
		bt := p.Tensor()
		shadow := gtensor.NewTensor(bt.Shape(), bt.Data(), true)
		e.paramShadows = append(e.paramShadows, shadow)
	}
}

func (e *BornExpert) Resize(newOutputDim int) {
	// Resize is a complex operation in Born-ML due to fixed shapes in compiled kernels.
	// For now, we'll re-initialize the fc2 layer if shape changes.
	if newOutputDim == e.outputDim {
		return
	}

	// Create new fc2
	newFc2 := nn.NewLinear(e.hiddenDim, newOutputDim, e.backend)

	// Copy old weights where possible
	oldW := e.fc2.Weight().Tensor().Data()
	newW := newFc2.Weight().Tensor().Data()

	copyLimit := e.outputDim
	if newOutputDim < copyLimit {
		copyLimit = newOutputDim
	}

	for i := 0; i < e.hiddenDim; i++ {
		copy(newW[i*newOutputDim:], oldW[i*e.outputDim:i*e.outputDim+copyLimit])
	}

	e.fc2 = newFc2
	e.outputDim = newOutputDim

	// Re-initialize param shadows for global optimizer
	e.paramShadows = nil
	for _, p := range e.allParams() {
		bt := p.Tensor()
		shadow := gtensor.NewTensor(bt.Shape(), bt.Data(), true)
		e.paramShadows = append(e.paramShadows, shadow)
	}
}

func (e *BornExpert) allParams() []*nn.Parameter[*autodiff.Backend[borntensor.Backend]] {
	return append(e.fc1.Parameters(), e.fc2.Parameters()...)
}
