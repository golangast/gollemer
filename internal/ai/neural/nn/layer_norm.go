package nn

import (
	"fmt"
	"math"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// LayerNorm represents a layer normalization module
type LayerNorm struct {
	NormalizedShape int
	Gamma           *Tensor // Learnable scale parameter
	Beta            *Tensor // Learnable shift parameter
	Eps             float32

	// Stored for backward pass
	input      *Tensor
	normalized *Tensor
	mean       *Tensor
	variance   *Tensor
}

// NewLayerNorm creates a new LayerNorm module
func NewLayerNorm(normalizedShape int) *LayerNorm {
	// Initialize gamma to ones and beta to zeros
	gamma := NewTensor([]int{normalizedShape}, make([]float32, normalizedShape), true)
	beta := NewTensor([]int{normalizedShape}, make([]float32, normalizedShape), true)

	for i := range gamma.Data {
		gamma.Data[i] = 1.0
		beta.Data[i] = 0.0
	}

	return &LayerNorm{
		NormalizedShape: normalizedShape,
		Gamma:           gamma,
		Beta:            beta,
		Eps:             1e-5,
	}
}

// Forward performs the forward pass of LayerNorm
// Input shape: [batchSize, (optional seqLen), normalizedShape]
func (ln *LayerNorm) Forward(input *Tensor) (*Tensor, error) {
	if len(input.Shape) < 2 {
		return nil, fmt.Errorf("LayerNorm expects at least 2D input, got shape %v", input.Shape)
	}
	if input.Shape[len(input.Shape)-1] != ln.NormalizedShape {
		return nil, fmt.Errorf("LayerNorm expects last dimension %d, got %d", ln.NormalizedShape, input.Shape[len(input.Shape)-1])
	}

	ln.input = input
	numRows := 1
	for i := 0; i < len(input.Shape)-1; i++ {
		numRows *= input.Shape[i]
	}
	batchSize := numRows // Total number of vectors to normalize

	// Calculate mean and variance for each sample using SIMD
	mean := NewTensor([]int{batchSize}, make([]float32, batchSize), false)
	variance := NewTensor([]int{batchSize}, make([]float32, batchSize), false)

	tmp := make([]float32, ln.NormalizedShape)

	for i := range batchSize {
		start := i * ln.NormalizedShape
		data := input.Data[start : start+ln.NormalizedShape]
		
		m := SumVector(data) / float32(ln.NormalizedShape)
		mean.Data[i] = m

		// Variance: sum((x - m)^2) / n
		AddScalar(data, -m, tmp)
		MulVectors(tmp, tmp, tmp)
		variance.Data[i] = SumVector(tmp) / float32(ln.NormalizedShape)
	}

	ln.mean = mean
	ln.variance = variance

	// Normalize, Scale and Shift in one pass
	output := NewTensor(input.Shape, make([]float32, len(input.Data)), input.RequiresGrad)
	normalized := NewTensor(input.Shape, make([]float32, len(input.Data)), input.RequiresGrad)
	
	for i := range batchSize {
		start := i * ln.NormalizedShape
		std := float32(math.Sqrt(float64(variance.Data[i] + ln.Eps)))
		m := mean.Data[i]
		
		inData := input.Data[start : start+ln.NormalizedShape]
		normData := normalized.Data[start : start+ln.NormalizedShape]
		outData := output.Data[start : start+ln.NormalizedShape]
		
		// (x - m) / std
		AddScalar(inData, -m, normData)
		DivScalar(normData, std, normData)
		
		// y = gamma * norm + beta
		MulVectors(ln.Gamma.Data, normData, outData)
		AddVectors(outData, ln.Beta.Data, outData)
	}
	ln.normalized = normalized

	return output, nil
}

// Backward performs the backward pass of LayerNorm
func (ln *LayerNorm) Backward(gradOutput *Tensor) error {
	numRows := 1
	for i := 0; i < len(ln.input.Shape)-1; i++ {
		numRows *= ln.input.Shape[i]
	}
	batchSize := numRows

	// Initialize gradients
	if ln.Gamma.Grad == nil {
		ln.Gamma.Grad = NewTensor(ln.Gamma.Shape, make([]float32, len(ln.Gamma.Data)), false)
	}
	if ln.Beta.Grad == nil {
		ln.Beta.Grad = NewTensor(ln.Beta.Shape, make([]float32, len(ln.Beta.Data)), false)
	}
	if ln.input.Grad == nil {
		ln.input.Grad = NewTensor(ln.input.Shape, make([]float32, len(ln.input.Data)), false)
	}

	// Gradient w.r.t. gamma and beta
	for i := range batchSize {
		start := i * ln.NormalizedShape
		gOut := gradOutput.Data[start : start+ln.NormalizedShape]
		norm := ln.normalized.Data[start : start+ln.NormalizedShape]
		
		MulAccumulate(ln.Gamma.Grad.Data, gOut, norm)
		AddAccumulate(ln.Beta.Grad.Data, gOut)
	}

	// Gradient w.r.t. input (simplified version)
	tmp := make([]float32, ln.NormalizedShape)
	for i := range batchSize {
		start := i * ln.NormalizedShape
		std := float32(math.Sqrt(float64(ln.variance.Data[i] + ln.Eps)))
		gOut := gradOutput.Data[start : start+ln.NormalizedShape]
		
		MulVectors(gOut, ln.Gamma.Data, tmp)
		MulScalar(tmp, 1.0/std, tmp)
		AddAccumulate(ln.input.Grad.Data[start:start+ln.NormalizedShape], tmp)
	}

	return nil
}

// Input returns the input tensor of the LayerNorm operation
func (ln *LayerNorm) Input() *Tensor {
	return ln.input
}

// Parameters returns the learnable parameters of LayerNorm
func (ln *LayerNorm) Parameters() []*Tensor {
	return []*Tensor{ln.Gamma, ln.Beta}
}

// ToGPU moves the parameters to the GPU.
func (ln *LayerNorm) ToGPU() {
	if ln.Gamma != nil {
		ln.Gamma.ToGPU()
	}
	if ln.Beta != nil {
		ln.Beta.ToGPU()
	}
}

// ClearState clears the intermediate tensors used for backward pass
func (ln *LayerNorm) ClearState() {
	ln.input = nil
	ln.normalized = nil
	ln.mean = nil
	ln.variance = nil
}
