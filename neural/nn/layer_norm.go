package nn

import (
	"fmt"
	"math"

	. "github.com/golangast/gollemer/neural/tensor"
)

// LayerNorm represents a layer normalization module
type LayerNorm struct {
	NormalizedShape int
	Gamma           *Tensor // Learnable scale parameter
	Beta            *Tensor // Learnable shift parameter
	Eps             float64

	// Stored for backward pass
	input      *Tensor
	normalized *Tensor
	mean       *Tensor
	variance   *Tensor
}

// NewLayerNorm creates a new LayerNorm module
func NewLayerNorm(normalizedShape int) *LayerNorm {
	// Initialize gamma to ones and beta to zeros
	gamma := NewTensor([]int{normalizedShape}, make([]float64, normalizedShape), true)
	beta := NewTensor([]int{normalizedShape}, make([]float64, normalizedShape), true)

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
	mean := NewTensor([]int{batchSize}, make([]float64, batchSize), false)
	variance := NewTensor([]int{batchSize}, make([]float64, batchSize), false)

	for i := range batchSize {
		start := i * ln.NormalizedShape
		data := input.Data[start : start+ln.NormalizedShape]
		
		m := SumVector(data) / float64(ln.NormalizedShape)
		mean.Data[i] = m

		// Variance: sum((x - m)^2) / n
		varSum := 0.0
		for _, val := range data {
			diff := val - m
			varSum += diff * diff
		}
		variance.Data[i] = varSum / float64(ln.NormalizedShape)
	}

	ln.mean = mean
	ln.variance = variance

	// Normalize, Scale and Shift in one pass if possible, or use optimized vectors
	normalized := NewTensor(input.Shape, make([]float64, len(input.Data)), input.RequiresGrad)
	output := NewTensor(input.Shape, make([]float64, len(input.Data)), input.RequiresGrad)
	
	for i := range batchSize {
		start := i * ln.NormalizedShape
		std := math.Sqrt(variance.Data[i] + ln.Eps)
		m := mean.Data[i]
		
		// Vectorized normalization
		normData := normalized.Data[start : start+ln.NormalizedShape]
		inData := input.Data[start : start+ln.NormalizedShape]
		
		// (x - m) / std
		AddScalar(inData, -m, normData)
		DivScalar(normData, std, normData)
		
		// Scale and shift: y = gamma * norm + beta
		outData := output.Data[start : start+ln.NormalizedShape]
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
		ln.Gamma.Grad = NewTensor(ln.Gamma.Shape, make([]float64, len(ln.Gamma.Data)), false)
	}
	if ln.Beta.Grad == nil {
		ln.Beta.Grad = NewTensor(ln.Beta.Shape, make([]float64, len(ln.Beta.Data)), false)
	}
	if ln.input.Grad == nil {
		ln.input.Grad = NewTensor(ln.input.Shape, make([]float64, len(ln.input.Data)), false)
	}

	// Gradient w.r.t. gamma and beta
	for i := range batchSize {
		for j := 0; j < ln.NormalizedShape; j++ {
			ln.Gamma.Grad.Data[j] += gradOutput.Data[i*ln.NormalizedShape+j] * ln.normalized.Data[i*ln.NormalizedShape+j]
			ln.Beta.Grad.Data[j] += gradOutput.Data[i*ln.NormalizedShape+j]
		}
	}

	// Gradient w.r.t. input (simplified version)
	for i := range batchSize {
		std := math.Sqrt(ln.variance.Data[i] + ln.Eps)
		for j := 0; j < ln.NormalizedShape; j++ {
			ln.input.Grad.Data[i*ln.NormalizedShape+j] += gradOutput.Data[i*ln.NormalizedShape+j] * ln.Gamma.Data[j] / std
		}
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
