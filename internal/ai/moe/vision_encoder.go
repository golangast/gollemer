package moe

import (
	"math/rand"
)

// VisionEncoder cleanly isolates the visual projection layer from the NLP code.
// This allows Gollemer to process image/video sequences without slowing down pure-text training.
type VisionEncoder struct {
	Weights []float32 // Size: patchDim * dModel
	Grads   []float32 // Size: patchDim * dModel

	PatchDim int // e.g., 256 (16x16 luma patch)
	DModel   int // e.g., 512 (Gollemer's hidden state size)
}

// NewVisionEncoder initializes a new VisionEncoder with trainable weights.
func NewVisionEncoder(patchDim, dModel int) *VisionEncoder {
	ve := &VisionEncoder{
		Weights:  make([]float32, patchDim*dModel),
		Grads:    make([]float32, patchDim*dModel),
		PatchDim: patchDim,
		DModel:   dModel,
	}

	// Xavier/Glorot Initialization for stable training
	limit := float32(1.0 / float32(patchDim))
	for i := range ve.Weights {
		ve.Weights[i] = rand.Float32()*(limit*2) - limit
	}

	return ve
}

// Forward projects the raw 2D patch data into sequence tokens of size DModel.
// This replaces the dummy weights we were using in the capture script!
func (ve *VisionEncoder) Forward(patches [][]float32) [][]float32 {
	numPatches := len(patches)
	if numPatches == 0 {
		return nil // Zero overhead for pure-text batches!
	}

	tokens := make([][]float32, numPatches)
	for i := 0; i < numPatches; i++ {
		token := make([]float32, ve.DModel)
		patch := patches[i]

		for d := 0; d < ve.DModel; d++ {
			var sum float32 = 0.0
			for j := 0; j < ve.PatchDim; j++ {
				// Row-major traversal through the 1D Weights slice
				sum += patch[j] * ve.Weights[j*ve.DModel+d]
			}
			token[d] = sum
		}
		tokens[i] = token
	}
	return tokens
}

// Backward computes gradients and updates the weights using standard SGD.
// gradOut is the error gradient flowing backwards from the MoE attention layers.
func (ve *VisionEncoder) Backward(gradOut [][]float32, patches [][]float32, learningRate float32) {
	if len(patches) == 0 || len(gradOut) == 0 {
		return // No vision input in this batch, skip calculation completely! (Zero overhead)
	}

	// 1. Reset gradients for this batch
	for i := range ve.Grads {
		ve.Grads[i] = 0.0
	}

	numPatches := len(patches)

	// 2. Accumulate gradients for the Weights (Matrix Math: Grads = Patches(T) * gradOut)
	for i := 0; i < numPatches; i++ {
		patch := patches[i]
		gOut := gradOut[i] // Error for this specific patch token

		for j := 0; j < ve.PatchDim; j++ {
			pVal := patch[j]
			for d := 0; d < ve.DModel; d++ {
				ve.Grads[j*ve.DModel+d] += pVal * gOut[d]
			}
		}
	}

	// 3. Apply SGD optimization to learn!
	for i := range ve.Weights {
		ve.Weights[i] -= learningRate * ve.Grads[i]
	}
}
