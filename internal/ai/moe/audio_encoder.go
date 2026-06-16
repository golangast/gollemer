package moe

import (
	"math"
	"math/rand"
)

// AudioEncoder projects raw audio frames (e.g., chunks of PCM samples)
// into a continuous vector space of dimension dModel.
// This is the audio equivalent of the VisionEncoder.
type AudioEncoder struct {
	InputDim int // Number of samples per frame (e.g., 400 samples for 25ms at 16kHz)
	DModel   int // Output dimension (matches the MoE or TemporalEncoder input dimension)
	Weights  []float32 // [InputDim * DModel]
}

// NewAudioEncoder initializes a new AudioEncoder with random Xavier weights.
func NewAudioEncoder(inputDim, dModel int) *AudioEncoder {
	ae := &AudioEncoder{
		InputDim: inputDim,
		DModel:   dModel,
		Weights:  make([]float32, inputDim*dModel),
	}

	// Xavier initialization
	limit := float32(1.0 / float32(inputDim))
	for i := range ae.Weights {
		ae.Weights[i] = (rand.Float32()*2.0 - 1.0) * limit
	}

	return ae
}

// Forward linearly projects a sequence of raw audio frames into tokens.
func (ae *AudioEncoder) Forward(frames [][]float32) [][]float32 {
	if len(frames) == 0 {
		return nil
	}

	tokens := make([][]float32, len(frames))
	for i, frame := range frames {
		token := make([]float32, ae.DModel)
		for d := 0; d < ae.DModel; d++ {
			var sum float32
			for j := 0; j < ae.InputDim; j++ {
				sum += frame[j] * ae.Weights[j*ae.DModel+d]
			}
			token[d] = float32(math.Abs(float64(sum)))
		}
		tokens[i] = token
	}
	return tokens
}

// Backward computes gradients for the audio projection weights.
func (ae *AudioEncoder) Backward(frames [][]float32, dTokens [][]float32, lr float32) {
	for i, frame := range frames {
		dToken := dTokens[i]
		for d := 0; d < ae.DModel; d++ {
			// Recompute sum to get the sign
			var sum float32
			for j := 0; j < ae.InputDim; j++ {
				sum += frame[j] * ae.Weights[j*ae.DModel+d]
			}
			sign := float32(1.0)
			if sum < 0 {
				sign = -1.0
			}

			for j := 0; j < ae.InputDim; j++ {
				// Gradient of weight = input * dOut * sign
				grad := frame[j] * dToken[d] * sign
				ae.Weights[j*ae.DModel+d] -= lr * grad
			}
		}
	}
}

// Parameters returns a reference to the weights for serialization/synchronization.
func (ae *AudioEncoder) Parameters() []*[]float32 {
	return []*[]float32{&ae.Weights}
}
