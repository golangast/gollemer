package moe

import (
	"fmt"

	"github.com/golangast/gollemer/neural/nn"
	"github.com/golangast/gollemer/neural/tensor"
)

// GatingNetwork (Router) determines which experts to activate for a given input.
type GatingNetwork struct {
	Linear *nn.Linear
	// Stored for backward pass
	inputTensor  *tensor.Tensor
	outputTensor *tensor.Tensor
}

// NewGatingNetwork creates a new GatingNetwork.
// inputDim is the dimension of the input to the gating network.
// numExperts is the number of experts in the MoE layer.
func NewGatingNetwork(inputDim, numExperts int) (*GatingNetwork, error) {
	linear, err := nn.NewLinear(inputDim, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create linear layer for gating network: %w", err)
	}
	return &GatingNetwork{Linear: linear},
		nil
}

// Forward performs the forward pass of the GatingNetwork.
// It computes router logits via SIMD-accelerated dot products between the
// (flattened) input tokens and each expert's routing weight vector, then adds
// the bias.  The result is a tensor of shape [batchSize, seqLength, numExperts]
// (matching the existing nn.Linear output shape).
func (gn *GatingNetwork) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GatingNetwork.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	gn.inputTensor = input

	// Determine dimensions.
	numExperts := gn.Linear.Weights.Shape[1] // [inputDim, numExperts]
	inputDim := gn.Linear.Weights.Shape[0]

	var numTokens int
	var logitsShape []int
	var inputFlat []float64

	switch len(input.Shape) {
	case 2:
		// [batchSize, inputDim]
		numTokens = input.Shape[0]
		logitsShape = []int{numTokens, numExperts}
		inputFlat = input.Data
	case 3:
		// [batchSize, seqLength, inputDim]
		numTokens = input.Shape[0] * input.Shape[1]
		logitsShape = []int{input.Shape[0], input.Shape[1], numExperts}
		inputFlat = input.Data
	default:
		return nil, fmt.Errorf("GatingNetwork.Forward: unsupported input shape %v", input.Shape)
	}

	// Allocate output (logits before bias).
	logitsData := make([]float64, numTokens*numExperts)

	// ── SIMD-accelerated router logit computation ─────────────────────────
	// computeRouterLogitsSIMD is defined in simd_ops.go (GOEXPERIMENT=simd)
	// or simd_ops_fallback.go (pure-Go). It computes, for every (token, expert):
	//   logit[token][expert] = dot(input[token], W[:][expert])
	// where W is the [inputDim × numExperts] weight matrix stored column-major.
	computeRouterLogitsSIMD(
		inputFlat,
		gn.Linear.Weights.Data,
		numTokens, numExperts, inputDim,
		logitsData,
	)

	// Add bias (broadcast over tokens).
	if gn.Linear.Biases != nil {
		biasData := gn.Linear.Biases.Data
		for t := 0; t < numTokens; t++ {
			base := t * numExperts
			for e := 0; e < numExperts; e++ {
				logitsData[base+e] += biasData[e]
			}
		}
	}

	logits := tensor.NewTensor(logitsShape, logitsData, input.RequiresGrad || gn.Linear.Weights.RequiresGrad)
	if logits.RequiresGrad {
		logits.Creator = gn
	}

	gn.outputTensor = logits
	return logits, nil
}

// Backward performs the backward pass for the GatingNetwork.
// It uses computeRouterGradSIMD to accumulate weight and input gradients in a
// SIMD-vectorised manner (or the pure-Go fallback when SIMD is unavailable).
func (gn *GatingNetwork) Backward(grad *tensor.Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	if gn.outputTensor.Grad == nil {
		gn.outputTensor.Grad = tensor.NewTensor(grad.Shape, make([]float64, len(grad.Data)), false)
	}
	// Accumulate the incoming gradient onto the stored output gradient.
	for i := range grad.Data {
		gn.outputTensor.Grad.Data[i] += grad.Data[i]
	}

	// ── Backpropagate through the SIMD router logit computation ──────────
	// We need:
	//   dW[k][e]     += input[token][k] * grad[token][e]   (weight gradient)
	//   dInput[token][k] += W[k][e] * grad[token][e]       (input gradient)

	numExperts := gn.Linear.Weights.Shape[1]
	inputDim := gn.Linear.Weights.Shape[0]

	var numTokens int
	switch len(gn.inputTensor.Shape) {
	case 2:
		numTokens = gn.inputTensor.Shape[0]
	case 3:
		numTokens = gn.inputTensor.Shape[0] * gn.inputTensor.Shape[1]
	}

	// Ensure weight and input gradient tensors exist.
	if gn.Linear.Weights.RequiresGrad {
		if gn.Linear.Weights.Grad == nil {
			gn.Linear.Weights.Grad = tensor.NewTensor(
				gn.Linear.Weights.Shape,
				make([]float64, len(gn.Linear.Weights.Data)),
				false,
			)
		}
	}
	if gn.inputTensor.RequiresGrad {
		if gn.inputTensor.Grad == nil {
			gn.inputTensor.Grad = tensor.NewTensor(
				gn.inputTensor.Shape,
				make([]float64, len(gn.inputTensor.Data)),
				false,
			)
		}
	}

	// Accumulate weight gradients.
	if gn.Linear.Weights.RequiresGrad {
		computeRouterGradSIMD(
			gn.inputTensor.Data,
			gn.Linear.Weights.Data,
			gn.outputTensor.Grad.Data,
			gn.Linear.Weights.Grad.Data,
			gn.inputTensor.Grad.Data, // may be nil-safe because we pre-allocated above
			numTokens, numExperts, inputDim,
		)
	}

	// Bias gradient: sum over all tokens for each expert.
	if gn.Linear.Biases != nil && gn.Linear.Biases.RequiresGrad {
		if gn.Linear.Biases.Grad == nil {
			gn.Linear.Biases.Grad = tensor.NewTensor(
				gn.Linear.Biases.Shape,
				make([]float64, len(gn.Linear.Biases.Data)),
				false,
			)
		}
		for t := 0; t < numTokens; t++ {
			base := t * numExperts
			for e := 0; e < numExperts; e++ {
				gn.Linear.Biases.Grad.Data[e] += gn.outputTensor.Grad.Data[base+e]
			}
		}
	}

	// Clear output gradient (consumed).
	if gn.outputTensor != nil {
		gn.outputTensor.Grad = nil
	}

	return nil
}

// Parameters returns all learnable parameters of the GatingNetwork.
func (gn *GatingNetwork) Parameters() []*tensor.Tensor {
	return gn.Linear.Parameters()
}

// Inputs returns the input tensors of the GatingNetwork's last forward operation.
func (gn *GatingNetwork) Inputs() []*tensor.Tensor {
	if gn.inputTensor != nil {
		return []*tensor.Tensor{gn.inputTensor}
	}
	return []*tensor.Tensor{}
}
