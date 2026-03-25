package moe

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// GatingNetwork (Router) determines which experts to activate for a given input.
type GatingNetwork struct {
	Linear       *nn.Linear
	NoiseLinear  *nn.Linear // For Noisy Top-K gating
	LayerNorm    *nn.LayerNorm
	Training     bool // training mode for noise injection
	// Stored for backward pass
	inputTensor       *tensor.Tensor
	logitsTensor      *tensor.Tensor
	noiseLogitsTensor *tensor.Tensor
	outputTensor      *tensor.Tensor
}

// NewGatingNetwork creates a new GatingNetwork.
func NewGatingNetwork(inputDim, numExperts int) (*GatingNetwork, error) {
	linear, err := nn.NewLinear(inputDim, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create linear layer for gating network: %w", err)
	}
	noiseLinear, err := nn.NewLinear(inputDim, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create noise linear layer: %w", err)
	}
	ln := nn.NewLayerNorm(numExperts)
	
	// Set IsRouter flag for differential learning rates
	linear.Weights.IsRouter = true
	if linear.Biases != nil {
		linear.Biases.IsRouter = true
	}
	noiseLinear.Weights.IsRouter = true
	if noiseLinear.Biases != nil {
		noiseLinear.Biases.IsRouter = true
	}
	
	return &GatingNetwork{Linear: linear, NoiseLinear: noiseLinear, LayerNorm: ln}, nil
}

// generateNoise creates Gaussian noise for exploration.
func (gn *GatingNetwork) generateNoise(size int, stddev float64) []float64 {
	noise := make([]float64, size)
	for i := range noise {
		noise[i] = rand.NormFloat64() * stddev
	}
	return noise
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

	// --- [Noisy Top-K Gating Implementation] ---
	var noiseLogitsData []float64
	if gn.Training {
		// Lazy initialize NoiseLinear for models loaded from older checkpoints
		if gn.NoiseLinear == nil {
			nl, err := nn.NewLinear(inputDim, numExperts)
			if err == nil {
				gn.NoiseLinear = nl
			}
		}

		if gn.NoiseLinear != nil {
			noiseLogitsData = make([]float64, numTokens*numExperts)
			computeRouterLogitsSIMD(
				inputFlat,
				gn.NoiseLinear.Weights.Data,
				numTokens, numExperts, inputDim,
				noiseLogitsData,
			)
			if gn.NoiseLinear.Biases != nil {
				biasData := gn.NoiseLinear.Biases.Data
				for t := 0; t < numTokens; t++ {
					base := t * numExperts
					for e := 0; e < numExperts; e++ {
						noiseLogitsData[base+e] += biasData[e]
					}
				}
			}

			// Inject Gaussian noise scaled by softplus(noiseLogits)
			for i := range logitsData {
				// Softplus: ln(1 + e^x) to keep noise magnitude positive
				x := noiseLogitsData[i]
				var sigma float64
				if x > 20 {
					sigma = x // Avoid exp overflow
				} else {
					sigma = math.Log(1.0 + math.Exp(x))
				}
				
				// N(0, 1) * sigma
				logitsData[i] += rand.NormFloat64() * sigma
			}
			
			gn.noiseLogitsTensor = tensor.NewTensor(logitsShape, noiseLogitsData, input.RequiresGrad || gn.NoiseLinear.Weights.RequiresGrad)
		} else {
			// Fallback: simple fixed noise if NoiseLinear failed to init
			noise := gn.generateNoise(len(logitsData), 0.02)
			for i := range logitsData {
				logitsData[i] += noise[i]
			}
		}
	}

	logits := tensor.NewTensor(logitsShape, logitsData, input.RequiresGrad || gn.Linear.Weights.RequiresGrad)
	gn.logitsTensor = logits

	// Apply LayerNorm for stability
	normalized, err := gn.LayerNorm.Forward(logits)
	if err != nil {
		return nil, fmt.Errorf("layer norm forward failed: %w", err)
	}

	if normalized.RequiresGrad {
		normalized.Creator = gn
	}

	gn.outputTensor = normalized
	return normalized, nil
}

func (gn *GatingNetwork) Backward(grad *tensor.Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// 1. Backward through LayerNorm
	err := gn.LayerNorm.Backward(grad)
	if err != nil {
		return fmt.Errorf("layer norm backward failed: %w", err)
	}
	
	lnGrad := gn.logitsTensor.Grad // Gradient from LayerNorm
	if lnGrad == nil {
		return nil
	}

	// 2. Backward through Linear (using SIMD)
	numExperts := gn.Linear.Weights.Shape[1]
	inputDim := gn.Linear.Weights.Shape[0]

	numTokens := 1
	for _, s := range gn.inputTensor.Shape[:len(gn.inputTensor.Shape)-1] {
		numTokens *= s
	}

	// Ensure weight and input gradient tensors exist.
	var dWeightsOut []float64
	var dInputOut []float64

	if gn.Linear.Weights.RequiresGrad {
		if gn.Linear.Weights.Grad == nil {
			gn.Linear.Weights.Grad = tensor.NewTensor(gn.Linear.Weights.Shape, make([]float64, len(gn.Linear.Weights.Data)), false)
		}
		dWeightsOut = gn.Linear.Weights.Grad.Data
	} else {
		dWeightsOut = make([]float64, len(gn.Linear.Weights.Data)) // Discard
	}

	if gn.inputTensor.RequiresGrad {
		if gn.inputTensor.Grad == nil {
			gn.inputTensor.Grad = tensor.NewTensor(gn.inputTensor.Shape, make([]float64, len(gn.inputTensor.Data)), false)
		}
		dInputOut = gn.inputTensor.Grad.Data
	} else {
		dInputOut = make([]float64, len(gn.inputTensor.Data)) // Discard
	}

	// Accumulate weight and input gradients using SIMD.
	computeRouterGradSIMD(
		gn.inputTensor.Data,
		gn.Linear.Weights.Data,
		lnGrad.Data,
		dWeightsOut,
		dInputOut,
		numTokens, numExperts, inputDim,
	)

	// Bias gradient
	if gn.Linear.Biases != nil && gn.Linear.Biases.RequiresGrad {
		if gn.Linear.Biases.Grad == nil {
			gn.Linear.Biases.Grad = tensor.NewTensor(gn.Linear.Biases.Shape, make([]float64, len(gn.Linear.Biases.Data)), false)
		}
		for t := 0; t < numTokens; t++ {
			base := t * numExperts
			for e := 0; e < numExperts; e++ {
				gn.Linear.Biases.Grad.Data[e] += lnGrad.Data[base+e]
			}
		}
	}

	// 3. Backward through NoiseLinear (Approximation: using same gradient as main linear)
	if gn.Training && gn.noiseLogitsTensor != nil {
		var dnWeightsOut []float64
		if gn.NoiseLinear.Weights.RequiresGrad {
			if gn.NoiseLinear.Weights.Grad == nil {
				gn.NoiseLinear.Weights.Grad = tensor.NewTensor(gn.NoiseLinear.Weights.Shape, make([]float64, len(gn.NoiseLinear.Weights.Data)), false)
			}
			dnWeightsOut = gn.NoiseLinear.Weights.Grad.Data
		} else {
			dnWeightsOut = make([]float64, len(gn.NoiseLinear.Weights.Data))
		}

		// Simplified: Noise weights also learn from the gating gradient to control exploration magnitude
		computeRouterGradSIMD(
			gn.inputTensor.Data,
			gn.NoiseLinear.Weights.Data,
			lnGrad.Data,
			dnWeightsOut,
			make([]float64, len(gn.inputTensor.Data)), // already computed input grad above
			numTokens, numExperts, inputDim,
		)

		if gn.NoiseLinear.Biases != nil && gn.NoiseLinear.Biases.RequiresGrad {
			if gn.NoiseLinear.Biases.Grad == nil {
				gn.NoiseLinear.Biases.Grad = tensor.NewTensor(gn.NoiseLinear.Biases.Shape, make([]float64, len(gn.NoiseLinear.Biases.Data)), false)
			}
			for t := 0; t < numTokens; t++ {
				base := t * numExperts
				for e := 0; e < numExperts; e++ {
					gn.NoiseLinear.Biases.Grad.Data[e] += lnGrad.Data[base+e]
				}
			}
		}
	}

	// Clear intermediate gradients
	gn.logitsTensor.Grad = nil
	return nil
}

// Parameters returns all learnable parameters of the GatingNetwork.
func (gn *GatingNetwork) Parameters() []*tensor.Tensor {
	params := gn.Linear.Parameters()
	if gn.NoiseLinear != nil {
		params = append(params, gn.NoiseLinear.Parameters()...)
	}
	params = append(params, gn.LayerNorm.Parameters()...)
	return params
}

// Inputs returns the input tensors of the GatingNetwork's last forward operation.
func (gn *GatingNetwork) Inputs() []*tensor.Tensor {
	if gn.inputTensor != nil {
		return []*tensor.Tensor{gn.inputTensor}
	}
	return []*tensor.Tensor{}
}
