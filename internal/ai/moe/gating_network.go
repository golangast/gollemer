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
	Linear               *nn.Linear
	NoiseLinear          *nn.Linear // For Noisy Top-K gating
	LayerNorm            *nn.LayerNorm
	Training             bool // training mode for noise injection
	DiversityCoefficient float32
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
func (gn *GatingNetwork) generateNoise(size int, stddev float32) []float32 {
	noise := make([]float32, size)
	for i := range noise {
		noise[i] = float32(rand.NormFloat64() * float64(stddev))
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

	// 🛡️ The Stability Floor (Prevent division by zero and Expert Collapse)
	inputNorm := input.L2Norm()
	if inputNorm < 1e-8 {
		// Inject a tiny bit of jitter to "wake up" the signal if it has collapsed
		input.AddJitter(1e-6)
		inputNorm = 1e-8
	}

	// Work with the stable, normalized signal for gating
	scaledInput := input
	if inputNorm != 1.0 {
		scaledInput = input.Scale(1.0 / inputNorm)
	}

	// Determine dimensions.
	numExperts := gn.Linear.Weights.Shape[1] // [inputDim, numExperts]
	inputDim := gn.Linear.Weights.Shape[0]

	var numTokens = 0
	var logitsShape []int
	var inputFlat []float32

	switch len(scaledInput.Shape) {
	case 2:
		// [batchSize, inputDim]
		numTokens = scaledInput.Shape[0]
		logitsShape = []int{numTokens, numExperts}
		inputFlat = scaledInput.Data
	case 3:
		// [batchSize, seqLength, inputDim]
		numTokens = scaledInput.Shape[0] * scaledInput.Shape[1]
		logitsShape = []int{scaledInput.Shape[0], scaledInput.Shape[1], numExperts}
		inputFlat = scaledInput.Data
	default:
		return nil, fmt.Errorf("GatingNetwork.Forward: unsupported input shape %v", input.Shape)
	}

	// Allocate output (logits before bias).
	logitsData := make([]float32, numTokens*numExperts)

	// ── SIMD-accelerated router logit computation ─────────────────────────
	computeRouterLogitsSIMD(
		inputFlat,
		gn.Linear.Weights.Data,
		numTokens, numExperts, inputDim,
		logitsData,
	)

	// 🛡️ Stability Hack: Scaling + Logit Clipping
	// Scaling by 1/sqrt(inputDim) keeps the variance of logits under control,
	// preventing 'Expert Monopolies' and large gradients during backprop.
	// We also clip to [-25, 25] to prevent softmax saturation.
	scaleFactor := float32(1.0 / math.Sqrt(float64(inputDim)))
	for i := range logitsData {
		logitsData[i] *= scaleFactor
		if logitsData[i] > 25.0 {
			logitsData[i] = 25.0
		} else if logitsData[i] < -25.0 {
			logitsData[i] = -25.0
		}
	}

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

	// 1. Gumbel-Softmax Noise for Expert Exploration (Break Expert Collusions)
	if gn.Training {
		// This forces the model to occasionally "miss" its favorite expert
		// and explore others.
		for i := range logitsData {
			// Zero-centered random noise ([-0.05, 0.05] magnitude)
			logitsData[i] += (rand.Float32() - 0.5) * 0.1
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
	var dWeightsOut []float32
	var dInputOut []float32
	
	if gn.Linear.Weights.RequiresGrad {
		if gn.Linear.Weights.Grad == nil {
			gn.Linear.Weights.Grad = tensor.NewTensor(gn.Linear.Weights.Shape, make([]float32, len(gn.Linear.Weights.Data)), false)
		}
		dWeightsOut = gn.Linear.Weights.Grad.Data
	} else {
		dWeightsOut = make([]float32, len(gn.Linear.Weights.Data)) // Discard
	}

	if gn.inputTensor.RequiresGrad {
		if gn.inputTensor.Grad == nil {
			gn.inputTensor.Grad = tensor.NewTensor(gn.inputTensor.Shape, make([]float32, len(gn.inputTensor.Data)), false)
		}
		dInputOut = gn.inputTensor.Grad.Data
	} else {
		dInputOut = make([]float32, len(gn.inputTensor.Data)) // Discard
	}

	// Accumulate weight and input gradients using SIMD.
	// Apply 1/sqrt(inputDim) scaling factor to match forward pass
	scaleFactor := float32(1.0 / math.Sqrt(float64(inputDim)))
 
	computeRouterGradSIMD(
		gn.inputTensor.Data,
		gn.Linear.Weights.Data,
		lnGrad.Data,
		dWeightsOut,
		dInputOut,
		numTokens, numExperts, inputDim,
		scaleFactor,
	)

	// Bias gradient
	if gn.Linear.Biases != nil && gn.Linear.Biases.RequiresGrad {
		if gn.Linear.Biases.Grad == nil {
			gn.Linear.Biases.Grad = tensor.NewTensor(gn.Linear.Biases.Shape, make([]float32, len(gn.Linear.Biases.Data)), false)
		}
		for t := 0; t < numTokens; t++ {
			base := t * numExperts
			for e := 0; e < numExperts; e++ {
				// Apply scaleFactor to bias as well
				gn.Linear.Biases.Grad.Data[e] += lnGrad.Data[base+e] * scaleFactor
			}
		}
	}

	// 🛡️ LOCAL STABILITY: Clip router gradients to prevent explosions
	// from propagating to the global optimizer.
	const localClipThreshold = 25.0
	if gn.Linear.Weights.RequiresGrad && gn.Linear.Weights.Grad != nil {
		gn.Linear.Weights.Grad.ClipGrad(localClipThreshold)
	}
	if gn.Linear.Biases != nil && gn.Linear.Biases.RequiresGrad && gn.Linear.Biases.Grad != nil {
		gn.Linear.Biases.Grad.ClipGrad(localClipThreshold)
	}

	// 3. Backward through NoiseLinear (Approximation: using same gradient as main linear)
	if gn.Training && gn.noiseLogitsTensor != nil {
		var dnWeightsOut []float32
		if gn.NoiseLinear.Weights.RequiresGrad {
			if gn.NoiseLinear.Weights.Grad == nil {
				gn.NoiseLinear.Weights.Grad = tensor.NewTensor(gn.NoiseLinear.Weights.Shape, make([]float32, len(gn.NoiseLinear.Weights.Data)), false)
			}
			dnWeightsOut = gn.NoiseLinear.Weights.Grad.Data
		} else {
			dnWeightsOut = make([]float32, len(gn.NoiseLinear.Weights.Data))
		}

		// Simplified: Noise weights also learn from the gating gradient to control exploration magnitude
		computeRouterGradSIMD(
			gn.inputTensor.Data,
			gn.NoiseLinear.Weights.Data,
			lnGrad.Data,
			dnWeightsOut,
			nil, // already computed input grad above
			numTokens, numExperts, inputDim,
			scaleFactor,
		)

		if gn.NoiseLinear.Biases != nil && gn.NoiseLinear.Biases.RequiresGrad {
			if gn.NoiseLinear.Biases.Grad == nil {
				gn.NoiseLinear.Biases.Grad = tensor.NewTensor(gn.NoiseLinear.Biases.Shape, make([]float32, len(gn.NoiseLinear.Biases.Data)), false)
			}
			for t := 0; t < numTokens; t++ {
				base := t * numExperts
				for e := 0; e < numExperts; e++ {
					gn.NoiseLinear.Biases.Grad.Data[e] += lnGrad.Data[base+e] * scaleFactor
				}
			}
		}

		// Local clip for noise linear
		if gn.NoiseLinear.Weights.RequiresGrad && gn.NoiseLinear.Weights.Grad != nil {
			gn.NoiseLinear.Weights.Grad.ClipGrad(localClipThreshold)
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

// CalculateDiversityLoss calculates the load balancing penalty based on routing distribution.
// It punishes the network if it starts relying too heavily on a few experts (Alpha Dominance).
func (gn *GatingNetwork) CalculateDiversityLoss() float32 {
	if gn.outputTensor == nil {
		return 0
	}

	numTokens := gn.outputTensor.Shape[0]
	if len(gn.outputTensor.Shape) > 2 {
		numTokens *= gn.outputTensor.Shape[1]
	}
	numExperts := gn.outputTensor.Shape[len(gn.outputTensor.Shape)-1]

	// 1. Calculate the average usage of each expert across the batch
	avgUsage := make([]float32, numExperts)
	for t := 0; t < numTokens; t++ {
		base := t * numExperts
		for e := 0; e < numExperts; e++ {
			avgUsage[e] += gn.outputTensor.Data[base+e]
		}
	}

	// 2. Normalize by token count
	var totalLoss float32
	targetUsage := float32(1.0 / float32(numExperts))

	for e := 0; e < numExperts; e++ {
		avgUsage[e] /= float32(numTokens)

		// 3. Penalty = (Actual Usage - Target Usage)^2
		diff := avgUsage[e] - targetUsage
		totalLoss += diff * diff
	}

	coeff := gn.DiversityCoefficient
	if coeff == 0 {
		coeff = 0.25 // Default "Anti-Lazy" coefficient
	}
	return totalLoss * coeff
}

// Inputs returns the input tensors of the GatingNetwork's last forward operation.
func (gn *GatingNetwork) Inputs() []*tensor.Tensor {
	if gn.inputTensor != nil {
		return []*tensor.Tensor{gn.inputTensor}
	}
	return []*tensor.Tensor{}
}

// ToGPU moves the parameters to the GPU.
func (gn *GatingNetwork) ToGPU() {
	if gn.Linear != nil {
		gn.Linear.ToGPU()
	}
	if gn.NoiseLinear != nil {
		gn.NoiseLinear.ToGPU()
	}
	if gn.LayerNorm != nil {
		gn.LayerNorm.ToGPU()
	}
}

func (gn *GatingNetwork) SyncParameters() error {
	// GatingNetwork uses standard nn.Linear which currently targets direct GPU execution
	// or CPU->GPU sync via ToGPU, so SyncParameters is a no-op for now.
	return nil
}
