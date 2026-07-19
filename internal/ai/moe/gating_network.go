package moe

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// RouterNoiseFactor controls the magnitude of random noise added during routing.
// Defaulted to 0.0 for Overfit Strategy (memorization test).
var RouterNoiseFactor float32 = 0.0

// SetRouterNoiseFactor updates the global router noise magnitude.
func SetRouterNoiseFactor(v float32) {
	RouterNoiseFactor = v
}

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

	// 🛡️ The Stability Floor
	inputNorm := input.L2Norm()
	if inputNorm < 1e-8 {
		input.AddJitter(1e-6)
		inputNorm = 1e-8
	}

	scaledInput := input
	if inputNorm != 1.0 {
		scaledInput = input.Scale(1.0 / inputNorm)
	}

	numExperts := gn.Linear.Weights.Shape[1]
	inputDim := gn.Linear.Weights.Shape[0]

	var numTokens = 0
	var logitsShape []int
	var inputFlat []float32

	switch len(scaledInput.Shape) {
	case 2:
		numTokens = scaledInput.Shape[0]
		logitsShape = []int{numTokens, numExperts}
		inputFlat = scaledInput.Data
	case 3:
		numTokens = scaledInput.Shape[0] * scaledInput.Shape[1]
		logitsShape = []int{scaledInput.Shape[0], scaledInput.Shape[1], numExperts}
		inputFlat = scaledInput.Data
	default:
		return nil, fmt.Errorf("GatingNetwork.Forward: unsupported input shape %v", input.Shape)
	}

	// 1. Compute Base Logits
	logitsData := make([]float32, numTokens*numExperts)
	computeRouterLogitsSIMD(inputFlat, gn.Linear.Weights.Data, numTokens, numExperts, inputDim, logitsData)

	// Apply Scaling & Clipping
	scaleFactor := float32(1.0 / math.Sqrt(float64(inputDim)))
	for i := range logitsData {
		logitsData[i] *= scaleFactor
	}

	// Add Bias
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
	gn.logitsTensor = logits

	// 2. NOISY TOP-K: Compute Noise Scale via NoiseLinear
	if gn.Training && gn.NoiseLinear != nil {
		noiseLogitsData := make([]float32, numTokens*numExperts)
		computeRouterLogitsSIMD(inputFlat, gn.NoiseLinear.Weights.Data, numTokens, numExperts, inputDim, noiseLogitsData)

		// noise_scale = Softplus(noise_logits)
		// We approximate Softplus here for performance
		for i := range noiseLogitsData {
			v := float64(noiseLogitsData[i])
			if v > 20 {
				noiseLogitsData[i] = float32(v)
			} else {
				noiseLogitsData[i] = float32(math.Log(1.0 + math.Exp(v)))
			}
		}

		gn.noiseLogitsTensor = tensor.NewTensor(logitsShape, noiseLogitsData, gn.NoiseLinear.Weights.RequiresGrad)

		// logits = logits + StandardNormal() * noise_scale * RouterNoiseFactor
		for i := range logits.Data {
			noise := float32(rand.NormFloat64())
			logits.Data[i] += noise * gn.noiseLogitsTensor.Data[i] * RouterNoiseFactor
		}
	}

	// Apply LayerNorm for stability
	normalized, err := gn.LayerNorm.Forward(logits)
	if err != nil {
		return nil, fmt.Errorf("layer norm forward failed: %w", err)
	}

	if normalized.RequiresGrad {
		normalized.Creator = gn
	}
	gn.outputTensor = normalized

	if scaledInput != nil && scaledInput != input {
		scaledInput.Release()
	}

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
// CalculateCVLoss implements the Coefficient of Variation load balancing loss.
// It penalizes the variance of expert utilization across the batch.
// CV = std(load) / mean(load)
func (gn *GatingNetwork) CalculateCVLoss() float32 {
	if gn.outputTensor == nil {
		return 0
	}

	numTokens := gn.outputTensor.Shape[0]
	if len(gn.outputTensor.Shape) > 2 {
		numTokens *= gn.outputTensor.Shape[1]
	}
	numExperts := gn.outputTensor.Shape[len(gn.outputTensor.Shape)-1]

	if numTokens <= 0 {
		return 0
	}

	probs, err := gn.outputTensor.Softmax(len(gn.outputTensor.Shape) - 1)
	if err != nil {
		return 0
	}

	// 1. Calculate the average probability assigned to each expert (Load)
	loads := make([]float64, numExperts)
	for t := 0; t < numTokens; t++ {
		base := t * numExperts
		for e := 0; e < numExperts; e++ {
			loads[e] += float64(probs.Data[base+e])
		}
	}

	// 2. Compute Mean
	var mean float64
	for _, l := range loads {
		mean += l
	}
	mean /= float64(numExperts)

	if mean < 1e-6 {
		return 0
	}

	// 3. Compute Variance
	var variance float64
	for _, l := range loads {
		diff := l - mean
		variance += diff * diff
	}
	variance /= float64(numExperts)

	// 4. CV^2 = variance / (mean^2)
	cvSquared := variance / (mean * mean)

	// Weight the CV loss
	coeff := gn.DiversityCoefficient
	if coeff == 0 {
		coeff = 0.5
	}

	res := float32(cvSquared) * coeff
	if math.IsNaN(float64(res)) || math.IsInf(float64(res), 0) {
		return 0.0
	}
	return res
}

func (gn *GatingNetwork) CalculateDiversityLoss() float32 {
	// Transitioning to CV Loss as the primary load balancer
	return gn.CalculateCVLoss()
}

// CalculateGatingEntropy computes the entropy of the expert selection distribution.
// Higher entropy means the router is more uncertain/diverse.
func (gn *GatingNetwork) CalculateGatingEntropy() float32 {
	if gn.outputTensor == nil {
		return 0
	}

	numTokens := gn.outputTensor.Shape[0]
	if len(gn.outputTensor.Shape) > 2 {
		numTokens *= gn.outputTensor.Shape[1]
	}
	numExperts := gn.outputTensor.Shape[len(gn.outputTensor.Shape)-1]

	if numTokens <= 0 {
		return 0
	}

	probs, err := gn.outputTensor.Softmax(len(gn.outputTensor.Shape) - 1)
	if err != nil {
		return 0
	}

	var totalEntropy float64
	for t := 0; t < numTokens; t++ {
		base := t * numExperts
		var tokenEntropy float64
		for e := 0; e < numExperts; e++ {
			p := float64(probs.Data[base+e])
			if p > 1e-10 {
				tokenEntropy -= p * math.Log(p)
			}
		}
		totalEntropy += tokenEntropy
	}

	res := float32(totalEntropy / float64(numTokens))
	if math.IsNaN(float64(res)) || math.IsInf(float64(res), 0) {
		return 0.0
	}
	return res
}

// ClearState clears all intermediate tensors.
func (gn *GatingNetwork) ClearState() {
	gn.inputTensor = nil
	gn.logitsTensor = nil
	gn.noiseLogitsTensor = nil
	gn.outputTensor = nil

	if gn.Linear != nil {
		gn.Linear.ClearState()
	}
	if gn.NoiseLinear != nil {
		gn.NoiseLinear.ClearState()
	}
	if gn.LayerNorm != nil {
		gn.LayerNorm.ClearState()
	}
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

func (gn *GatingNetwork) RepairArchitecture() {
	if gn.Linear == nil || gn.Linear.Weights == nil {
		return
	}
	numExperts := gn.Linear.Weights.Shape[1]
	if gn.LayerNorm == nil || gn.LayerNorm.NormalizedShape != numExperts {
		gn.LayerNorm = nn.NewLayerNorm(numExperts)
		// 🛡️ DEVICE SYNC: If the gating network is on GPU, the new LayerNorm must be too.
		if gn.Linear.Weights.Device == tensor.GPU {
			gn.LayerNorm.ToGPU()
		}
	}
}

func (gn *GatingNetwork) SyncParameters() error {
	// GatingNetwork uses standard nn.Linear which currently targets direct GPU execution
	// or CPU->GPU sync via ToGPU, so SyncParameters is a no-op for now.
	return nil
}

// PruneExperts shrinks the gating network's projection matrices by dropping the columns
// associated with the pruned expert indices.
func (gn *GatingNetwork) PruneExperts(droppedIndices map[int]bool) {
	shrinkLinear := func(lin *nn.Linear) {
		if lin == nil || lin.Weights == nil {
			return
		}
		oldW := lin.Weights
		inputDim := oldW.Shape[0]
		oldN := oldW.Shape[1]
		newN := oldN - len(droppedIndices)
		if newN <= 0 {
			return
		}

		newWData := make([]float32, inputDim*newN)
		for row := 0; row < inputDim; row++ {
			newCol := 0
			for col := 0; col < oldN; col++ {
				if !droppedIndices[col] {
					newWData[row*newN+newCol] = oldW.Data[row*oldN+col]
					newCol++
				}
			}
		}
		lin.Weights = tensor.NewTensor([]int{inputDim, newN}, newWData, true)

		if lin.Biases != nil {
			oldB := lin.Biases.Data
			newBData := make([]float32, newN)
			newCol := 0
			for col := 0; col < oldN; col++ {
				if !droppedIndices[col] {
					newBData[newCol] = oldB[col]
					newCol++
				}
			}
			lin.Biases = tensor.NewTensor([]int{newN}, newBData, true)
		}
	}

	shrinkLinear(gn.Linear)
	shrinkLinear(gn.NoiseLinear)

	// LayerNorm needs to be rebuilt since the hidden dimension changed
	if gn.LayerNorm != nil && gn.Linear != nil {
		newN := gn.Linear.Weights.Shape[1]
		gn.LayerNorm = nn.NewLayerNorm(newN)
	}
}
