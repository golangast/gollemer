package moe

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/golangast/gollemer/neural/nn"
	. "github.com/golangast/gollemer/neural/tensor"
)

// ActiveLayers tracks all MoE layers created, useful for monitoring utilization.
var ActiveLayers []*MoELayer

// MoEState holds the state of a single forward pass for BPTT.
type MoEState struct {
	inputTensor        *Tensor
	input2D            *Tensor
	expertOutputs      []*Tensor
	expertTokenIndices [][]int
	selectedExperts    [][]int
	gateOutputs        *Tensor
	expertProbSums     []float64
	LoadBalancingLoss  float64
	RouterZLoss         float64
	gateLogits         *Tensor
	lastOutput         *Tensor
}

// MoELayer implements a Mixture of Experts layer.
type MoELayer struct {
	GatingNetwork *GatingNetwork
	Experts       []Expert
	K             int // Number of top experts to select
	InputDim      int
	OutputDim     int

	// Stored for backward pass
	inputTensor        *Tensor
	gateLogits         *Tensor // Raw logits for Z-loss gradient
	expertOutputs      []*Tensor
	expertTokenIndices [][]int // Indices of tokens assigned to each expert
	selectedExperts    [][]int // Indices of selected experts for each input in the batch
	gateOutputs        *Tensor // Output of the gating network (probabilities)
	LoadBalancingLoss  float64 // Load balancing loss
	Training           bool    // training mode
	GRPOEnabled        bool    // whether to use Training-Free GRPO (Group Relative Policy Optimization) for expert selection
	expertProbSums     []float64 // Sum of probabilities for each expert in the batch
	LoadBalancingWeight float64   // Weight for the load balancing loss
	CapacityFactor      float64   // Capacity factor to limit tokens per expert (e.g. 1.25)
	RouterTemperature   float64   // Temperature for router softmax (default 1.0)
	ExpertDropoutRate   float64   // Probability of dropping an expert during training (0.0 to 1.0)
	TopExpertIDs        []int     // The #1 expert chosen for each token in the last batch (diagnostic)
	RouterZLoss         float64   // Penalty for large router logits to keep them stable
	stateStack             []MoEState
	AccumulatedUtilization []int // Tracks token assignments across steps/batches
	ResidualScale          *Tensor   // Learned scale for the expert output in residual connection
	ExpertFrozen           []bool    // Toggles whether an expert's weights can be updated
	StagnationCounters     []int     // Counts consecutive steps/epochs with low utilization
	ExpertGradMultiplier   []float64 // Boosts gradients for experts recovering from freeze
}

// ResetUtilizationStats clears the accumulated utilization counters.
func (moe *MoELayer) ResetUtilizationStats() {
	if moe.AccumulatedUtilization == nil || len(moe.AccumulatedUtilization) != len(moe.Experts) {
		moe.AccumulatedUtilization = make([]int, len(moe.Experts))
	}
	for i := range moe.AccumulatedUtilization {
		moe.AccumulatedUtilization[i] = 0
	}
}

// safeAccumulate adds src to dst, ensuring we don't go out of bounds.
func safeAccumulate(dst, src []float64) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		dst[i] += src[i]
	}
}

// NewMoELayer creates a new MoELayer.
// inputDim is the dimension of the input to the MoE layer.
// numExperts is the total number of experts.
// k is the number of top experts to select for each input.
// expertBuilder is a function that constructs an expert given its index.
func NewMoELayer(inputDim, outputDim, numExperts, k int, expertBuilder func(int) (Expert, error)) (*MoELayer, error) {
	if k <= 0 || k > numExperts {
		return nil, fmt.Errorf("k (%d) must be between 1 and numExperts (%d)", k, numExperts)
	}

	gatingNetwork, err := NewGatingNetwork(inputDim, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create gating network: %w", err)
	}

	experts := make([]Expert, numExperts)
	for i := range numExperts {
		expert, err := expertBuilder(i)
		if err != nil {
			return nil, fmt.Errorf("failed to create expert %d: %w", i, err)
		}
		experts[i] = expert
	}

	layer := &MoELayer{
		GatingNetwork: gatingNetwork,
		Experts:       experts,
		K:             k,
		GRPOEnabled:   false, // Default to false. True requires implementing GRPO backward pass.
		InputDim:      inputDim,
		OutputDim:     outputDim,
		LoadBalancingWeight: 0.01, // Default weight
		CapacityFactor:      1.25, // Default capacity factor
		RouterTemperature:   0.8,  // Default temperature
		ExpertDropoutRate:   0.1,  // Default dropout
		ResidualScale:       NewTensor([]int{1}, []float64{1.0}, true), // Default to 1.0
		ExpertFrozen:        make([]bool, numExperts),
		StagnationCounters:  make([]int, numExperts),
		ExpertGradMultiplier: make([]float64, numExperts),
	}
	for i := range layer.ExpertGradMultiplier {
		layer.ExpertGradMultiplier[i] = 1.0
	}
	ActiveLayers = append(ActiveLayers, layer)
	return layer, nil
}

// Parameters returns all learnable parameters of the MoELayer.
func (moe *MoELayer) Parameters() []*Tensor {
	params := moe.GatingNetwork.Parameters()
	for _, expert := range moe.Experts {
		params = append(params, expert.Parameters()...)
	}
	if moe.ResidualScale != nil {
		params = append(params, moe.ResidualScale)
	}
	return params
}

// Forward performs the forward pass of the MoELayer.
// It takes an input tensor and returns the combined output of selected experts.
func (moe *MoELayer) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MoELayer.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	moe.inputTensor = input
	moe.TopExpertIDs = nil // Clear previous diagnostic state

	// 1. Gating Network (Router) forward pass to get logits
	gateLogits, err := moe.GatingNetwork.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("moe layer gating network forward failed: %w", err)
	}

	// 2. Calculate Router Z-Loss on RAW logits (Regularization)
	// This prevents logits from exploding and keeps the router stable.
	moe.RouterZLoss = CalculateRouterZLoss(gateLogits)
	moe.gateLogits = gateLogits // Store raw logits for BPTT

	// --- [Diagnostic] ---
	// Check input health to debug "Ghost Town" layers
	inputNorm := math.Sqrt(simdDotProductF64(input.Data, input.Data))
	if inputNorm < 1e-6 {
		fmt.Printf("⚠️ [MoELayer Diagnostic] Signal Collapse! Input L2 Norm: %.6f\n", inputNorm)
	}
	// --- [/Diagnostic] ---

	numExperts := len(moe.Experts)
	batchSize := input.Shape[0]
	seqLength := input.Shape[1]
	embeddingDim := input.Shape[2]

	// Add Router Jitter: Add noise to logits during training to encourage exploration.
	// Reduced noise to 0.01 as requested for "kickstarting" cold experts.
	if moe.Training {
		for i := range gateLogits.Data {
			noise := (rand.Float64() - 0.5) * 0.01 
			gateLogits.Data[i] += noise
		}
	}

	// Apply Temperature Scaling to logits
	// Default to 0.5 sharpening as requested ("divide by 0.5")
	routerScale := 1.0 / 0.5
	if moe.RouterTemperature > 0 {
		routerScale = 1.0 / moe.RouterTemperature
	}
	simdScaleF64(gateLogits.Data, routerScale)

	// --- [Penalty Mask for Over-Used Experts] ---
	// Aggressive Penalty for Dominant Experts (Router Level)
	// Subtract from the logit to make this expert less attractive if over-utilized
	if moe.Training {
		var totalTokensProcessed int
		for _, u := range moe.AccumulatedUtilization {
			totalTokensProcessed += u
		}
		if totalTokensProcessed > 1000 { // Only apply after some warmup
			avgTokens := float64(totalTokensProcessed) / float64(numExperts)
			for i := 0; i < batchSize*seqLength; i++ {
				for j := 0; j < numExperts; j++ {
					usage := float64(moe.AccumulatedUtilization[j])
					if usage > avgTokens {
						// log(usage/avg) is the penalty strength
						penalty := math.Log(usage/avgTokens) * 1.0 // 1.0 coefficient for strong nudge
						gateLogits.Data[i*numExperts+j] -= penalty
					}
				}
			}
		}
	}

	// Apply Expert Dropout during training
	if moe.Training && moe.ExpertDropoutRate > 0 {
		// Decide which experts to drop for this batch
		droppedMask := make([]bool, numExperts)
		activeCount := 0

		for i := 0; i < numExperts; i++ {
			if rand.Float64() < moe.ExpertDropoutRate {
				droppedMask[i] = true
			} else {
				activeCount++
			}
		}

		// Safety: Ensure at least K experts remain active
		for i := 0; i < numExperts && activeCount < moe.K; i++ {
			if droppedMask[i] {
				droppedMask[i] = false
				activeCount++
			}
		}

		// Apply mask to logits (set to -infinity)
		for i := 0; i < batchSize*seqLength; i++ {
			for j := 0; j < numExperts; j++ {
				if droppedMask[j] {
					gateLogits.Data[i*numExperts+j] = -1e9
				}
			}
		}
	}

	// Apply Training-Free GRPO if enabled
	var scoresTensor *Tensor
	if moe.GRPOEnabled {
		// Treat experts for each token as a group
		// Calculate mean and std of logits across experts for each token
		grpoLogits := make([]float64, len(gateLogits.Data))
		for i := 0; i < batchSize*seqLength; i++ {
			tokenLogits := gateLogits.Data[i*numExperts : (i+1)*numExperts]

			// Calculate Mean
			sum := 0.0
			for _, val := range tokenLogits {
				sum += val
			}
			mean := sum / float64(numExperts)

			// Calculate StdDev
			sqDiffSum := 0.0
			for _, val := range tokenLogits {
				diff := val - mean
				sqDiffSum += diff * diff
			}
			std := math.Sqrt(sqDiffSum / float64(numExperts))
			if std < 1e-8 {
				std = 1.0 // Prevent division by zero
			}

			// Advantage = (Logit - Mean) / Std
			for j := range numExperts {
				grpoLogits[i*numExperts+j] = (tokenLogits[j] - mean) / std
			}
		}
		scoresTensor = NewTensor(gateLogits.Shape, grpoLogits, gateLogits.RequiresGrad)
		if gateLogits.RequiresGrad {
			// We skip backprop complexity for GRPO normalization in "Training-Free" mode if it's meant for inference
			// But for completeness, we could link it. For now, we'll just use the data.
			scoresTensor.Creator = gateLogits.Creator
		}
	} else {
		scoresTensor = gateLogits
	}

	// Apply softmax to get probabilities
	gateOutputs, err := scoresTensor.Softmax(len(scoresTensor.Shape) - 1)
	if err != nil {
		return nil, fmt.Errorf("gating network softmax failed: %w", err)
	}
	moe.gateOutputs = gateOutputs

	// 4. Sum gating probabilities early (used for LoadBalancingLoss later)
	numTokens := batchSize * seqLength
	moe.expertProbSums = make([]float64, numExperts)
	if numTokens > 0 {
		for i := 0; i < numTokens; i++ {
			for j := 0; j < numExperts; j++ {
				moe.expertProbSums[j] += gateOutputs.Data[i*numExperts+j]
			}
		}
	}

	// 5. Hard Top-K selection and Gating Probability Zeroing

	// Calculate capacity limit per expert
	capacity := int(math.Ceil(moe.CapacityFactor * float64(batchSize*seqLength) / float64(numExperts)))
	if capacity < 1 {
		capacity = 1
	}

	moe.selectedExperts = make([][]int, batchSize*seqLength)
	moe.expertTokenIndices = make([][]int, numExperts)
	for i := range moe.expertTokenIndices {
		moe.expertTokenIndices[i] = make([]int, 0, capacity)
	}

	// Reshape input to 2D [batch*seq, dim] for gathering
	input2D, err := input.Reshape([]int{batchSize * seqLength, embeddingDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape input to 2D: %w", err)
	}

	// Store relative indices for scatter step
	tokenExpertRelativeIndices := make([][]int, batchSize*seqLength)

	// Pre-allocate indices buffers to reduce GC churn
	allTopKIndices := make([]int, batchSize*seqLength*numExperts)
	allRelativeIndices := make([]int, batchSize*seqLength*moe.K)

	moe.TopExpertIDs = make([]int, batchSize*seqLength)
	for i := 0; i < batchSize*seqLength; i++ {
		scores := scoresTensor.Data[i*numExperts : (i+1)*numExperts]
		topKIndices := allTopKIndices[i*numExperts : (i+1)*numExperts]
		for j := range topKIndices {
			topKIndices[j] = j
		}
		sort.SliceStable(topKIndices, func(a, b int) bool {
			return scores[topKIndices[a]] > scores[topKIndices[b]]
		})
		
		// Record the #1 expert for each token in the batch
		moe.TopExpertIDs[i] = topKIndices[0]

		// --- Hard top-K assignment with Overflow ---
		// We try to pick the best experts, but if they are full, we spill over to the next best
		// to ensure Expert Curiosity and stop the "Expert 3 Bottleneck".
		selected := make([]int, 0, moe.K)
		assignedCount := 0
		tokenRelativeIndices := allRelativeIndices[i*moe.K : (i+1)*moe.K]
		
		for _, expertIdx := range topKIndices {
			if assignedCount >= moe.K {
				break
			}

			stretchedCapacity := capacity
			// If router confidence is extremely high, allow some stretch
			if gateOutputs.Data[i*numExperts+expertIdx] > 0.8 {
				stretchedCapacity = int(float64(capacity) * 1.5)
			}

			if len(moe.expertTokenIndices[expertIdx]) < stretchedCapacity {
				tokenRelativeIndices[assignedCount] = len(moe.expertTokenIndices[expertIdx])
				moe.expertTokenIndices[expertIdx] = append(moe.expertTokenIndices[expertIdx], i)
				selected = append(selected, expertIdx)
				assignedCount++
				
				// Track stats
				if moe.AccumulatedUtilization == nil {
					moe.AccumulatedUtilization = make([]int, numExperts)
				}
				moe.AccumulatedUtilization[expertIdx]++
			}
		}
		
		// Pad with -1 if we couldn't find enough capacity (unlikely but safe)
		for j := assignedCount; j < moe.K; j++ {
			tokenRelativeIndices[j] = -1
		}

		moe.selectedExperts[i] = selected
		tokenExpertRelativeIndices[i] = tokenRelativeIndices

		// --- Hard Top-K: Zero out non-selected probabilities and re-normalize ---
		// This forces the model to ignore noise from "unskilled" experts.
		var rowSum float64
		expertMap := make(map[int]bool)
		for _, idx := range selected {
			expertMap[idx] = true
		}
		for j := 0; j < numExperts; j++ {
			idx := i*numExperts + j
			if !expertMap[j] {
				gateOutputs.Data[idx] = 0
			}
			rowSum += gateOutputs.Data[idx]
		}
		// Re-normalize remaining weights to sum to 1.0 for numerical stability
		if rowSum > 1e-12 {
			simdScaleF64(gateOutputs.Data[i*numExperts:(i+1)*numExperts], 1.0/rowSum)
		}
	}

	// 6. Finalize Load Balancing Loss (ST-MoE / Switch Transformer Style)
	if numTokens > 0 {
		moe.LoadBalancingLoss = 0
		for e := 0; e < numExperts; e++ {
			fraction := float64(len(moe.expertTokenIndices[e])) / float64(numTokens)
			meanProb := moe.expertProbSums[e] / float64(numTokens)
			moe.LoadBalancingLoss += fraction * meanProb
		}
		moe.LoadBalancingLoss *= float64(numExperts)
	} else {
		moe.LoadBalancingLoss = 0
	}

	moe.expertOutputs = make([]*Tensor, numExperts)
	var wg sync.WaitGroup
	var errMutex sync.Mutex
	var firstErr error

	// fmt.Println("Starting parallel expert execution (Forward)")
	// Run experts in parallel
	for i := range numExperts {
		indices := moe.expertTokenIndices[i]
		if len(indices) == 0 {
			continue
		}

		wg.Add(1)
		go func(expertIdx int, tokenIndices []int) {
			defer wg.Done()

			// Gather inputs for this expert
			batchedInput, err := input2D.Gather(tokenIndices)
			if err != nil {
				errMutex.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("failed to gather inputs for expert %d: %w", expertIdx, err)
				}
				errMutex.Unlock()
				return
			}

			// Forward pass
			output, err := moe.Experts[expertIdx].Forward(batchedInput)
			if err != nil {
				errMutex.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("expert %d forward failed: %w", expertIdx, err)
				}
				errMutex.Unlock()
				return
			}
			
			// --- Activation Clipping ---
			// Clamps every value between -15.0 and 15.0 to prevent activation explosion.
			output.Clip(-15.0, 15.0)
			
			moe.expertOutputs[expertIdx] = output
			// fmt.Printf("Expert %d finished forward\n", expertIdx)
		}(i, indices)
	}
	wg.Wait()
	// fmt.Println("Finished parallel expert execution (Forward)")

	if firstErr != nil {
		return nil, firstErr
	}

	// Scatter results back to final output
	finalOutput := NewTensor([]int{batchSize, seqLength, moe.OutputDim}, make([]float64, batchSize*seqLength*moe.OutputDim), true)

	// Stitch the graph: Register this layer as the creator of the output
	if moe.inputTensor.RequiresGrad {
		finalOutput.Creator = moe
	}

	// fmt.Println("Starting scattering")

	// Parallelize scattering by token
	var wgScatter sync.WaitGroup
	numWorkers := runtime.NumCPU()
	totalTokens := batchSize * seqLength
	tokensPerWorker := (totalTokens + numWorkers - 1) / numWorkers

	for w := range numWorkers {
		startToken := w * tokensPerWorker
		endToken := min(startToken+tokensPerWorker, totalTokens)
		if startToken >= endToken {
			break
		}

		wgScatter.Add(1)
		go func(start, end int) {
			defer wgScatter.Done()
			for i := start; i < end; i++ {
				selected := moe.selectedExperts[i]
				outStart := i * moe.OutputDim

				for j, expertIdx := range selected {
					output := moe.expertOutputs[expertIdx]
					if output == nil {
						continue
					}

					// Get weight
					weight := gateOutputs.Data[i*numExperts+expertIdx]

					// Get expert output row
					relativeRow := tokenExpertRelativeIndices[i][j]
					if relativeRow == -1 {
						continue // Token was dropped for this expert
					}
					expertRowStart := relativeRow * moe.OutputDim
					expertRow := output.Data[expertRowStart : expertRowStart+moe.OutputDim]
					outRow := finalOutput.Data[outStart : outStart+moe.OutputDim]

					simdAddScalarMulF64(outRow, expertRow, weight)
				}
			}
		}(startToken, endToken)
	}
	wgScatter.Wait()
	// fmt.Println("Finished scattering")

	// Push state to stack for BPTT
	state := MoEState{
		inputTensor:        moe.inputTensor,
		input2D:            input2D,
		expertOutputs:      moe.expertOutputs,
		expertTokenIndices: moe.expertTokenIndices,
		selectedExperts:    moe.selectedExperts,
		gateOutputs:        moe.gateOutputs,
		expertProbSums:     moe.expertProbSums,
		LoadBalancingLoss:  moe.LoadBalancingLoss,
		RouterZLoss:        moe.RouterZLoss,
		gateLogits:         moe.gateLogits,
		lastOutput:         finalOutput,
	}
	moe.stateStack = append(moe.stateStack, state)

	return finalOutput, nil
}

// Backward performs the backward pass for the MoELayer.
// Returns the gradient with respect to the input tensor.
func (moe *MoELayer) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// Pop state for BPTT
	var input2D *Tensor
	if len(moe.stateStack) > 0 {
		state := moe.stateStack[len(moe.stateStack)-1]
		moe.stateStack = moe.stateStack[:len(moe.stateStack)-1]

		moe.inputTensor = state.inputTensor
		input2D = state.input2D
		moe.expertOutputs = state.expertOutputs
		moe.expertTokenIndices = state.expertTokenIndices
		moe.selectedExperts = state.selectedExperts
		moe.gateOutputs = state.gateOutputs
		moe.expertProbSums = state.expertProbSums
		moe.LoadBalancingLoss = state.LoadBalancingLoss
		moe.RouterZLoss = state.RouterZLoss
		moe.gateLogits = state.gateLogits
	}

	// Remember if original grad was 2D (context vector) before reshaping/padding

	// Handle 2D gradient from decoder by processing only the last time step
	if len(grad.Shape) == 2 {
		// The grad is for the context vector, which corresponds to the last element of the sequence.
		// We will create a new grad tensor that has zeros everywhere except for the last time step.
		targetBatch := moe.inputTensor.Shape[0]
		embeddingDim := grad.Shape[1]
		seqLength := moe.inputTensor.Shape[1]

		// Create fullGrad with correct size and shape
		fullGradSize := targetBatch * seqLength * embeddingDim
		fullGradShape := []int{targetBatch, seqLength, embeddingDim}
		fullGrad := NewTensor(fullGradShape, make([]float64, fullGradSize), false)

		// Copy gradient to last time step for each batch
		minBatch := min(grad.Shape[0], targetBatch)
		for i := range minBatch {
			// Calculate the start index for the last time step of batch i
			lastTimeStepStart := (i*seqLength + (seqLength - 1)) * embeddingDim
			lastTimeStepEnd := lastTimeStepStart + embeddingDim

			// Source data from grad
			gradStart := i * embeddingDim
			gradEnd := gradStart + embeddingDim

			// Bounds check before copy
			if lastTimeStepEnd <= len(fullGrad.Data) && gradEnd <= len(grad.Data) {
				copy(fullGrad.Data[lastTimeStepStart:lastTimeStepEnd], grad.Data[gradStart:gradEnd])
			}
		}
		grad = fullGrad
	}

	// Ensure gradient shape matches input shape (specifically sequence length)
	if moe.inputTensor != nil && len(grad.Shape) == 3 && len(moe.inputTensor.Shape) == 3 {
		if grad.Shape[0] != moe.inputTensor.Shape[0] || grad.Shape[1] != moe.inputTensor.Shape[1] {
			targetBatch := moe.inputTensor.Shape[0]
			targetSeqLen := moe.inputTensor.Shape[1]
			embeddingDim := grad.Shape[2]

			newSize := targetBatch * targetSeqLen * embeddingDim
			newGrad := NewTensor(moe.inputTensor.Shape, make([]float64, newSize), false)

			minBatch := min(grad.Shape[0], targetBatch)
			minSeq := min(grad.Shape[1], targetSeqLen)

			for b := 0; b < minBatch; b++ {
				srcStart := b * grad.Shape[1] * embeddingDim
				dstStart := b * targetSeqLen * embeddingDim
				copySize := minSeq * embeddingDim
				copy(newGrad.Data[dstStart:dstStart+copySize], grad.Data[srcStart:srcStart+copySize])
			}
			grad = newGrad
		}
	}

	// Get dimensions from the gradient tensor (which may have been converted from 2D)
	batchSize := grad.Shape[0]
	seqLength := grad.Shape[1]
	embeddingDim := grad.Shape[2]
	numExperts := len(moe.Experts)

	// Initialize gradients for the MoE layer's input
	if moe.inputTensor.RequiresGrad {
		if moe.inputTensor.Grad == nil {
			moe.inputTensor.Grad = NewTensor(moe.inputTensor.Shape, make([]float64, len(moe.inputTensor.Data)), false)
		}
	}

	// Reshape grad to be [batchSize*seqLength, embeddingDim]
	gradReshaped, err := grad.Reshape([]int{batchSize * seqLength, embeddingDim})
	if err != nil {
		return fmt.Errorf("failed to reshape grad: %w", err)
	}

	// Initialize a tensor to accumulate gradients for the gating network
	gateGradReshaped := NewTensor([]int{batchSize * seqLength, numExperts}, make([]float64, batchSize*seqLength*numExperts), true)

	// fmt.Println("Starting parallel expert execution (Backward)")
	// Prepare gradients for each expert
	// We need to group gradients exactly as we grouped inputs in Forward
	// moe.expertTokenIndices has the mapping

	var wg sync.WaitGroup
	var errMutex sync.Mutex
	var inputGradMutex sync.Mutex
	var firstErr error

	// Run experts backward in parallel
	for i := range numExperts {
		indices := moe.expertTokenIndices[i]
		if len(indices) == 0 {
			continue
		}

		wg.Add(1)
		go func(expertIdx int, tokenIndices []int) {
			defer wg.Done()

			// Gather gradients for this expert
			// We use Gather on the reshaped grad tensor
			batchedGrad, err := gradReshaped.Gather(tokenIndices)
			if err != nil {
				errMutex.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("failed to gather grads for expert %d: %w", expertIdx, err)
				}
				errMutex.Unlock()
				return
			}

			// Get weight
			// dL/dExpertOutput = dL/dCombinedOutput * weight
			// We need to multiply batchedGrad by the corresponding gate output weights.
			// The batchedGrad is [numTokensForExpert, embeddingDim]
			// The weights are moe.gateOutputs.Data[tokenIdx*numExperts+expertIdx]
			// We need to create a weightedBatchedGrad
			weightedBatchedGradData := make([]float64, len(batchedGrad.Data))
			for k, tokenIdx := range tokenIndices {
				gateIdx := tokenIdx*numExperts + expertIdx
				if gateIdx < 0 || gateIdx >= len(moe.gateOutputs.Data) {
					continue
				}
				weight := moe.gateOutputs.Data[gateIdx]
				simdMulScalarF64(weightedBatchedGradData[k*embeddingDim:(k+1)*embeddingDim], batchedGrad.Data[k*embeddingDim:(k+1)*embeddingDim], weight)
			}
			weightedBatchedGrad := NewTensor(batchedGrad.Shape, weightedBatchedGradData, false)

			// Backward pass
			err = moe.Experts[expertIdx].Backward(weightedBatchedGrad)
			if err != nil {
				errMutex.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("expert %d backward failed: %w", expertIdx, err)
				}
				errMutex.Unlock()
				return
			}

			// --- [Expert Grad Multiplier (Jump Start)] ---
			// Apply a multiplier to experts recovering from freeze or stagnation
			if moe.ExpertGradMultiplier != nil && moe.ExpertGradMultiplier[expertIdx] > 1.0 {
				params := moe.Experts[expertIdx].Parameters()
				for _, p := range params {
					if p.Grad != nil {
						simdScaleF64(p.Grad.Data, moe.ExpertGradMultiplier[expertIdx])
					}
				}
				// Decay multiplier toward 1.0 (50 steps to normalization @ 0.98 decay)
				moe.ExpertGradMultiplier[expertIdx] *= 0.98
				if moe.ExpertGradMultiplier[expertIdx] < 1.02 {
					moe.ExpertGradMultiplier[expertIdx] = 1.0
				}
			}

			// Accumulate input gradients
			// The expert's input was created via Gather.
			// So expert.Inputs()[0].Grad contains the gradients w.r.t the gathered input.
			// We need to scatter these back to the original input.
			// Fortunately, GatherOperation.Backward does exactly this!
			// We just need to trigger backward on the gathered input.

			if moe.inputTensor.RequiresGrad {
				expertInputs := moe.Experts[expertIdx].Inputs()
				if len(expertInputs) == 0 {
					return
				}
				if len(expertInputs) > 0 {
					gatheredInput := expertInputs[0]
					// Trigger backward on the gathered input to scatter gradients to input2D (and then to inputTensor)
					// Note: gatheredInput.Creator is the GatherOperation.
					if gatheredInput.Creator != nil && gatheredInput.Grad != nil {
						inputGradMutex.Lock()
						err := gatheredInput.Creator.Backward(gatheredInput.Grad)
						inputGradMutex.Unlock()
						if err != nil {
							errMutex.Lock()
							if firstErr == nil {
								firstErr = fmt.Errorf("failed to scatter grads for expert %d: %w", expertIdx, err)
							}
							errMutex.Unlock()
							return
						}
					}
				}
			}

			// Accumulate gating gradients
			// dL/dGate = dot(grad_token, expert_output)
			// expertOutput is stored in moe.expertOutputs[expertIdx]
			expertOutput := moe.expertOutputs[expertIdx]

			for k, tokenIdx := range tokenIndices {
				// Re-fetch grad for token
				startIdx := tokenIdx * embeddingDim
				endIdx := (tokenIdx + 1) * embeddingDim
				if startIdx >= len(gradReshaped.Data) || endIdx > len(gradReshaped.Data) {
					continue
				}
				gradForTokenData := gradReshaped.Data[startIdx:endIdx]

				expertRowStart := k * embeddingDim
				expertRowEnd := (k + 1) * embeddingDim
				if expertRowStart >= len(expertOutput.Data) || expertRowEnd > len(expertOutput.Data) {
					continue
				}
				expertOutRow := expertOutput.Data[expertRowStart:expertRowEnd]

				gradForGateProb := simdDotProductF64(gradForTokenData, expertOutRow)

				// This write is safe?
				// gateGradReshaped is [batch*seq, numExperts].
				// Each expert writes to a DIFFERENT column (expertIdx).
				// So this IS thread-safe.
				gateGradIdx := tokenIdx*numExperts + expertIdx
				if gateGradIdx >= 0 && gateGradIdx < len(gateGradReshaped.Data) {
					gateGradReshaped.Data[gateGradIdx] += gradForGateProb
				}
			}

		}(i, indices)
	}
	wg.Wait()
	// fmt.Println("Finished parallel expert execution (Backward)")

	if firstErr != nil {
		return firstErr
	}

	// Add load balancing (Square of Sums) loss gradient
	if moe.LoadBalancingWeight > 0 && len(moe.expertProbSums) == numExperts && batchSize*seqLength > 0 {
		// Differentiable penalty: d(2.0 * sum(mean_prob_j^2)) / d(p_ij)
		// which is calculated using our SIMD helper.
		simdSquareOfSumsGradF64(gateGradReshaped.Data, moe.expertProbSums, batchSize*seqLength, numExperts, moe.LoadBalancingWeight*2.0)
	}

	// The input gradients have been accumulated into input2D.Grad by the GatherOperation.Backward calls.
	// Now we need to propagate them from input2D to inputTensor.
	// input2D was created by Reshape. Reshape's backward pass handles this.
	// But wait, we didn't call input2D.Backward(). We manually called GatherOperation.Backward.
	// So input2D.Grad is populated.
	// We need to manually propagate from input2D to inputTensor if we don't use the full autograd graph.
	// Since input2D shares data with inputTensor (in Reshape implementation), does it share Grad?
	// Let's check Reshape implementation.
	// Reshape: resultTensor.Grad is new. It does NOT share Grad with input.
	// So we need to call input2D.Creator.Backward(input2D.Grad) if it exists, or just manually map it.
	// input2D.Creator is ReshapeOperation.

	// Actually, simpler: input2D is just a reshaped view.
	// If we just call input2D.Backward(input2D.Grad), it should propagate to inputTensor.
	// But we need to be careful not to double count or mess up if we are doing partial backward.

	// Let's look at how we set up input2D:
	// input2D, err := input.Reshape(...)
	// So input2D.Creator is ReshapeOperation{Input: input}.
	// If we call input2D.Creator.Backward(input2D.Grad), it will add to input.Grad.

	// However, we need to access input2D here. It was created in Forward.
	// We didn't store input2D in the struct.
	// We can recreate it (it's cheap) or just know that input2D.Grad has the same shape as inputTensor.Grad (flattened).
	// Actually, input2D.Grad data is what we want to add to inputTensor.Grad.

	// Wait, the GatherOperation.Backward updated input2D.Grad.
	// But where is input2D? It's lost after Forward.
	// Ah, we need to store input2D or recreate it to get access to its Grad.
	// OR, we can pass the inputTensor to Gather if we flatten it first?
	// No, Gather expects 2D.

	// PROBLEM: GatherOperation stores a reference to its Input.
	// In Forward: input2D.Gather(...) -> GatherOperation{Input: input2D}.
	// So the GatherOperation holds input2D.
	// When we call gatheredInput.Creator.Backward(), it updates input2D.Grad.
	// So input2D is kept alive by the graph.
	// But we don't have a direct reference to input2D here in Backward to call its backward.

	// BUT, we can get input2D from the expert inputs!
	// expertInputs[0] is the gathered tensor.
	// expertInputs[0].Creator is the GatherOperation.
	// expertInputs[0].Creator.Input is input2D!
	// So we can find input2D from there.

	if moe.inputTensor.RequiresGrad {
		if input2D != nil && input2D.Grad != nil {
			// Propagate from input2D to inputTensor
			// input2D was created from inputTensor via Reshape.
			// We can manually call Reshape's backward logic or just copy/add data.
			// Reshape backward just adds gradients.
			if moe.inputTensor.Grad == nil {
				moe.inputTensor.Grad = NewTensor(moe.inputTensor.Shape, make([]float64, len(moe.inputTensor.Data)), false)
			}
			safeAccumulate(moe.inputTensor.Grad.Data, input2D.Grad.Data)
		}
	}

	// Finally, backpropagate through the gating network with the accumulated gateGrad.
	// Workaround: GatingNetwork.Backward (Linear.Backward) seems to cause moe.inputTensor.Grad to become nil
	// or overwritten in some cases. We explicitly copy the expert gradients to preserve them.
	var expertGrads []float64
	if moe.inputTensor.Grad != nil {
		expertGrads = make([]float64, len(moe.inputTensor.Grad.Data))
		copy(expertGrads, moe.inputTensor.Grad.Data)
	}

	// Convert probability gradients to logit gradients (Softmax Backward)
	logitsGrad := NewTensor(gateGradReshaped.Shape, make([]float64, len(gateGradReshaped.Data)), false)

	// Parallelize softmax backward
	var wgSoftmax sync.WaitGroup
	numWorkers := runtime.NumCPU()
	rowsPerWorker := (batchSize * seqLength + numWorkers - 1) / numWorkers
	for w := 0; w < numWorkers; w++ {
		start := w * rowsPerWorker
		end := min(start+rowsPerWorker, batchSize*seqLength)
		if start >= end {
			break
		}
		wgSoftmax.Add(1)
		go func(s, e int) {
			defer wgSoftmax.Done()
			for i := s; i < e; i++ {
				offset := i * numExperts
				p := moe.gateOutputs.Data[offset : offset+numExperts]
				dp := gateGradReshaped.Data[offset : offset+numExperts]
				out := logitsGrad.Data[offset : offset+numExperts]

				sumDP := simdDotProductF64(dp, p)
				simdSoftmaxBackwardRowF64(out, p, dp, sumDP)
			}
		}(start, end)
	}
	wgSoftmax.Wait()

	// --- [Auxiliary Loss Gradient (Manual Inject)] ---
	// dL/dP_ie = lambda * (N/T) * f_e
	// We add this to gateGradReshaped before the softmax backward if it's w.r.t probabilities
	// OR we add it to logitsGrad. Usually it's w.r.t. probabilities.
	// But the user's SIMD pseudocode adds it to gateGrads (which I named gateGradReshaped).
	// 6. Finalize Load Balancing Loss Gradient
	numTokens := batchSize * seqLength
	if moe.LoadBalancingWeight > 0 && numTokens > 0 {
		numExperts := len(moe.Experts)
		T := float64(numTokens)
		N := float64(numExperts)
		scalar := moe.LoadBalancingWeight * (N / (T * T))

		scaledFractions := make([]float64, numExperts)
		for e := 0; e < numExperts; e++ {
			scaledFractions[e] = float64(len(moe.expertTokenIndices[e])) * scalar
		}

		// Prepare 2D view for UpdateRouterGrads
		gateGrads2D := make([][]float64, numTokens)
		for i := 0; i < numTokens; i++ {
			gateGrads2D[i] = gateGradReshaped.Data[i*numExperts : (i+1)*numExperts]
		}

		moe.UpdateRouterGrads(gateGrads2D, scaledFractions)
	}

	// Apply Temperature Scaling to gradients (dL/dx = dL/dy * 1/T)
	if moe.RouterTemperature != 1.0 && moe.RouterTemperature > 0 {
		simdScaleF64(logitsGrad.Data, 1.0/moe.RouterTemperature)
	}

	// 4. Add Router Z-Loss Gradient (dL_z / d_logits)
	// L_z = (sum(logits^2) / n) * 0.0001
	// dL_z / d_logit_i = (2 * logit_i / n) * 0.0001
	if moe.gateLogits != nil && len(moe.gateLogits.Data) == len(logitsGrad.Data) {
		n := float64(len(moe.gateLogits.Data))
		const c = 0.0001
		for i := range logitsGrad.Data {
			logitsGrad.Data[i] += (2.0 * moe.gateLogits.Data[i] / n) * c
		}
	}

	// Router-Fast approach: Scale gradients for router to make it learn faster
	routerGradientScale := 5.0
	for i := range logitsGrad.Data {
		logitsGrad.Data[i] *= routerGradientScale
	}

	err = moe.GatingNetwork.Backward(logitsGrad)
	if err != nil {
		return err
	}

	// Restore/Accumulate expert gradients
	if expertGrads != nil {
		if moe.inputTensor.Grad == nil {
			moe.inputTensor.Grad = NewTensor(moe.inputTensor.Shape, expertGrads, false)
		} else {
			safeAccumulate(moe.inputTensor.Grad.Data, expertGrads)
		}
	}

	// Gradient is stored in moe.inputTensor.Grad
	return nil
}

// Inputs returns the input tensors of the MoELayer's last forward operation.
func (moe *MoELayer) Inputs() []*Tensor {
	if moe.inputTensor != nil {
		return []*Tensor{moe.inputTensor}
	}
	return []*Tensor{}
}

// SetMode sets the mode for the MoELayer and all its experts.
func (moe *MoELayer) SetMode(training bool) {
	moe.Training = training
	for _, expert := range moe.Experts {
		expert.SetMode(training)
	}
}

func (moe *MoELayer) GetOutputShape() []int {
	return moe.inputTensor.Shape
}

// GetSelectedExperts returns the indices of experts selected for each token in the last forward pass.
// The returned slice has length batchSize * seqLength.
func (moe *MoELayer) GetSelectedExperts() [][]int {
	return moe.selectedExperts
}

// ClearState clears the internal state of the layer to free memory.
func (moe *MoELayer) ClearState() {
	moe.inputTensor = nil
	moe.expertOutputs = nil
	moe.expertTokenIndices = nil
	moe.selectedExperts = nil
	moe.gateOutputs = nil
	moe.expertProbSums = nil
	moe.stateStack = nil
	moe.gateLogits = nil

	// Clear state for all experts
	for _, expert := range moe.Experts {
		if expert != nil {
			expert.ClearState()
		}
	}
}

// UtilizationStats returns a map of expert index to the number of tokens it processed in the last forward pass.
// This is useful for checking if all experts are active.
func (moe *MoELayer) UtilizationStats() map[int]int {
	stats := make(map[int]int)
	if moe.AccumulatedUtilization == nil {
		for i := range moe.Experts {
			stats[i] = 0
		}
		return stats
	}
	for i, count := range moe.AccumulatedUtilization {
		stats[i] = count
	}
	return stats
}

// VisualizeUtilization prints a text-based bar chart of expert utilization to stdout.
// This helps visualize load balancing during training.
func (moe *MoELayer) VisualizeUtilization() {
	stats := moe.UtilizationStats()
	total := 0
	for _, count := range stats {
		total += count
	}

	fmt.Printf("Expert Utilization (Capacity Factor: %.2f):\n", moe.CapacityFactor)
	var keys []int
	for k := range stats {
		keys = append(keys, k)
	}
	sort.Ints(keys)

	for _, i := range keys {
		count := stats[i]
		percent := 0.0
		if total > 0 {
			percent = float64(count) / float64(total) * 100
		}
		barLen := int(percent / 2) // 50 chars = 100%
		bar := strings.Repeat("#", barLen)
		fmt.Printf("  Expert %d: %4d (%5.1f%%) %s\n", i, count, percent, bar)
	}
}

// CalculateRouterZLoss penalizes large logit values to keep the router stable.
func CalculateRouterZLoss(routerLogits *Tensor) float64 {
	if routerLogits == nil || len(routerLogits.Data) == 0 {
		return 0
	}
	sumSq := simdDotProductF64(routerLogits.Data, routerLogits.Data)
	// Multiply by a small coefficient (e.g., 1e-4) as suggested by PaLM/ST-MoE
	return (sumSq / float64(len(routerLogits.Data))) * 0.0001
}

// CalculateSquareOfSumsLoss implements an aggressive diversity penalty to break monopolies.
// This calculates sum((count_i/total)^2) which rewards equal distribution.
func CalculateSquareOfSumsLoss(usageCounts []int, totalTokens int) float64 {
	// Weight heavily (e.g. 2.0) as requested to break stalemates
	return simdSquareOfSumsLossF64(usageCounts, totalTokens, 2.0)
}
// RebalanceExperts ensures all experts have a similar weight magnitude (L2 Norm).
// This prevents one expert from becoming a "gravity well" for the router.
func (moe *MoELayer) RebalanceExperts() {
	numExperts := len(moe.Experts)
	if numExperts == 0 {
		return
	}

	expertNorms := make([]float64, numExperts)
	var totalNorm float64

	// 1. Calculate individual norms and the average
	for i := 0; i < numExperts; i++ {
		params := moe.Experts[i].Parameters()
		var expertSumSq float64
		var count int
		for _, p := range params {
			for _, v := range p.Data {
				expertSumSq += v * v
				count++
			}
		}
		if count > 0 {
			expertNorms[i] = math.Sqrt(expertSumSq)
		}
		totalNorm += expertNorms[i]
	}
	avgNorm := totalNorm / float64(numExperts)

	fmt.Printf("⚖️ Rebalancing Experts to target average L2 Norm: %.4f\n", avgNorm)

	// 2. Scale each expert's weights to match the average
	for i := 0; i < numExperts; i++ {
		if expertNorms[i] == 0 {
			continue
		}

		scalingFactor := avgNorm / expertNorms[i]
		params := moe.Experts[i].Parameters()
		for _, p := range params {
			for j := range p.Data {
				p.Data[j] *= scalingFactor
			}
		}

		fmt.Printf("   Expert %d: Norm %.4f -> %.4f (Scaled by %.4f)\n", i, expertNorms[i], avgNorm, scalingFactor)
	}

	// 3. Optional: Reset Router weights to small values to force re-learning
	for i := range moe.GatingNetwork.Linear.Weights.Data {
		moe.GatingNetwork.Linear.Weights.Data[i] = (rand.Float64()*2 - 1) * 0.01
	}
	if moe.GatingNetwork.Linear.Biases != nil {
		for i := range moe.GatingNetwork.Linear.Biases.Data {
			moe.GatingNetwork.Linear.Biases.Data[i] = 0
		}
	}
	fmt.Println("👉 Router weights reset to small random values.")
}

// EvolutionaryReset identifies stagnant experts and replaces them with a slightly
// mutated clone of the most successful active expert.
func (moe *MoELayer) EvolutionaryReset(stagnationThreshold int) {
	numExperts := len(moe.Experts)
	if numExperts < 2 {
		return
	}

	// 1. Identify the "Best" Expert (highest utilization in this epoch/window)
	bestExpertIdx := -1
	maxUsage := -1
	for i, usage := range moe.AccumulatedUtilization {
		if usage > maxUsage {
			maxUsage = usage
			bestExpertIdx = i
		}
	}

	if bestExpertIdx == -1 || maxUsage == 0 {
		return // No data to reset from
	}

	// 2. Identify and Reset Stagnant Experts
	for i := 0; i < numExperts; i++ {
		if i == bestExpertIdx {
			moe.StagnationCounters[i] = 0
			continue
		}

		// If usage is extremely low (e.g., < 1% of total tokens), increment counter
		totalTokens := 0
		for _, u := range moe.AccumulatedUtilization {
			totalTokens += u
		}
		
		isStagnant := moe.AccumulatedUtilization[i] < (totalTokens / (numExperts * 10))
		if isStagnant {
			moe.StagnationCounters[i]++
		} else {
			moe.StagnationCounters[i] = 0
		}

		if moe.StagnationCounters[i] >= stagnationThreshold {
			fmt.Printf("🧬 [Evolutionary Reset] Expert %d stagnant for %d cycles. Cloning Expert %d...\n", i, moe.StagnationCounters[i], bestExpertIdx)
			
			// CLONE: Copy weights from bestExpert to stagnant expert with Jitter
			sourceParams := moe.Experts[bestExpertIdx].Parameters()
			targetParams := moe.Experts[i].Parameters()
			
			for pIdx := range sourceParams {
				if pIdx >= len(targetParams) {
					continue
				}
				// Use 15% Gaussian jitter as requested to encourage specialization
				simdAddJitterF64(targetParams[pIdx].Data, sourceParams[pIdx].Data, 0.15)
			}

			// RESET ROUTER: Clear the router's bias against this expert and set small random weights
			// Router weights are [inputDim, numExperts]. Expert i is the i-th column.
			inputDim := moe.GatingNetwork.Linear.Weights.Shape[0]
			for k := 0; k < inputDim; k++ {
				// Reset weight from all input dimensions to this expert
				moe.GatingNetwork.Linear.Weights.Data[k*numExperts+i] = (rand.Float64()*2 - 1) * 0.05
			}
			
			if moe.GatingNetwork.Linear.Biases != nil {
				moe.GatingNetwork.Linear.Biases.Data[i] = 0
			}

			moe.StagnationCounters[i] = 0
		}
	}
}

// ResizeExperts resizes the output dimension of all experts.
func (moe *MoELayer) ResizeExperts(newOutputDim int) {
	fmt.Printf("🔧 Resizing %d MoE Experts to new OutputDim: %d\n", len(moe.Experts), newOutputDim)
	
	for i, exp := range moe.Experts {
		// Type assert to FeedForwardExpert
		ff, ok := exp.(*FeedForwardExpert)
		if !ok {
			// Try to handle other expert types if added later
			fmt.Printf("⚠️ Skip resizing expert %d: unexpected type\n", i)
			continue
		}

		oldWeights := ff.Layer2.Weights
		oldBiases := ff.Layer2.Biases
		oldVocabSize := ff.Layer2.Weights.Shape[1]
		inputDim := ff.Layer2.Weights.Shape[0]

		// Create new Linear layer
		newLinear, _ := nn.NewLinear(inputDim, newOutputDim)

		// Copy weights: [inputDim * vocabSize]
		copyLimit := oldVocabSize
		if newOutputDim < copyLimit {
			copyLimit = newOutputDim
		}

		// Use a safe copy for weights
		for row := 0; row < inputDim; row++ {
			for col := 0; col < copyLimit; col++ {
				oldIdx := row*oldVocabSize + col
				newIdx := row*newOutputDim + col
				if oldIdx < len(oldWeights.Data) && newIdx < len(newLinear.Weights.Data) {
					newLinear.Weights.Data[newIdx] = oldWeights.Data[oldIdx]
				}
			}
		}

		// Copy biases
		if oldBiases != nil && newLinear.Biases != nil {
			for col := 0; col < copyLimit; col++ {
				if col < len(oldBiases.Data) && col < len(newLinear.Biases.Data) {
					newLinear.Biases.Data[col] = oldBiases.Data[col]
				}
			}
		}

		// Swap the old layer for the new resized one
		ff.Layer2 = newLinear
	}
	moe.OutputDim = newOutputDim
}

// SetExpertFreeze toggles the learnability of a specific expert.
func (moe *MoELayer) SetExpertFreeze(expertID int, freeze bool) {
	if expertID < 0 || expertID >= len(moe.Experts) {
		return
	}
	// If unfreezing, set a jump-start multiplier
	if !freeze && moe.ExpertFrozen[expertID] {
		moe.ExpertGradMultiplier[expertID] = 2.5 // Initial boost
		fmt.Printf("🚀 Expert %d UNROZEN: Applying 2.5x Gradient Jump-Start\n", expertID)
	}

	moe.ExpertFrozen[expertID] = freeze
	
	// Set RequiresGrad for all parameters of the expert
	params := moe.Experts[expertID].Parameters()
	for _, p := range params {
		p.RequiresGrad = !freeze
	}
	fmt.Printf("❄️ Expert %d Freeze State: %v\n", expertID, freeze)
}


// GetStateStack returns the internal state stack for BPTT.
func (moe *MoELayer) GetStateStack() []MoEState {
	return moe.stateStack
}

// GetMoELayers returns the layer itself.
func (moe *MoELayer) GetMoELayers() []*MoELayer {
	return []*MoELayer{moe}
}

// UpdateRouterGrads handles the load balancing penalty in pure Go.
func (moe *MoELayer) UpdateRouterGrads(gateGrads [][]float64, scaledFractions []float64) {
	numExperts := len(scaledFractions)

	for i := range gateGrads {
		// This slice capture helps the compiler eliminate bounds checks in the inner loop
		grads := gateGrads[i]
		if len(grads) < numExperts {
			continue
		}

		// Loop unrolling: process 4 experts at a time
		// This encourages the Go compiler to use AVX registers automatically
		e := 0
		for ; e <= numExperts-4; e += 4 {
			grads[e]   += scaledFractions[e]
			grads[e+1] += scaledFractions[e+1]
			grads[e+2] += scaledFractions[e+2]
			grads[e+3] += scaledFractions[e+3]
		}

		// Handle the tail
		for ; e < numExperts; e++ {
			grads[e] += scaledFractions[e]
		}
	}
}
