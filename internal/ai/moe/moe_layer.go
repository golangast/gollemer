package moe

import (
	"fmt"
	"iter"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// ActiveLayers tracks all MoE layers created, useful for monitoring utilization.
var ActiveLayers []*MoELayer

// MoEState holds the state of a single forward pass for BPTT.
type MoEState struct {
	inputTensor        *Tensor
	input2D            *Tensor
	expertOutputs      []*Tensor
	ExpertTokenIndices [][]int
	SelectedExperts    [][]int
	GateOutputs        *Tensor
	ExpertProbSums     []float32
	LoadBalancingLoss  float32
	RouterZLoss        float32
	gateLogits         *Tensor
	lastOutput         *Tensor
	TargetRouting      []int // Ground truth expert IDs (0-7) for tokens in this pass
}

// MoELayer implements a Mixture of Experts layer.
type MoELayer struct {
	GatingNetwork *GatingNetwork
	Experts       []Expert
	K             int // Number of top experts to select
	InputDim      int
	OutputDim     int
	NumExperts    int

	// Stored for backward pass
	inputTensor                *Tensor
	gateLogits                 *Tensor // Raw logits for Z-loss gradient
	expertOutputs              []*Tensor
	ExpertTokenIndices         [][]int // Indices of tokens assigned to each expert
	SelectedExperts            [][]int // Indices of selected experts for each input in the batch
	GateOutputs                *Tensor // Output of the gating network (probabilities)
	LoadBalancingLoss          float32 // Load balancing loss
	GatingEntropyLoss          float32
	Training                   bool      // training mode
	OverfitMode                bool      // If true, disable autonomous resets and exploration noise
	GRPOEnabled                bool      // whether to use Training-Free GRPO (Group Relative Policy Optimization) for expert selection
	ExpertProbSums             []float32 // Sum of probabilities for each expert in the batch
	LoadBalancingWeight        float32   // Weight for the load balancing loss
	CapacityFactor             float32   // Capacity factor to limit tokens per expert (e.g. 1.25)
	RouterTemperature          float32   // Temperature for router softmax (default 1.0)
	ExpertDropoutRate          float32   // Probability of dropping an expert during training (0.0 to 1.0)
	TopExpertIDs               []int     // The #1 expert chosen for each token in the last batch (diagnostic)
	RouterZLoss                float32   // Penalty for large router logits to keep them stable
	stateStack                 []MoEState
	AccumulatedUtilization     []int     // Tracks token assignments across steps/batches
	ResidualScale              *Tensor   // Learned scale for the expert output in residual connection
	ExpertFrozen               []bool    // Toggles whether an expert's weights can be updated
	StagnationCounters         []int     // Counts consecutive steps/epochs with low utilization
	ExpertGradMultiplier       []float32 // Boosts gradients for experts recovering from freeze
	DiversityLoss              float32   // Penalty for experts being too similar
	TargetRouting              []int     // Ground truth expert IDs for each token (set before Forward)
	PersistenceBias            float32   // Bias for experts selected in the previous token step
	LastSelectedExperts        [][]int   // Stores experts chosen for each batch item in the previous step
	MutedTokenID               int       // Token ID to mute obsession (e.g. "i")
	MutedTokenScale            float32   // Gradient multiplier for the muted token
	ResetCount                 int32     // Atomic counter for experts reset in this epoch
	expertResets               map[int]int
	resetsMu                   sync.RWMutex
	CurrentStepIndex           int
	StepRoutingBias            map[int][]float32 // [step]bias_per_expert
	SoftRouting                bool              // If true, use weighted average of all experts
	StructuralRoutingWeight    float32           // Penalty strength for deviating from TargetRouting
	StructuralBiasIntensity    float32           // Positive boost for TargetRouting during Forward
	ExpertOutputScale          []float32         // Per-expert output multiplier
	ExpertRegularizationWeight float32           // Penalty for experts deviating from healthy mean
	ExpertSparsityWeight       float32           // Penalty for non-sparse or unbalanced expert selection
	HealthyExpertIDs           []int             // List of IDs for experts considered "healthy" (e.g. 14, 9)
}

// ExpertTask represents work to be done by one expert on a subset of tokens.
type ExpertTask struct {
	ExpertIdx    int
	TokenIndices []int
}

// ExpertTasks returns an iterator over experts that have tokens assigned to them.
func (moe *MoELayer) ExpertTasks() iter.Seq[ExpertTask] {
	return func(yield func(ExpertTask) bool) {
		for i := 0; i < moe.NumExperts; i++ {
			indices := moe.ExpertTokenIndices[i]
			if len(indices) > 0 {
				if !yield(ExpertTask{ExpertIdx: i, TokenIndices: indices}) {
					return
				}
			}
		}
	}
}

// ResetUtilizationStats clears the accumulated utilization counters and expert resets.
func (moe *MoELayer) ResetUtilizationStats() {
	if moe.AccumulatedUtilization == nil || len(moe.AccumulatedUtilization) != len(moe.Experts) {
		moe.AccumulatedUtilization = make([]int, len(moe.Experts))
	}
	for i := range moe.AccumulatedUtilization {
		moe.AccumulatedUtilization[i] = 0
	}

	moe.resetsMu.Lock()
	moe.expertResets = make(map[int]int)
	moe.resetsMu.Unlock()
}

// safeAccumulate adds src to dst, ensuring we don't go out of bounds.
func safeAccumulate(dst, src []float32) {
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
		if expertBuilder != nil {
			expert, err := expertBuilder(i)
			if err != nil {
				return nil, fmt.Errorf("failed to create expert %d: %w", i, err)
			}
			experts[i] = expert
		} else {
			// Default to InternalExpert for unified native performance
			hiddenDim := inputDim * 4 // Common multiplier
			expert, err := NewInternalExpert(i, inputDim, hiddenDim, outputDim)
			if err != nil {
				return nil, fmt.Errorf("failed to create internal expert %d: %w", i, err)
			}
			experts[i] = expert
		}
	}

	layer := &MoELayer{
		GatingNetwork:              gatingNetwork,
		Experts:                    experts,
		NumExperts:                 numExperts,
		K:                          k,
		GRPOEnabled:                false, // Default to false. True requires implementing GRPO backward pass.
		InputDim:                   inputDim,
		OutputDim:                  outputDim,
		LoadBalancingWeight:        0.1,                                       // Increased from 0.01 for tiny datasets
		CapacityFactor:             1.25,                                      // Default capacity factor
		RouterTemperature:          0.8,                                       // Default temperature
		ExpertDropoutRate:          0.3,                                       // Increased from 0.1 to force alternative expert learning
		ResidualScale:              NewTensor([]int{1}, []float32{1.0}, true), // Default to 1.0
		ExpertFrozen:               make([]bool, numExperts),
		StagnationCounters:         make([]int, numExperts),
		ExpertGradMultiplier:       make([]float32, numExperts),
		MutedTokenID:               -1,
		MutedTokenScale:            1.0,
		StepRoutingBias:            make(map[int][]float32),
		StructuralRoutingWeight:    5.0, // Default strength
		StructuralBiasIntensity:    8.0, // Default boost
		ExpertOutputScale:          make([]float32, numExperts),
		ExpertRegularizationWeight: 0.0, // Disabled by default
		ExpertSparsityWeight:       0.0, // Disabled by default
	}
	for i := range layer.ExpertOutputScale {
		layer.ExpertOutputScale[i] = 1.0
	}
	for i := range layer.ExpertGradMultiplier {
		layer.ExpertGradMultiplier[i] = 1.0
	}
	layer.PersistenceBias = 1.5 // Default persistence strength
	ActiveLayers = append(ActiveLayers, layer)
	return layer, nil
}

// ResetRouterWeights targets the gating mechanism to break "Expert Dominance."
func (moe *MoELayer) ResetRouterWeights() {
	if moe.GatingNetwork == nil {
		return
	}

	weights := moe.GatingNetwork.Linear.Weights
	inputDim := weights.Shape[0]
	numExperts := weights.Shape[1]

	// Use a very small Xavier initialization to start with high uncertainty (Max Entropy)
	stdDev := float32(math.Sqrt(2.0/float64(inputDim+numExperts))) * 0.1

	for i := range weights.Data {
		weights.Data[i] = float32(rand.NormFloat64()) * stdDev
	}

	if moe.GatingNetwork.Linear.Biases != nil {
		for i := range moe.GatingNetwork.Linear.Biases.Data {
			moe.GatingNetwork.Linear.Biases.Data[i] = 0
		}
	}

	// 🛡️ Weight Clamping: Routers should never be "too certain"
	moe.GatingNetwork.Linear.Weights.Clip(-2.5, 2.5)

	fmt.Printf("🚀 Router Gating Weights Reset & Clamped for %T: Forcing Exploration.\n", moe)
}

// ResetExpertWeights re-initializes a specific expert's weights to break "Semantic Sink" behavior.
func (moe *MoELayer) ResetExpertWeights(expertIdx int) {
	if expertIdx < 0 || expertIdx >= len(moe.Experts) {
		return
	}

	moe.resetsMu.Lock()
	if moe.expertResets == nil {
		moe.expertResets = make(map[int]int)
	}
	moe.expertResets[expertIdx]++
	moe.resetsMu.Unlock()

	atomic.AddInt32(&moe.ResetCount, 1)
	fmt.Printf("🔥 Expert E%d Weights Reset for %T: Breaking Semantic Sink.\n", expertIdx, moe)

	expert := moe.Experts[expertIdx]
	params := expert.Parameters()
	for _, p := range params {
		if p == nil {
			continue
		}
		// Xavier Initialization
		var fanIn int
		if len(p.Shape) >= 2 {
			fanIn = p.Shape[0]
		} else {
			fanIn = p.Shape[0]
		}
		stdDev := float32(math.Sqrt(1.0 / float64(fanIn)))
		for i := range p.Data {
			p.Data[i] = float32(rand.NormFloat64()) * stdDev
		}
	}
}

// PerformSurgery clones weights from an alpha expert to a sink expert with tiny mutation noise.
func (moe *MoELayer) PerformSurgery(alphaID, sinkID int) {
	if alphaID < 0 || alphaID >= len(moe.Experts) || sinkID < 0 || sinkID >= len(moe.Experts) {
		return
	}
	if alphaID == sinkID {
		return
	}

	fmt.Printf("🧬 [Surgery] Cloning Expert E%d (Alpha) -> Expert E%d (Sink) for %T\n", alphaID, sinkID, moe)

	alphaParams := moe.Experts[alphaID].Parameters()
	sinkParams := moe.Experts[sinkID].Parameters()

	for i := range sinkParams {
		if i >= len(alphaParams) {
			break
		}
		if sinkParams[i] == nil || alphaParams[i] == nil {
			continue
		}

		// Clone and add tiny mutation jitter (0.01)
		copy(sinkParams[i].Data, alphaParams[i].Data)
		sinkParams[i].ApplyJitter(0.01)
		sinkParams[i].ZeroGrad()
	}
}

// HealExpert resets an expert's weights slightly toward the mean of 'healthy' experts.
func (moe *MoELayer) HealExpert(expertIdx int, healthyIDs []int) {
	if expertIdx < 0 || expertIdx >= len(moe.Experts) || len(healthyIDs) == 0 {
		return
	}

	fmt.Printf("🏥 [Heal] Expert E%d is recovering. Blending weights with healthy experts: %v\n", expertIdx, healthyIDs)

	targetParams := moe.Experts[expertIdx].Parameters()

	// Create mean parameters from healthy experts
	for i := range targetParams {
		if targetParams[i] == nil {
			continue
		}

		meanData := make([]float32, len(targetParams[i].Data))
		count := 0
		for _, hID := range healthyIDs {
			if hID < 0 || hID >= len(moe.Experts) || hID == expertIdx {
				continue
			}
			hParams := moe.Experts[hID].Parameters()
			if i < len(hParams) && hParams[i] != nil && len(hParams[i].Data) == len(targetParams[i].Data) {
				for j, v := range hParams[i].Data {
					meanData[j] += v
				}
				count++
			}
		}

		if count > 0 {
			invCount := 1.0 / float32(count)
			for j := range meanData {
				meanData[j] *= invCount

				// Genetic Blending: 70% mean of healthy, 30% original (with jitter)
				// This gives them a "fresh start" without wiping all their learned features.
				targetParams[i].Data[j] = (meanData[j] * 0.7) + (targetParams[i].Data[j] * 0.3)
			}
			targetParams[i].ApplyJitter(0.02)
			targetParams[i].ZeroGrad()
		}
	}
}

// ValidateHealth triggers a health check based on accumulated utilization.
func (moe *MoELayer) ValidateHealth(label string) {
	if moe.AccumulatedUtilization == nil {
		return
	}
	ValidateExpertHealth(label, moe.AccumulatedUtilization)
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

// SetGateTemperature updates the router softmax temperature.
func (moe *MoELayer) SetGateTemperature(temp float32) {
	moe.RouterTemperature = temp
}

func (moe *MoELayer) SyncParameters() error {
	var wg sync.WaitGroup
	errCh := make(chan error, len(moe.Experts)+1)

	if moe.GatingNetwork != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := moe.GatingNetwork.SyncParameters(); err != nil {
				errCh <- err
			}
		}()
	}

	for _, expert := range moe.Experts {
		if expert != nil {
			wg.Add(1)
			go func(ex Expert) {
				defer wg.Done()
				if err := ex.SyncParameters(); err != nil {
					errCh <- err
				}
			}(expert)
		}
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	return nil
}

// ToGPU moves the parameters to the GPU.
func (moe *MoELayer) ToGPU() {
	if moe.GatingNetwork != nil {
		moe.GatingNetwork.ToGPU()
	}
	for _, expert := range moe.Experts {
		if expert != nil {
			expert.ToGPU()
		}
	}
	if moe.ResidualScale != nil {
		moe.ResidualScale.ToGPU()
	}
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
	// 🛡️ NUMERICAL SAFETY: Check for NaNs in router logits
	if len(gateLogits.Data) > 0 && math.IsNaN(float64(gateLogits.Data[0])) {
		fmt.Printf("⚠️ [MoELayer] NaNs detected in gateLogits! Resetting to small random values.\n")
		for i := range gateLogits.Data {
			gateLogits.Data[i] = (rand.Float32() * 0.02) - 0.01
		}
	}
	moe.RouterZLoss = CalculateRouterZLoss(gateLogits)
	moe.gateLogits = gateLogits // Store raw logits for BPTT

	// --- [Diagnostic] ---
	// Check input health to debug "Ghost Town" layers
	inputNorm := float32(math.Sqrt(float64(DotProduct(input.Data, input.Data))))
	if inputNorm < 1e-6 {
		fmt.Printf("⚠️ [MoELayer Diagnostic] Signal Collapse! Input L2 Norm: %.6f\n", inputNorm)
	}
	// --- [/Diagnostic] ---

	embeddingDim := input.Shape[2]
	numExperts := len(moe.Experts)

	// 🧬 RE-SYNC: Ensure utilization tracking matches current expert count
	if moe.AccumulatedUtilization == nil || len(moe.AccumulatedUtilization) != numExperts {
		moe.ResetUtilizationStats()
	}
	batchSize := input.Shape[0]
	seqLength := input.Shape[1]

	// Noise injection (Expert Curiosity) is now handled inside GatingNetwork.Forward
	// to ensure consistent Gaussian jitter before TopK selection.

	// Apply Temperature Scaling to logits
	// Default to 0.5 sharpening as requested ("divide by 0.5")
	routerScale := float32(1.0 / 0.5)
	if moe.RouterTemperature > 0 {
		routerScale = 1.0 / moe.RouterTemperature
	}
	SimdScaleF32(gateLogits.Data, routerScale)

	// --- [Penalty Mask for Over-Used Experts] ---
	// Aggressive Penalty for Dominant Experts (Router Level)
	if moe.Training {
		var totalTokensProcessed int
		for _, u := range moe.AccumulatedUtilization {
			totalTokensProcessed += u
		}
		if totalTokensProcessed > 200 { // Start earlier (was 1000)
			avgTokens := float32(totalTokensProcessed) / float32(numExperts)
			for i := 0; i < batchSize*seqLength; i++ {
				for j := 0; j < numExperts; j++ {
					usage := float32(moe.AccumulatedUtilization[j])
					if usage > avgTokens*1.1 { // If 10% over average
						// Apply exponential penalty to discourage this expert
						ratio := usage / avgTokens
						penalty := float32(math.Pow(float64(ratio), 2.0)) * 5.0
						gateLogits.Data[i*numExperts+j] -= penalty
					}
				}
			}
		}

		// 🎲 RANDOM EXPERT SHUFFLE (Training Only)
		// 15% of the time, zero out the top expert's logit to force model to learn alternatives
		if rand.Float32() < 0.15 {
			for i := 0; i < batchSize*seqLength; i++ {
				maxIdx := 0
				maxVal := gateLogits.Data[i*numExperts]
				for j := 1; j < numExperts; j++ {
					if gateLogits.Data[i*numExperts+j] > maxVal {
						maxVal = gateLogits.Data[i*numExperts+j]
						maxIdx = j
					}
				}
				gateLogits.Data[i*numExperts+maxIdx] = -1e9 // Temporary dropout
			}
		}
	}

	// Apply Expert Dropout during training
	if moe.Training && moe.ExpertDropoutRate > 0 {
		// Decide which experts to drop for this batch
		droppedMask := make([]bool, numExperts)
		activeCount := 0

		// Targeted Dropout: random zeroing of Overactive experts (E0, E1, E3, E4, E6, E11, E13)
		// as requested by user to force specialization in E8-E12.
		overactive := map[int]bool{0: true, 1: true, 3: true, 4: true, 6: true, 11: true, 13: true}

		for i := 0; i < numExperts; i++ {
			if overactive[i] {
				if rand.Float32() < 0.20 { // Increased to 20% to force breakout
					droppedMask[i] = true
				}
			} else if rand.Float32() < moe.ExpertDropoutRate*0.5 { // Normal dropout for others
				droppedMask[i] = true
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
		grpoLogits := make([]float32, len(gateLogits.Data))
		for i := 0; i < batchSize*seqLength; i++ {
			tokenLogits := gateLogits.Data[i*numExperts : (i+1)*numExperts]

			// Calculate Mean
			var sum float32
			for _, val := range tokenLogits {
				sum += val
			}
			mean := sum / float32(numExperts)

			// Calculate StdDev
			var sqDiffSum float32
			for _, val := range tokenLogits {
				diff := val - mean
				sqDiffSum += diff * diff
			}
			std := float32(math.Sqrt(float64(sqDiffSum / float32(numExperts))))
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

	// 🧬 EXPERT PERSISTENCE (Bias for N+1)
	// If we have stored experts from the previous step, bias the router toward them
	// to prevent "expert jumping" mid-phrase.
	numTokens := batchSize * seqLength

	if batchSize > 0 {
		if moe.LastSelectedExperts == nil || len(moe.LastSelectedExperts) != batchSize {
			moe.LastSelectedExperts = make([][]int, batchSize)
		}

		for b := 0; b < batchSize; b++ {
			for t := 0; t < seqLength; t++ {
				tokenIdx := b*seqLength + t

				// Identify experts to bias for this token
				var expertsToBias []int
				if t > 0 {
					// Within a sequence (Training): bias toward previous token's target expert
					if moe.Training && len(moe.TargetRouting) > tokenIdx-1 {
						target := moe.TargetRouting[tokenIdx-1]
						if target >= 0 && target < numExperts {
							expertsToBias = []int{target}
						}
					}
					// Note: during parallel Forward, we don't have current pass's SelectedExperts yet for t > 0
				} else {
					// Start of sequence or Inference (seqLen=1): bias toward saved experts from previous Forward call
					expertsToBias = moe.LastSelectedExperts[b]
				}

				for _, eid := range expertsToBias {
					if eid >= 0 && eid < numExperts {
						gateLogits.Data[tokenIdx*numExperts+eid] += moe.PersistenceBias
					}
				}
			}
		}
		// 🧬 STRUCTURAL ROUTING BIAS (Training Only)
		// If ground-truth routing is provided (from POS tags), boost the target expert.
		if moe.Training && len(moe.TargetRouting) > 0 {
			for i := 0; i < numTokens && i < len(moe.TargetRouting); i++ {
				target := moe.TargetRouting[i]
				if target >= 0 && target < numExperts {
					// Apply a structural prior (configurable boost)
					boost := moe.StructuralBiasIntensity
					if boost == 0 {
						boost = 8.0
					}
					gateLogits.Data[i*numExperts+target] += boost
				}
			}
		}

		// 🧬 STEP-AWARE ROUTING BIAS
		// Apply manual expert nudges for specific steps (e.g., forcing E14:GREET at Step 0).
		for i := 0; i < numTokens; i++ {
			stepIdx := i % seqLength
			if bias, ok := moe.StepRoutingBias[stepIdx]; ok {
				base := i * numExperts
				for e, b := range bias {
					if e < numExperts {
						gateLogits.Data[base+e] += b
					}
				}
			}
		}

		// Apply softmax to get probabilities
		GateOutputs, err := scoresTensor.Softmax(len(scoresTensor.Shape) - 1)
		if err != nil {
			return nil, fmt.Errorf("gating network softmax failed: %w", err)
		}

		// 🛡️ NUMERICAL SAFETY: Check for NaNs in softmax output
		if len(GateOutputs.Data) > 0 && math.IsNaN(float64(GateOutputs.Data[0])) {
			fmt.Printf("⚠️ [MoELayer] NaNs detected in GateOutputs! Recovering with uniform distribution.\n")
			uniform := 1.0 / float32(numExperts)
			for i := range GateOutputs.Data {
				GateOutputs.Data[i] = uniform
			}
		}
		moe.GateOutputs = GateOutputs

		// 4. Sum gating probabilities early (used for LoadBalancingLoss later)
		moe.ExpertProbSums = make([]float32, numExperts)
		if numTokens > 0 {
			for i := 0; i < numTokens; i++ {
				AddAccumulate(moe.ExpertProbSums, GateOutputs.Data[i*numExperts:(i+1)*numExperts])
			}
		}

		// 5. Hard Top-K selection and Gating Probability Zeroing

		// Calculate capacity limit per expert
		capacity := int(math.Ceil(float64(moe.CapacityFactor * float32(batchSize*seqLength) / float32(numExperts))))
		if capacity < 1 {
			capacity = 1
		}

		moe.SelectedExperts = make([][]int, batchSize*seqLength)
		moe.ExpertTokenIndices = make([][]int, numExperts)
		for i := range moe.ExpertTokenIndices {
			moe.ExpertTokenIndices[i] = make([]int, 0, capacity)
		}

		// Reshape input to 2D [batch*seq, dim] for gathering
		input2D, err := input.Reshape([]int{batchSize * seqLength, embeddingDim})
		if err != nil {
			return nil, fmt.Errorf("failed to reshape input to 2D: %w", err)
		}

		// Store relative indices for scatter step
		tokenExpertRelativeIndices := make([][]int, batchSize*seqLength)

		// --- Optimized Expert Routing decision (Low Allocation) ---
		var wgRoute sync.WaitGroup
		numWorkersRoute := runtime.NumCPU()
		if numWorkersRoute > 16 {
			numWorkersRoute = 16
		}
		totalTokensRoute := batchSize * seqLength
		// 🔍 Structural Diagnostic: Verify routing integrity

		tokensPerWorkerRoute := (totalTokensRoute + numWorkersRoute - 1) / numWorkersRoute

		moe.TopExpertIDs = make([]int, totalTokensRoute)
		allSelectedExperts := make([][]int, totalTokensRoute)

		if moe.SoftRouting {
			// Soft Routing: Select ALL experts for ALL tokens
			for i := 0; i < totalTokensRoute; i++ {
				moe.TopExpertIDs[i] = 0 // Just for diagnostics
				selected := make([]int, numExperts)
				for j := range numExperts {
					selected[j] = j
				}
				allSelectedExperts[i] = selected
			}
		} else {
			for w := 0; w < numWorkersRoute; w++ {
				start := w * tokensPerWorkerRoute
				end := min(start+tokensPerWorkerRoute, totalTokensRoute)
				if start >= end {
					break
				}

				wgRoute.Add(1)
				go func(tokenStart, tokenEnd int) {
					defer wgRoute.Done()

					for i := tokenStart; i < tokenEnd; i++ {
						scores := scoresTensor.Data[i*numExperts : (i+1)*numExperts]

						// Fast Top-K (manual for small K)
						e1, e2 := -1, -1
						v1, v2 := float32(-1e30), float32(-1e30)

						for j, score := range scores {
							// 🎲 Tie-breaker jitter: Add a tiny random value to break the E0 monopoly on ties
							jitteredScore := score + (rand.Float32() * 1e-6)
							if jitteredScore > v1 {
								v2 = v1
								e2 = e1
								v1 = jitteredScore
								e1 = j
							} else if jitteredScore > v2 {
								v2 = jitteredScore
								e2 = j
							}
						}

						moe.TopExpertIDs[i] = e1
						selected := []int{e1}
						if moe.K > 1 && e2 != -1 {
							selected = append(selected, e2)
						}
						allSelectedExperts[i] = selected

						// Re-normalize probabilities (in-place)
						var rowSum float32
						for j := 0; j < numExperts; j++ {
							idx := i*numExperts + j
							if j != e1 && (moe.K < 2 || j != e2) {
								GateOutputs.Data[idx] = 0
							}
							rowSum += GateOutputs.Data[idx]
						}
						if rowSum > 1e-12 {
							invSum := 1.0 / rowSum
							for j := 0; j < numExperts; j++ {
								GateOutputs.Data[i*numExperts+j] *= invSum
							}
						}
					}
				}(start, end)
			}
			wgRoute.Wait()
		}

		// 🧬 STAGNATION RECOVERY (Training Only)
		if moe.Training && !moe.OverfitMode {
			batchUsed := make([]bool, numExperts)
			for _, selected := range allSelectedExperts {
				for _, eid := range selected {
					if eid >= 0 && eid < numExperts {
						batchUsed[eid] = true
					}
				}
			}

			for j := 0; j < numExperts; j++ {
				if !batchUsed[j] {
					moe.StagnationCounters[j] += totalTokensRoute
					if moe.StagnationCounters[j] >= 500 { // Stagnant for 500 tokens
						moe.ResetExpertWeights(j)
						moe.StagnationCounters[j] = 0
					}
				} else {
					moe.StagnationCounters[j] = 0
				}
			}
		}

		moe.SelectedExperts = allSelectedExperts
		// Clear and re-fill ExpertTokenIndices sequentially to ensure correct relative indexing
		for i := range moe.ExpertTokenIndices {
			moe.ExpertTokenIndices[i] = moe.ExpertTokenIndices[i][:0]
		}

		tokenExpertRelativeIndices = make([][]int, totalTokensRoute)
		for i, selected := range allSelectedExperts {
			tokenExpertRelativeIndices[i] = make([]int, moe.K)
			// Initialize with -1
			for j := range tokenExpertRelativeIndices[i] {
				tokenExpertRelativeIndices[i][j] = -1
			}

			for j, expertIdx := range selected {
				if j >= moe.K {
					break
				}
				// Assign tokens to experts while respecting capacity (sequentially for safety)
				if len(moe.ExpertTokenIndices[expertIdx]) < capacity*2 { // permissive limit
					tokenExpertRelativeIndices[i][j] = len(moe.ExpertTokenIndices[expertIdx])
					moe.ExpertTokenIndices[expertIdx] = append(moe.ExpertTokenIndices[expertIdx], i)

					// CRITICAL FIX: Track utilization for health monitoring and resets
					if moe.AccumulatedUtilization == nil || len(moe.AccumulatedUtilization) != numExperts {
						moe.AccumulatedUtilization = make([]int, numExperts)
					}
					moe.AccumulatedUtilization[expertIdx]++
				}
			}
		}

		moe.expertOutputs = make([]*Tensor, numExperts)
		var firstErr error

		// 3. Parallel Expert Execution (Using Iterator Pattern)
		var wg sync.WaitGroup
		var errOnce sync.Once

		for task := range moe.ExpertTasks() {
			wg.Add(1)
			go func(t ExpertTask) {
				defer wg.Done()

				// Gather inputs for this expert
				batchedInput, err := input2D.Gather(t.TokenIndices)
				if err != nil {
					errOnce.Do(func() { firstErr = fmt.Errorf("failed to gather inputs for expert %d: %w", t.ExpertIdx, err) })
					return
				}

				// Forward pass
				output, err := moe.Experts[t.ExpertIdx].Forward(batchedInput)
				if err != nil {
					errOnce.Do(func() { firstErr = fmt.Errorf("expert %d forward failed: %w", t.ExpertIdx, err) })
					return
				}

				// --- Activation Clipping ---
				output.Clip(-15.0, 15.0)

				// --- Expert Output Scaling ---
				if len(moe.ExpertOutputScale) > t.ExpertIdx && moe.ExpertOutputScale[t.ExpertIdx] != 1.0 {
					SimdScaleF32(output.Data, moe.ExpertOutputScale[t.ExpertIdx])
				} else if t.ExpertIdx == 1 {
					SimdScaleF32(output.Data, 1.1)
				}

				moe.expertOutputs[t.ExpertIdx] = output
			}(task)
		}
		wg.Wait()
		// fmt.Println("Finished parallel expert execution (Forward)")

		if firstErr != nil {
			return nil, firstErr
		}

		// Scatter results back to final output
		finalOutput := NewTensor([]int{batchSize, seqLength, moe.OutputDim}, make([]float32, batchSize*seqLength*moe.OutputDim), true)

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
					selected := moe.SelectedExperts[i]
					outStart := i * moe.OutputDim
					outRow := finalOutput.Data[outStart : outStart+moe.OutputDim]

					// Optimized scattering for Top-K
					if len(selected) == 2 {
						e0, e1 := selected[0], selected[1]

						w0 := GateOutputs.Data[i*numExperts+e0]
						w1 := GateOutputs.Data[i*numExperts+e1]

						r0 := tokenExpertRelativeIndices[i][0]
						r1 := tokenExpertRelativeIndices[i][1]

						if r0 != -1 && moe.expertOutputs[e0] != nil {
							expertRow := moe.expertOutputs[e0].Data[r0*moe.OutputDim : (r0+1)*moe.OutputDim]
							SimdAddScalarMulF32(outRow, expertRow, w0)
						}
						if r1 != -1 && moe.expertOutputs[e1] != nil {
							expertRow := moe.expertOutputs[e1].Data[r1*moe.OutputDim : (r1+1)*moe.OutputDim]
							SimdAddScalarMulF32(outRow, expertRow, w1)
						}
					} else {
						for j, expertIdx := range selected {
							output := moe.expertOutputs[expertIdx]
							if output == nil {
								continue
							}

							weight := GateOutputs.Data[i*numExperts+expertIdx]
							relativeRow := tokenExpertRelativeIndices[i][j]
							if relativeRow == -1 {
								continue
							}
							expertRow := output.Data[relativeRow*moe.OutputDim : (relativeRow+1)*moe.OutputDim]
							SimdAddScalarMulF32(outRow, expertRow, weight)
						}
					}
				}
			}(startToken, endToken)
		}
		wgScatter.Wait()

		// Finalize Load Balancing Loss
		if numTokens > 0 {
			stLoss := float32(0.0)
			for e := 0; e < numExperts; e++ {
				fraction := float32(len(moe.ExpertTokenIndices[e])) / (float32(numTokens) + 1e-8)
				meanProb := moe.ExpertProbSums[e] / (float32(numTokens) + 1e-8)
				stLoss += fraction * meanProb
			}
			stLoss *= float32(numExperts)
			divLoss := moe.CalculateDiversityLoss()
			moe.DiversityLoss = divLoss

			// 4. Router Load Balancing Diversity (Entropy-Based)
			routerDivLoss := moe.GatingNetwork.CalculateDiversityLoss()

			// 5. Direct Shannon Entropy (Sharpness vs Fairness)
			shannonEntropy := CalculateDiversityLoss(moe.GateOutputs)

			// Combine them (stLoss, divLoss, and routerDivLoss)
			// Increased weight for CV loss (routerDivLoss) to force diverse selection
			moe.LoadBalancingLoss = 0.4*stLoss + 0.2*divLoss + 0.3*routerDivLoss + 0.1*shannonEntropy
		} else {
			moe.LoadBalancingLoss = 0
			moe.DiversityLoss = 0
		}

		// Update Expert Health (EMA)
		for i, expert := range moe.Experts {
			wasUsed := len(moe.ExpertTokenIndices[i]) > 0
			expert.UpdateHealth(wasUsed)
		}

		// Push state to stack for BPTT
		if finalOutput.RequiresGrad {
			state := MoEState{
				inputTensor:        moe.inputTensor,
				input2D:            input2D,
				expertOutputs:      moe.expertOutputs,
				ExpertTokenIndices: moe.ExpertTokenIndices,
				SelectedExperts:    moe.SelectedExperts,
				GateOutputs:        moe.GateOutputs,
				ExpertProbSums:     moe.ExpertProbSums,
				LoadBalancingLoss:  moe.LoadBalancingLoss,
				RouterZLoss:        moe.RouterZLoss,
				gateLogits:         moe.gateLogits,
				lastOutput:         finalOutput,
				TargetRouting:      moe.TargetRouting,
			}
			moe.stateStack = append(moe.stateStack, state)
		}

		// 🧬 Update Persistence Memory for next call
		if batchSize > 0 && len(moe.SelectedExperts) >= batchSize*seqLength {
			for b := 0; b < batchSize; b++ {
				lastTokenIdx := (b+1)*seqLength - 1
				moe.LastSelectedExperts[b] = moe.SelectedExperts[lastTokenIdx]
			}
		}
		return finalOutput, nil
	}
	return nil, nil
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
		moe.ExpertTokenIndices = state.ExpertTokenIndices
		moe.SelectedExperts = state.SelectedExperts
		moe.GateOutputs = state.GateOutputs
		moe.ExpertProbSums = state.ExpertProbSums
		moe.LoadBalancingLoss = state.LoadBalancingLoss
		moe.RouterZLoss = state.RouterZLoss
		moe.gateLogits = state.gateLogits
		moe.TargetRouting = state.TargetRouting
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
		fullGrad := NewTensor(fullGradShape, make([]float32, fullGradSize), false)

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
			newGrad := NewTensor(moe.inputTensor.Shape, make([]float32, newSize), false)

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
			moe.inputTensor.Grad = NewTensor(moe.inputTensor.Shape, make([]float32, len(moe.inputTensor.Data)), false)
		}
	}

	// Reshape grad to be [batchSize*seqLength, embeddingDim]
	gradReshaped, err := grad.Reshape([]int{batchSize * seqLength, embeddingDim})
	if err != nil {
		return fmt.Errorf("failed to reshape grad: %w", err)
	}

	// Initialize a tensor to accumulate gradients for the gating network
	gateGradReshaped := NewTensor([]int{batchSize * seqLength, numExperts}, make([]float32, batchSize*seqLength*numExperts), true)

	// fmt.Println("Starting parallel expert execution (Backward)")
	// Prepare gradients for each expert
	// We need to group gradients exactly as we grouped inputs in Forward
	// moe.ExpertTokenIndices has the mapping

	var wg sync.WaitGroup
	var errOnce sync.Once
	var inputGradMutex sync.Mutex
	var firstErr error

	// Run experts backward in parallel (Using Iterator Pattern)
	for task := range moe.ExpertTasks() {
		wg.Add(1)
		go func(t ExpertTask) {
			defer wg.Done()

			// Gather gradients for this expert
			batchedGrad, err := gradReshaped.Gather(t.TokenIndices)
			if err != nil {
				errOnce.Do(func() { firstErr = fmt.Errorf("failed to gather grads for expert %d: %w", t.ExpertIdx, err) })
				return
			}

			// Prepare weighted gradients
			weightedBatchedGradData := make([]float32, len(batchedGrad.Data))
			for k, tokenIdx := range t.TokenIndices {
				gateIdx := tokenIdx*numExperts + t.ExpertIdx
				weight := moe.GateOutputs.Data[gateIdx]
				SimdMulScalarF32(weightedBatchedGradData[k*embeddingDim:(k+1)*embeddingDim], batchedGrad.Data[k*embeddingDim:(k+1)*embeddingDim], weight)
			}
			weightedBatchedGrad := NewTensor(batchedGrad.Shape, weightedBatchedGradData, false)

			// Backward pass
			err = moe.Experts[t.ExpertIdx].Backward(weightedBatchedGrad)
			if err != nil {
				errOnce.Do(func() { firstErr = fmt.Errorf("expert %d backward failed: %w", t.ExpertIdx, err) })
				return
			}

			// --- Expert Grad Multiplier ---
			if moe.ExpertGradMultiplier != nil && moe.ExpertGradMultiplier[t.ExpertIdx] > 1.0 {
				params := moe.Experts[t.ExpertIdx].Parameters()
				for _, p := range params {
					if p.Grad != nil {
						SimdScaleF32(p.Grad.Data, moe.ExpertGradMultiplier[t.ExpertIdx])
					}
				}
				moe.ExpertGradMultiplier[t.ExpertIdx] *= 0.98
				if moe.ExpertGradMultiplier[t.ExpertIdx] < 1.02 {
					moe.ExpertGradMultiplier[t.ExpertIdx] = 1.0
				}
			}

			// --- Expert Surgery: Expert-Specific Backprop ---
			if t.ExpertIdx == 1 {
				params := moe.Experts[t.ExpertIdx].Parameters()
				for _, p := range params {
					if p.Grad != nil {
						SimdScaleF32(p.Grad.Data, 1.2)
					}
				}
			}

			// Accumulate input gradients via GatherOperation.Backward
			if moe.inputTensor.RequiresGrad {
				expertInputs := moe.Experts[t.ExpertIdx].Inputs()
				if len(expertInputs) > 0 {
					gatheredInput := expertInputs[0]
					if gatheredInput.Creator != nil && gatheredInput.Grad != nil {
						inputGradMutex.Lock()
						gatheredInput.Creator.Backward(gatheredInput.Grad)
						inputGradMutex.Unlock()
					}
				}
			}

			// Accumulate gating gradients
			expertOutput := moe.expertOutputs[t.ExpertIdx]
			for k, tokenIdx := range t.TokenIndices {
				startIdx := tokenIdx * embeddingDim
				gradForTokenData := gradReshaped.Data[startIdx : startIdx+embeddingDim]
				expertOutRow := expertOutput.Data[k*embeddingDim : (k+1)*embeddingDim]

				gradForGateProb := SimdDotProductF32(gradForTokenData, expertOutRow)
				gateGradReshaped.Data[tokenIdx*numExperts+t.ExpertIdx] += gradForGateProb
			}
		}(task)
	}
	wg.Wait()
	// fmt.Println("Finished parallel expert execution (Backward)")

	if firstErr != nil {
		return firstErr
	}

	// [Removed redundant redundant LB gradient here as it is handled below]

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
				moe.inputTensor.Grad = NewTensor(moe.inputTensor.Shape, make([]float32, len(moe.inputTensor.Data)), false)
			}
			safeAccumulate(moe.inputTensor.Grad.Data, input2D.Grad.Data)
		}
	}

	// Finally, backpropagate through the gating network with the accumulated gateGrad.
	// Workaround: GatingNetwork.Backward (Linear.Backward) seems to cause moe.inputTensor.Grad to become nil
	// or overwritten in some cases. We explicitly copy the expert gradients to preserve them.
	var expertGrads []float32
	if moe.inputTensor.Grad != nil {
		expertGrads = make([]float32, len(moe.inputTensor.Grad.Data))
		copy(expertGrads, moe.inputTensor.Grad.Data)
	}

	// Convert probability gradients to logit gradients (Softmax Backward)
	logitsGrad := NewTensor(gateGradReshaped.Shape, make([]float32, len(gateGradReshaped.Data)), false)

	// --- [Auxiliary Loss Gradient (Manual Inject) BEFORE Softmax Backward] ---
	// Combined Loss: 0.5 * SwitchTransformerLoss + 0.5 * CV^2_Importance
	// dST/dP_ie = (N/T^2) * n_e
	// dCV2/dP_ie = (2N/T^2) * (I_e - T/N)
	numTokens := batchSize * seqLength
	if moe.LoadBalancingWeight > 0 && numTokens > 0 {
		numExperts := len(moe.Experts)
		T := float32(numTokens)
		N := float32(numExperts)

		// Base scalar (N/T^2) * Weight
		baseScalar := moe.LoadBalancingWeight * (N / (T * T))

		scaledFractions := make([]float32, numExperts)
		for e := 0; e < numExperts; e++ {
			// n_e is the number of tokens assigned to expert e
			n_e := float32(len(moe.ExpertTokenIndices[e]))
			// I_e is the sum of probabilities for expert e
			I_e := moe.ExpertProbSums[e]

			// Gradient from ST loss: 0.5 * (N/T^2) * n_e
			stGrad := 0.5 * n_e * baseScalar
			// Gradient from CV^2 Importance loss: 0.5 * (2N/T^2) * (I_e - T/N)
			auxGrad := 0.5 * 2.0 * (I_e - T/N) * baseScalar

			scaledFractions[e] = stGrad + auxGrad
		}

		// Prepare 2D view for UpdateRouterGrads
		gateGrads2D := make([][]float32, numTokens)
		for i := 0; i < numTokens; i++ {
			gateGrads2D[i] = gateGradReshaped.Data[i*numExperts : (i+1)*numExperts]
		}

		moe.UpdateRouterGrads(gateGrads2D, scaledFractions)
	}

	// Parallelize softmax backward
	var wgSoftmax sync.WaitGroup
	numWorkers := runtime.NumCPU()
	rowsPerWorker := (batchSize*seqLength + numWorkers - 1) / numWorkers
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
				p := moe.GateOutputs.Data[offset : offset+numExperts]
				dp := gateGradReshaped.Data[offset : offset+numExperts]
				out := logitsGrad.Data[offset : offset+numExperts]

				SoftmaxBackwardRow(p, dp, out) // Standard Softmax Backward (Generic)
			}
		}(start, end)
	}
	wgSoftmax.Wait()

	// Apply Temperature Scaling to gradients (dL/dx = dL/dy * 1/T)
	if moe.RouterTemperature != 1.0 && moe.RouterTemperature > 0 {
		SimdScaleF32(logitsGrad.Data, 1.0/moe.RouterTemperature)
	}

	// 4. Add Router Z-Loss Gradient (dL_z / d_logits)
	// L_z = (sum(logits^2) / n) * 0.0001
	// dL_z / d_logit_i = (2 * logit_i / n) * 0.0001
	if moe.gateLogits != nil && len(moe.gateLogits.Data) == len(logitsGrad.Data) {
		n := float32(len(moe.gateLogits.Data))
		const c = float32(0.0001)
		for i := range logitsGrad.Data {
			logitsGrad.Data[i] += (2.0 * moe.gateLogits.Data[i] / n) * c
		}
	}

	// --- [STRUCTURAL GRAMMAR ROUTING LOSS] ---
	// If TargetRouting is provided, we apply a direct Cross-Entropy gradient
	// to the gate logits to force the router toward specific grammar experts.
	if len(moe.TargetRouting) > 0 && len(logitsGrad.Data) > 0 {
		// Logits gradient for Cross-Entropy + Softmax is (p_i - target_i)
		// We add this to the existing gradient from the main loss.
		numExperts := len(moe.Experts)
		numTokens := len(moe.TargetRouting)
		routingWeight := moe.StructuralRoutingWeight

		for i := 0; i < numTokens; i++ {
			targetIdx := moe.TargetRouting[i]
			if targetIdx < 0 || targetIdx >= numExperts {
				continue // Skip invalid or padding roles
			}

			// Add (p_i - 1) to the target expert's logit grad,
			// and (p_j) to all other expert's logit grads.
			for j := 0; j < numExperts; j++ {
				gradIdx := i*numExperts + j
				if gradIdx >= len(logitsGrad.Data) {
					break
				}

				p := moe.GateOutputs.Data[gradIdx]
				var dLdz float32
				if j == targetIdx {
					dLdz = (p - 1.0)
				} else {
					dLdz = p
				}

				logitsGrad.Data[gradIdx] += dLdz * routingWeight
			}
		}
	}

	// --- [LOAD BALANCING / SPARSITY PENALTY] ---
	// If ExpertSparsityWeight is provided, we apply a penalty to discourage expert monopolies.
	// We want the average probability of each expert to be roughly equal (1/N).
	if moe.ExpertSparsityWeight > 0 && len(logitsGrad.Data) > 0 {
		numExperts := len(moe.Experts)
		numTokens := len(moe.GateOutputs.Data) / numExperts
		if numTokens > 0 {
			// Compute mean probability for each expert across all tokens in this batch
			meanProbs := make([]float32, numExperts)
			for i := 0; i < numTokens; i++ {
				for j := 0; j < numExperts; j++ {
					meanProbs[j] += moe.GateOutputs.Data[i*numExperts+j]
				}
			}
			for j := 0; j < numExperts; j++ {
				meanProbs[j] /= float32(numTokens)
			}

			// Gradient for Balance Loss (simplified):
			// dL/dp_ij = lambda * mean_prob_j
			// This encourages reducing probabilities for experts that already have high mean prob.
			lambda := moe.ExpertSparsityWeight
			for i := 0; i < numTokens; i++ {
				for j := 0; j < numExperts; j++ {
					gradIdx := i*numExperts + j
					logitsGrad.Data[gradIdx] += meanProbs[j] * lambda
				}
			}
		}
	}

	err = moe.GatingNetwork.Backward(logitsGrad)
	if err != nil {
		return err
	}

	// --- [EXPERT WEIGHT REGULARIZATION] ---
	// If an expert's weights deviate too far from the mean of "healthy" experts, apply a penalty.
	if moe.ExpertRegularizationWeight > 0 && len(moe.HealthyExpertIDs) > 0 {
		moe.applyExpertRegularization()
	}

	// Expert gradients were already accumulated into moe.inputTensor.Grad via input2D.Grad
	// in the loop above. Double accumulation removed to prevent gradient explosion.

	// Gradient is stored in moe.inputTensor.Grad
	return nil
}

// applyExpertRegularization computes the mean weights of healthy experts and adds a penalty gradient
// to experts whose weights deviate too far from this mean.
func (moe *MoELayer) applyExpertRegularization() {
	if len(moe.HealthyExpertIDs) == 0 {
		return
	}

	// 1. Identify healthy experts and compute their mean parameters
	numParams := len(moe.Experts[0].Parameters())
	means := make([][]float32, numParams)
	for i := 0; i < numParams; i++ {
		p := moe.Experts[0].Parameters()[i]
		means[i] = make([]float32, len(p.Data))
	}

	healthyCount := 0
	for _, hID := range moe.HealthyExpertIDs {
		if hID < 0 || hID >= len(moe.Experts) {
			continue
		}
		params := moe.Experts[hID].Parameters()
		for i, p := range params {
			if i < numParams && p != nil {
				for j, v := range p.Data {
					means[i][j] += v
				}
			}
		}
		healthyCount++
	}

	if healthyCount == 0 {
		return
	}

	// Average the means
	invCount := 1.0 / float32(healthyCount)
	for i := range means {
		for j := range means[i] {
			means[i][j] *= invCount
		}
	}

	// 2. Apply penalty gradient to experts NOT in the healthy set
	isHealthy := make([]bool, len(moe.Experts))
	for _, id := range moe.HealthyExpertIDs {
		if id >= 0 && id < len(isHealthy) {
			isHealthy[id] = true
		}
	}

	lambda := moe.ExpertRegularizationWeight
	for id, expert := range moe.Experts {
		if isHealthy[id] {
			continue // Don't penalize the anchors
		}

		params := expert.Parameters()
		for i, p := range params {
			if i < numParams && p != nil && p.Grad != nil {
				// Penalty gradient: 2 * lambda * (W - mean)
				for j := range p.Data {
					diff := p.Data[j] - means[i][j]
					p.Grad.Data[j] += 2.0 * lambda * diff
				}
			}
		}
	}
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
	if moe.GatingNetwork != nil {
		moe.GatingNetwork.Training = training
	}
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
	return moe.SelectedExperts
}

// ClearState clears the internal state of the layer to free memory.
func (moe *MoELayer) ClearState() {
	if moe.GateOutputs != nil {
		moe.GateOutputs.Release()
	}
	if moe.gateLogits != nil {
		moe.gateLogits.Release()
	}
	for _, out := range moe.expertOutputs {
		if out != nil {
			out.Release()
		}
	}
	for _, state := range moe.stateStack {
		if state.input2D != nil {
			state.input2D.Release()
		}
		if state.GateOutputs != nil {
			state.GateOutputs.Release()
		}
		if state.gateLogits != nil {
			state.gateLogits.Release()
		}
		for _, exOut := range state.expertOutputs {
			if exOut != nil {
				exOut.Release()
			}
		}
	}

	moe.inputTensor = nil
	moe.expertOutputs = nil
	moe.ExpertTokenIndices = nil
	moe.SelectedExperts = nil
	moe.GateOutputs = nil
	moe.ExpertProbSums = nil
	moe.stateStack = nil
	moe.gateLogits = nil
	moe.TargetRouting = nil
	moe.LastSelectedExperts = nil

	// Clear state for all experts
	for _, expert := range moe.Experts {
		if expert != nil {
			expert.ClearState()
		}
	}

	if moe.GatingNetwork != nil {
		moe.GatingNetwork.ClearState()
	}
}

func (moe *MoELayer) GetResetCount() int {
	return int(atomic.LoadInt32(&moe.ResetCount))
}

func (moe *MoELayer) GetExpertResets() map[int]int {
	moe.resetsMu.RLock()
	defer moe.resetsMu.RUnlock()
	resets := make(map[int]int)
	for i, c := range moe.expertResets {
		resets[i] = c
	}
	return resets
}

func (moe *MoELayer) ClearResetCount() {
	atomic.StoreInt32(&moe.ResetCount, 0)
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
func CalculateRouterZLoss(routerLogits *Tensor) float32 {
	if routerLogits == nil || len(routerLogits.Data) == 0 {
		return 0
	}
	sumSq := float32(0.0)
	for _, v := range routerLogits.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			continue
		}
		sumSq += v * v
	}
	// Multiply by a small coefficient (e.g., 1e-4) as suggested by PaLM/ST-MoE
	return (sumSq / float32(len(routerLogits.Data))) * 0.0001
}

// CalculateSquareOfSumsLoss implements an aggressive diversity penalty to break monopolies.
// This calculates sum((count_i/total)^2) which rewards equal distribution.
func CalculateSquareOfSumsLoss(usageCounts []int, totalTokens int) float32 {
	// Weight heavily (e.g. 2.0) as requested to break stalemates
	return float32(SimdSquareOfSumsLossF32(usageCounts, totalTokens, 2.0))
}

// CalculateAuxLoss computes the load balancing loss (CV^2 of importance) to prevent expert starvation.
func CalculateAuxLoss(gateProbs []float32, numExperts int) float32 {
	if len(gateProbs) == 0 || numExperts == 0 {
		return 0
	}
	numTokens := len(gateProbs) / numExperts

	// 1. Calculate Importance (Sum of probabilities per expert)
	importance := make([]float32, numExperts)
	for i := 0; i < numTokens; i++ {
		for j := 0; j < numExperts; j++ {
			importance[j] += gateProbs[i*numExperts+j]
		}
	}

	// 2. Compute the coefficient of variation (CV) squared
	var sumImp float32
	for _, imp := range importance {
		sumImp += imp
	}

	meanImp := sumImp / float32(numExperts)
	if meanImp == 0 {
		return 0
	}

	var variance float32
	for _, imp := range importance {
		diff := imp - meanImp
		variance += diff * diff
	}
	variance /= float32(numExperts)

	// 🛡️ Denominator Stability Guard
	denom := meanImp * meanImp
	if denom < 1e-12 {
		return 0.0
	}

	res := variance / denom
	if math.IsNaN(float64(res)) || math.IsInf(float64(res), 0) {
		return 0.0
	}
	return res
}

// RebalanceExperts ensures all experts have a similar weight magnitude (L2 Norm).
// This prevents one expert from becoming a "gravity well" for the router.
func (moe *MoELayer) RebalanceExperts() {
	numExperts := len(moe.Experts)
	if numExperts == 0 {
		return
	}

	expertNorms := make([]float32, numExperts)
	var totalNorm float32

	// 1. Calculate individual norms and the average
	for i := 0; i < numExperts; i++ {
		params := moe.Experts[i].Parameters()
		var expertSumSq float32
		var count int
		for _, p := range params {
			for _, v := range p.Data {
				expertSumSq += v * v
				count++
			}
		}
		if count > 0 {
			expertNorms[i] = float32(math.Sqrt(float64(expertSumSq)))
		}
		totalNorm += expertNorms[i]
	}
	avgNorm := totalNorm / float32(numExperts)

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
		moe.GatingNetwork.Linear.Weights.Data[i] = (rand.Float32()*2 - 1) * 0.01
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

			// CLONE: Use the expert's own EvolutionaryReset method
			moe.Experts[i].EvolutionaryReset(moe.Experts[bestExpertIdx], 0.15)

			// RESET ROUTER: Clear the router's bias against this expert and set small random weights
			// Router weights are [inputDim, numExperts]. Expert i is the i-th column.
			inputDim := moe.GatingNetwork.Linear.Weights.Shape[0]
			for k := 0; k < inputDim; k++ {
				// Reset weight from all input dimensions to this expert
				moe.GatingNetwork.Linear.Weights.Data[k*numExperts+i] = (rand.Float32()*2 - 1) * 0.05
			}

			if moe.GatingNetwork.Linear.Biases != nil {
				moe.GatingNetwork.Linear.Biases.Data[i] = 0
			}

			moe.StagnationCounters[i] = 0
		}
	}
}

// ResizeExperts resizes the output dimension of all experts.
// To minimize peak memory usage, each expert's old weights are explicitly
// freed before allocating new ones, and the GC is hinted between iterations.
func (moe *MoELayer) ResizeExperts(newOutputDim int) {
	fmt.Printf("🔧 Resizing %d MoE Experts to new OutputDim: %d\n", len(moe.Experts), newOutputDim)

	for i, exp := range moe.Experts {
		exp.Resize(newOutputDim)
		// Hint GC to reclaim old memory before next expert
		runtime.GC()
		fmt.Printf("  ✓ Expert %d resized\n", i)
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
func (moe *MoELayer) UpdateRouterGrads(gateGrads [][]float32, scaledFractions []float32) {
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
			grads[e] += scaledFractions[e]
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

// CalculateDiversityLoss computes the cosine similarity between the outputs of the top-2 experts
// selected for each token and averages it over the batch.
func (moe *MoELayer) CalculateDiversityLoss() float32 {
	if moe.K < 2 {
		return 0
	}

	numTokens := len(moe.SelectedExperts)
	if numTokens == 0 {
		return 0
	}

	var totalSimilarity float32
	var count int

	// For each token, compare the top-2 experts
	for i := 0; i < numTokens; i++ {
		selected := moe.SelectedExperts[i]
		if len(selected) < 2 {
			continue
		}

		expertA := selected[0]
		expertB := selected[1]

		// Get the tokens' outputs from each expert
		outA := moe.expertOutputs[expertA]
		outB := moe.expertOutputs[expertB]

		if outA == nil || outB == nil {
			continue
		}

		// Find the relative index of this token for each expert
		// In Forward, we store the relative index in tokenExpertRelativeIndices.
		// Since we didn't store it in the struct, we can find it by searching.
		// Or better, we can modify Forward to store it if needed.
		// For now, let's look at ExpertTokenIndices.

		relIdxA := -1
		for idx, tIdx := range moe.ExpertTokenIndices[expertA] {
			if tIdx == i {
				relIdxA = idx
				break
			}
		}

		relIdxB := -1
		for idx, tIdx := range moe.ExpertTokenIndices[expertB] {
			if tIdx == i {
				relIdxB = idx
				break
			}
		}

		if relIdxA == -1 || relIdxB == -1 {
			continue
		}

		// Calculate cosine similarity between the two expert outputs for this token
		rowA := outA.Data[relIdxA*moe.OutputDim : (relIdxA+1)*moe.OutputDim]
		rowB := outB.Data[relIdxB*moe.OutputDim : (relIdxB+1)*moe.OutputDim]

		sim := CosineSimilarity(rowA, rowB)
		// We only care about positive correlation (experts being too similar)
		if sim > 0 {
			totalSimilarity += sim
		}
		count++
	}

	if count == 0 {
		return 0
	}
	return totalSimilarity / float32(count)
}

// CosineSimilarity calculates the cosine similarity between two vectors.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA <= 0 || normB <= 0 || math.IsNaN(float64(normA)) || math.IsNaN(float64(normB)) {
		return 0
	}

	denom := float32(math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
	if denom < 1e-12 {
		return 0
	}

	res := dot / denom
	if math.IsNaN(float64(res)) {
		return 0
	}
	return res
}

// UpdateExpertMultipliers adjusts the gradient multipliers based on expert utilization.
// Transitions an expert from "Recovery Mode" to "Standard Mode" once it proves it can handle a fair share of the load.
func (moe *MoELayer) UpdateExpertMultipliers() {
	const (
		TargetUsage           = 0.125 // 1/8 experts = 12.5% ideal
		MaxMult               = 2.5
		MinMult               = 1.0
		DecayRate             = 0.85
		LowUsageThreshold     = 0.02
		HealthyUsageThreshold = 0.10
	)

	numExperts := len(moe.Experts)
	totalTokens := 0
	for _, count := range moe.AccumulatedUtilization {
		totalTokens += count
	}

	if totalTokens == 0 {
		return
	}

	for i := 0; i < numExperts; i++ {
		utilization := float32(moe.AccumulatedUtilization[i]) / float32(totalTokens)

		// If utilization is healthy (>10%), start decaying the boost
		if utilization > HealthyUsageThreshold && moe.ExpertGradMultiplier[i] > MinMult {
			moe.ExpertGradMultiplier[i] *= float32(DecayRate)
			if moe.ExpertGradMultiplier[i] < MinMult {
				moe.ExpertGradMultiplier[i] = MinMult
			}
			fmt.Printf("🚀 Expert %d (L) stabilized: New Multiplier %.2f (Usage: %.2f%%)\n",
				i, moe.ExpertGradMultiplier[i], utilization*100)
		} else if utilization < LowUsageThreshold {
			// If it's still "Dead", keep the boost high
			moe.ExpertGradMultiplier[i] = float32(MaxMult)
		}
	}
}

// ShakeExperts performs an in-place noise injection to all stagnant experts.
func (moe *MoELayer) ShakeExperts(intensity float32, loopCount int) {
	// Scale noise by the number of consecutive loops detected
	// This ensures we shake harder if the model is truly stuck
	adjustedIntensity := intensity * float32(math.Log1p(float64(loopCount)))
	if adjustedIntensity < intensity {
		adjustedIntensity = intensity
	}

	for i, expert := range moe.Experts {
		// Use both expert-internal stagnation check and MoE-level usage counters
		stagnancyScore := 0
		if moe.StagnationCounters != nil {
			stagnancyScore = moe.StagnationCounters[i]
		}

		if expert.IsStagnant() || stagnancyScore > 5 {
			fmt.Printf("🌊 Expert %d is stagnant (Score: %d). Shaking at intensity %.4f to break circuit...\n",
				i, stagnancyScore, adjustedIntensity)
			expert.Shake(adjustedIntensity)

			// Reset stagnation after shake to see if it recovers
			if moe.StagnationCounters != nil {
				moe.StagnationCounters[i] = 0
			}
		}
	}
}

func (moe *MoELayer) RepairArchitecture() {
	if moe.GatingNetwork != nil {
		moe.GatingNetwork.RepairArchitecture()
	}
}
