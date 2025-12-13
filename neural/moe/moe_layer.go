package moe

import (
	"fmt"
	"runtime"
	"sort"
	"sync"

	. "github.com/golangast/gollemer/neural/tensor"
)

// MoELayer implements a Mixture of Experts layer.
type MoELayer struct {
	GatingNetwork *GatingNetwork
	Experts       []Expert
	K             int // Number of top experts to select
	// InputDim      int // Add InputDim to MoELayer struct

	// Stored for backward pass
	inputTensor        *Tensor
	expertOutputs      []*Tensor
	expertTokenIndices [][]int // Indices of tokens assigned to each expert
	selectedExperts    [][]int // Indices of selected experts for each input in the batch
	gateOutputs        *Tensor // Output of the gating network (probabilities)
	LoadBalancingLoss  float64 // Load balancing loss
	Training           bool    // training mode
}

// NewMoELayer creates a new MoELayer.
// inputDim is the dimension of the input to the MoE layer.
// numExperts is the total number of experts.
// k is the number of top experts to select for each input.
// expertBuilder is a function that constructs an expert given its index.
func NewMoELayer(inputDim, numExperts, k int, expertBuilder func(int) (Expert, error)) (*MoELayer, error) {
	if k <= 0 || k > numExperts {
		return nil, fmt.Errorf("k (%d) must be between 1 and numExperts (%d)", k, numExperts)
	}

	gatingNetwork, err := NewGatingNetwork(inputDim, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create gating network: %w", err)
	}

	experts := make([]Expert, numExperts)
	for i := 0; i < numExperts; i++ {
		expert, err := expertBuilder(i)
		if err != nil {
			return nil, fmt.Errorf("failed to create expert %d: %w", i, err)
		}
		experts[i] = expert
	}

	return &MoELayer{
		GatingNetwork: gatingNetwork,
		Experts:       experts,
		K:             k,
		// InputDim:      inputDim, // Initialize InputDim
	}, nil
}

// Parameters returns all learnable parameters of the MoELayer.
func (moe *MoELayer) Parameters() []*Tensor {
	params := moe.GatingNetwork.Parameters()
	for _, expert := range moe.Experts {
		params = append(params, expert.Parameters()...)
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

	// 1. Gating Network (Router) forward pass to get logits
	gateLogits, err := moe.GatingNetwork.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("moe layer gating network forward failed: %w", err)
	}

	// Apply softmax to get probabilities
	gateOutputs, err := gateLogits.Softmax(len(gateLogits.Shape) - 1)
	if err != nil {
		return nil, fmt.Errorf("gating network softmax failed: %w", err)
	}
	moe.gateOutputs = gateOutputs

	batchSize := input.Shape[0]
	seqLength := input.Shape[1]
	embeddingDim := input.Shape[2]
	numExperts := len(moe.Experts)

	moe.selectedExperts = make([][]int, batchSize*seqLength)
	moe.expertTokenIndices = make([][]int, numExperts)

	// Reshape input to 2D [batch*seq, dim] for gathering
	input2D, err := input.Reshape([]int{batchSize * seqLength, embeddingDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape input to 2D: %w", err)
	}

	// Store relative indices for scatter step
	tokenExpertRelativeIndices := make([][]int, batchSize*seqLength)

	for i := 0; i < batchSize*seqLength; i++ {
		scores := gateOutputs.Data[i*numExperts : (i+1)*numExperts]
		topKIndices := make([]int, numExperts)
		for j := range topKIndices {
			topKIndices[j] = j
		}
		sort.SliceStable(topKIndices, func(a, b int) bool {
			return scores[topKIndices[a]] > scores[topKIndices[b]]
		})
		selected := topKIndices[:moe.K]
		moe.selectedExperts[i] = selected

		tokenExpertRelativeIndices[i] = make([]int, len(selected))
		for j, expertIdx := range selected {
			tokenExpertRelativeIndices[i][j] = len(moe.expertTokenIndices[expertIdx])
			moe.expertTokenIndices[expertIdx] = append(moe.expertTokenIndices[expertIdx], i)
		}
	}

	moe.expertOutputs = make([]*Tensor, numExperts)
	var wg sync.WaitGroup
	var errMutex sync.Mutex
	var firstErr error

	// fmt.Println("Starting parallel expert execution (Forward)")
	// Run experts in parallel
	for i := 0; i < numExperts; i++ {
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
	finalOutput := NewTensor([]int{batchSize, seqLength, embeddingDim}, make([]float64, batchSize*seqLength*embeddingDim), true)

	// fmt.Println("Starting scattering")

	// Parallelize scattering by token
	var wgScatter sync.WaitGroup
	numWorkers := runtime.NumCPU()
	totalTokens := batchSize * seqLength
	tokensPerWorker := (totalTokens + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startToken := w * tokensPerWorker
		endToken := startToken + tokensPerWorker
		if endToken > totalTokens {
			endToken = totalTokens
		}
		if startToken >= endToken {
			break
		}

		wgScatter.Add(1)
		go func(start, end int) {
			defer wgScatter.Done()
			for i := start; i < end; i++ {
				selected := moe.selectedExperts[i]
				outStart := i * embeddingDim

				for j, expertIdx := range selected {
					output := moe.expertOutputs[expertIdx]
					if output == nil {
						continue
					}

					// Get weight
					weight := gateOutputs.Data[i*numExperts+expertIdx]

					// Get expert output row
					relativeRow := tokenExpertRelativeIndices[i][j]
					expertRowStart := relativeRow * embeddingDim
					expertRow := output.Data[expertRowStart : expertRowStart+embeddingDim]

					for k := 0; k < embeddingDim; k++ {
						finalOutput.Data[outStart+k] += expertRow[k] * weight
					}
				}
			}
		}(startToken, endToken)
	}
	wgScatter.Wait()
	// fmt.Println("Finished scattering")

	return finalOutput, nil
}

// Backward performs the backward pass for the MoELayer.
// Returns the gradient with respect to the input tensor.
func (moe *MoELayer) Backward(grad *Tensor) (*Tensor, error) {
	if grad == nil || grad.Data == nil {
		return nil, nil
	}

	// Remember if original grad was 2D (context vector) before reshaping/padding

	// Handle 2D gradient from decoder by processing only the last time step
	if len(grad.Shape) == 2 {
		// The grad is for the context vector, which corresponds to the last element of the sequence.
		// We will create a new grad tensor that has zeros everywhere except for the last time step.
		batchSize := grad.Shape[0]
		embeddingDim := grad.Shape[1]
		seqLength := moe.inputTensor.Shape[1]

		// Create fullGrad with correct size and shape
		fullGradSize := batchSize * seqLength * embeddingDim
		fullGradShape := []int{batchSize, seqLength, embeddingDim}
		fullGrad := NewTensor(fullGradShape, make([]float64, fullGradSize), false)

		// Copy gradient to last time step for each batch
		for i := 0; i < batchSize; i++ {
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
		return nil, fmt.Errorf("failed to reshape grad: %w", err)
	}

	// Initialize a tensor to accumulate gradients for the gating network
	gateGradReshaped := NewTensor([]int{batchSize * seqLength, numExperts}, make([]float64, batchSize*seqLength*numExperts), true)

	// fmt.Println("Starting parallel expert execution (Backward)")
	// Prepare gradients for each expert
	// We need to group gradients exactly as we grouped inputs in Forward
	// moe.expertTokenIndices has the mapping

	var wg sync.WaitGroup
	var errMutex sync.Mutex
	var firstErr error

	// Run experts backward in parallel
	for i := 0; i < numExperts; i++ {
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
				weight := moe.gateOutputs.Data[tokenIdx*numExperts+expertIdx]
				for j := 0; j < embeddingDim; j++ {
					weightedBatchedGradData[k*embeddingDim+j] = batchedGrad.Data[k*embeddingDim+j] * weight
				}
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

			// Accumulate input gradients
			// The expert's input was created via Gather.
			// So expert.Inputs()[0].Grad contains the gradients w.r.t the gathered input.
			// We need to scatter these back to the original input.
			// Fortunately, GatherOperation.Backward does exactly this!
			// We just need to trigger backward on the gathered input.

			if moe.inputTensor.RequiresGrad {
				expertInputs := moe.Experts[expertIdx].Inputs()
				if len(expertInputs) > 0 {
					gatheredInput := expertInputs[0]
					// Trigger backward on the gathered input to scatter gradients to input2D (and then to inputTensor)
					// Note: gatheredInput.Creator is the GatherOperation.
					if gatheredInput.Creator != nil {
						err := gatheredInput.Creator.Backward(gatheredInput.Grad)
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
				gradForTokenData := gradReshaped.Data[tokenIdx*embeddingDim : (tokenIdx+1)*embeddingDim]

				expertOutRow := expertOutput.Data[k*embeddingDim : (k+1)*embeddingDim]

				gradForGateProb := 0.0
				for j := 0; j < embeddingDim; j++ {
					gradForGateProb += gradForTokenData[j] * expertOutRow[j]
				}

				// This write is safe?
				// gateGradReshaped is [batch*seq, numExperts].
				// Each expert writes to a DIFFERENT column (expertIdx).
				// So this IS thread-safe.
				gateGradReshaped.Data[tokenIdx*numExperts+expertIdx] += gradForGateProb
			}

		}(i, indices)
	}
	wg.Wait()
	// fmt.Println("Finished parallel expert execution (Backward)")

	if firstErr != nil {
		return nil, firstErr
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
		var input2D *Tensor
		// Find input2D from one of the experts
		for _, expert := range moe.Experts {
			inputs := expert.Inputs()
			if len(inputs) > 0 {
				if inputs[0].Creator != nil { // Check if Creator exists
					gatherOp, ok := inputs[0].Creator.(*GatherOperation)
					if ok {
						input2D = gatherOp.Input
						break
					}
				}
			}
		}

		if input2D != nil && input2D.Grad != nil {
			// Propagate from input2D to inputTensor
			// input2D was created from inputTensor via Reshape.
			// We can manually call Reshape's backward logic or just copy/add data.
			// Reshape backward just adds gradients.
			if moe.inputTensor.Grad == nil {
				moe.inputTensor.Grad = NewTensor(moe.inputTensor.Shape, make([]float64, len(moe.inputTensor.Data)), false)
			}
			for i := range input2D.Grad.Data {
				moe.inputTensor.Grad.Data[i] += input2D.Grad.Data[i]
			}
		}
	}

	// Finally, backpropagate through the gating network with the accumulated gateGrad.
	// Workaround: GatingNetwork.Backward (Linear.Backward) seems to cause moe.inputTensor.Grad to become nil
	// in some cases, even though it shouldn't. We save the gradient pointer and restore it if needed.
	// Since Linear.Backward updates the gradient in place, savedGrad will point to the updated data.
	savedGrad := moe.inputTensor.Grad

	err = moe.GatingNetwork.Backward(gateGradReshaped)
	if err != nil {
		return nil, err
	}

	if moe.inputTensor.Grad == nil && savedGrad != nil {
		moe.inputTensor.Grad = savedGrad
	}

	// Return the gradient with respect to the input
	return moe.inputTensor.Grad, nil
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
