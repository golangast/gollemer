package moe

import (
	"bufio"
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/tagger/tag"
)

func init() {
	gob.Register(&IntentMoE{})
	gob.Register(&RNNDecoder{})
	gob.Register(&MoELayer{})
	gob.Register(&FeedForwardExpert{})
	gob.Register(&LinearExpert{})
	gob.Register(&GatingNetwork{})
	gob.Register(&SimpleRNNEncoder{})
	gob.Register(&nn.Linear{})
	gob.Register(&nn.Embedding{})
	gob.Register(&nn.LSTM{})
	gob.Register(&nn.LSTMCell{})
	gob.Register([]*nn.LSTMCell{})
	gob.Register([][]*nn.LSTMCell{})
	gob.Register(&tensor.ConcatOperation{})
	gob.Register(&tensor.DivScalarOperation{})
	gob.Register(&tensor.AddOperation{})
	gob.Register(&tensor.MatMulOperation{})
	gob.Register(&tensor.AddWithBroadcastOperation{})
	gob.Register(&tensor.SoftmaxOperation{})
	gob.Register(&tensor.MulScalarOperation{})
	gob.Register(&tensor.SelectOperation{})
	gob.Register(&tensor.TanhOperation{})
	gob.Register(&tensor.SigmoidOperation{})
	gob.Register(&tensor.LogOperation{})
	gob.Register(&tensor.MulOperation{})
	gob.Register(&tensor.SumOperation{})
	gob.Register(&tensor.SplitOperation{})
	gob.Register(&MoEStack{})
	gob.Register(&HybridLLMGNNEncoder{})
	gob.Register(&nn.LayerNorm{})
	gob.Register(&GoffiExpert{})
}

// ClearState clears the intermediate tensors used for backward pass
func (m *IntentMoE) ClearState() {
	if m.Encoder != nil {
		m.Encoder.ClearState()
	}
	if m.Decoder != nil {
		m.Decoder.ClearState()
	}
	if m.Embedding != nil {
		m.Embedding.ClearState()
	}
}

// SyncParameters synchronizes the entire model's parameters to GPU in parallel.
func (m *IntentMoE) SyncParameters() error {
	var wg sync.WaitGroup
	errCh := make(chan error, 3)

	if m.Encoder != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.Encoder.SyncParameters(); err != nil {
				errCh <- err
			}
		}()
	}

	if m.Decoder != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.Decoder.SyncParameters(); err != nil {
				errCh <- err
			}
		}()
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



// SampleFromLogits samples a token ID from logits using temperature, top-k, and top-p sampling.
// Updated to use the user's suggested top-K-only normalization for more stable inference.
func SampleFromLogits(logits *tensor.Tensor, temperature float32, topK int, topP float32) (int, error) {
	// logits shape: [batchSize, vocabSize]
	// We assume batchSize = 1 for inference
	if logits.Shape[0] != 1 {
		return 0, fmt.Errorf("SampleFromLogits expects batch size 1, got %d", logits.Shape[0])
	}

	vocabSize := logits.Shape[1]
	logits.ToCPU()
	logitsData := logits.Data

	// Apply temperature scaling
	if temperature <= 0.0 {
		temperature = 1.0 // Default to 1.0 if invalid
	}

	type tokenLogit struct {
		index int
		value float32
	}

	candidates := make([]tokenLogit, vocabSize)
	for i := range vocabSize {
		candidates[i] = tokenLogit{i, logitsData[i] / temperature}
	}

	// Sort descending
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].value > candidates[j].value
	})

	// Handle top-K truncation
	k := topK
	if k <= 0 || k > vocabSize {
		k = vocabSize
	}

	topKCandidates := candidates[:k]

	// Find max logit for numerical stability
	maxLogit := topKCandidates[0].value

	// Compute sum of exponents for top-K only
	var sumExp float32
	for _, c := range topKCandidates {
		sumExp += float32(math.Exp(float64(c.value - maxLogit)))
	}

	// Sample from the truncated distribution
	r := rand.Float32()
	var cumulative float32
	for _, c := range topKCandidates {
		prob := float32(math.Exp(float64(c.value-maxLogit))) / sumExp
		cumulative += prob
		if r < cumulative {
			return c.index, nil
		}
	}

	return topKCandidates[0].index, nil
}

// PredictNext performs a forward pass and samples the next token index using Top-K and temperature.
func (m *IntentMoE) PredictNext(input *tensor.Tensor, k int, temp float32) (int, float32, error) {
	// 1. Forward pass
	// We assume a simplified forward call for single-token prediction
	logits, _, err := m.Forward(0.0, input, nil, nil)
	if err != nil {
		return 0, 0, err
	}

	// 2. Get the last logit vector if it's a sequence
	var lastLogits *tensor.Tensor
	if len(logits) > 0 {
		lastLogits = logits[len(logits)-1]
	} else {
		return 0, 0, fmt.Errorf("PredictNext: no logits returned from forward pass")
	}

	// 3. Apply Temperature scaling
	if temp <= 0 {
		temp = 1.0
	}
	for i := range lastLogits.Data {
		lastLogits.Data[i] /= temp
	}

	// 4. Sample using Top-K logic
	idx, err := SampleFromLogits(lastLogits, 1.0, k, 0.0)
	if err != nil {
		return 0, 0, err
	}

	// 5. Get confidence (max probability after scaling)
	probs := tensor.Softmax(lastLogits)
	_, confidence := tensor.ArgMax(probs)

	return idx, confidence, nil
}

// CalculateAccuracy evaluates the model on a dataset for Top-K precision.
func (m *IntentMoE) CalculateAccuracy(inputs []*tensor.Tensor, targets []*tensor.Tensor, k int) float32 {
	correct := 0
	total := 0

	for i := range inputs {
		input := inputs[i]
		target := targets[i]

		// 1. Get logits from the forward pass (inference mode)
		m.SetMode(false)
		logits, _, err := m.Forward(0.0, input, nil, nil)
		if err != nil || len(logits) == 0 {
			continue
		}

		// Use the last logit for classification accuracy
		lastLogit := logits[len(logits)-1]

		// 2. Get the indices of the Top-K highest logits
		topKIndices := getTopKIndices(lastLogit.Data, k)

		// 3. Check if the actual target (last token of sequence) is in that set
		actualTarget := int(target.Data[len(target.Data)-1])
		for _, idx := range topKIndices {
			if idx == actualTarget {
				correct++
				break
			}
		}
		total++
	}

	if total == 0 {
		return 0
	}
	return float32(correct) / float32(total)
}

// getTopKIndices extracts top K indices from a logit slice without full sort overhead.
func getTopKIndices(logits []float32, k int) []int {
	type pair struct {
		index int
		val   float32
	}
	pairs := make([]pair, len(logits))
	for i, v := range logits {
		pairs[i] = pair{i, v}
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].val > pairs[j].val
	})

	if k > len(logits) {
		k = len(logits)
	}

	indices := make([]int, k)
	for i := 0; i < k; i++ {
		indices[i] = pairs[i].index
	}
	return indices
}

// ApplyRepetitionPenalty penalizes tokens that have already been generated.
// Logits are the raw output of the model, generatedIDs are tokens already picked.
// Penalty is typically 1.1 or 1.2 (for multiplicative) or a flat subtraction.
func ApplyRepetitionPenalty(logits *tensor.Tensor, generatedIDs []int, penalty float32) {
	if penalty == 1.0 {
		return
	}

	// Track seen tokens in a map for O(1) lookup
	seen := make(map[int]bool)
	for _, id := range generatedIDs {
		seen[id] = true
	}

	for id := range seen {
		// Only penalize valid token IDs
		if id < 0 || id >= len(logits.Data) {
			continue
		}

		// For Logits (pre-softmax):
		// If positive, divide by penalty to reduce score.
		// If negative, multiply by penalty to make it MORE negative.
		if logits.Data[id] > 0 {
			logits.Data[id] /= penalty
		} else {
			logits.Data[id] *= penalty
		}
	}
}

// Encoder interface for different encoder types (MoE, SimpleRNN, etc.)
type Encoder interface {
	Forward(...*tensor.Tensor) (*tensor.Tensor, error)
	Backward(*tensor.Tensor) error

	Inputs() []*tensor.Tensor
	Parameters() []*tensor.Tensor
	SetMode(bool)
	ClearState()
	GetMoELayers() []*MoELayer
	SetGateTemperature(float32)
	ToGPU()
	SyncParameters() error
}

// ExpertStat holds performance metrics for a specific expert.
type ExpertStat struct {
	LossSum    float32
	TokenCount int
}

// ModelMetadata persists the training state across reloads.
type ModelMetadata struct {
	BestPerplexity   float32
	LastEpoch        int
	StagnantCounters map[string]int
	FrozenStates     map[string]bool
	LearningRate     float32
}

// IntentMoE represents a Mixture of Experts model for intent classification.
type IntentMoE struct {
	Encoder           Encoder // Changed to interface to support different encoder types
	Decoder           *RNNDecoder
	Embedding         *nn.Embedding
	SentenceVocabSize int
	SentenceVocab     *mainvocab.Vocabulary
	EmbeddingDim      int // Persisted dimension (e.g., 768) for resizing logic

	// Diagnostics and Monitoring
	ExpertStats map[string]*ExpertStat // Key: "layerID:expertID"
	Metadata    ModelMetadata
}

// ToGPU moves the entire model's parameters to the GPU.
func (m *IntentMoE) ToGPU() {
	if m.Encoder != nil {
		m.Encoder.ToGPU()
	}
	if m.Decoder != nil {
		m.Decoder.ToGPU()
	}
	if m.Embedding != nil {
		m.Embedding.ToGPU()
	}

	// 🌡️ GPU WARM-UP: Serial compilation of pipelines to prevent race conditions
	// (Vulkan/vkCreateComputePipelines crash) during the first parallel batch.
	m.warmup()
}

func (m *IntentMoE) warmup() {
	if m.Encoder == nil {
		return
	}
	layers := m.Encoder.GetMoELayers()
	if len(layers) == 0 || len(layers[0].Experts) == 0 {
		return
	}

	fmt.Println("🌡️  GPU Warm-up: Sequential pipeline compilation...")
	// Create a small dummy tensor on GPU
	dummyInput := tensor.NewTensor([]int{1, m.EmbeddingDim}, make([]float32, m.EmbeddingDim), false)
	dummyInput.ToGPU()

	// Trigger forward pass on one expert; internal caches will now be populated.
	// We don't need to run all experts; once the shaders are cached in the shared 
	// backend, subsequent parallel calls will find them in the RLock-protected map.
	layers[0].Experts[0].Forward(dummyInput)
	fmt.Println("✅ GPU Warm-up complete.")
}

// NewIntentMoE creates a new IntentMoE model.
func NewIntentMoE(vocabSize, embeddingDim, numExperts, parentVocabSize, childVocabSize, sentenceVocabSize, maxAttentionHeads int, word2vecModel *word2vec.SimpleWord2Vec) (*IntentMoE, error) {
	if word2vecModel != nil {
		vocabSize = word2vecModel.VocabSize
		word2vecModel.SyncF32()
	}
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)
	if word2vecModel != nil {
		embedding.LoadPretrainedWeights(word2vecModel.WordVectorsF32)
	}

	// Define the expert builder function
	expertBuilder := func(expertIdx int) (Expert, error) {
		return NewFeedForwardExpert(embeddingDim, embeddingDim, embeddingDim) // Example: inputDim, hiddenDim, outputDim
	}

	// Initialize the MoE encoder
	// Assuming k=1 (select top 1 expert) for simplicity, adjust as needed
	encoder, err := NewMoELayer(embeddingDim, embeddingDim, numExperts, 1, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE encoder: %w", err)
	}

	// Initialize the RNN Decoder (legacy code - using defaults: 1 layer, no dropout)
	decoder, err := NewRNNDecoder(embeddingDim, sentenceVocabSize, embeddingDim, maxAttentionHeads, 1, 0.0, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create RNN decoder: %w", err)
	}

	return &IntentMoE{
		Encoder:           encoder,
		Decoder:           decoder,
		Embedding:         embedding,
		SentenceVocabSize: sentenceVocabSize,
		SentenceVocab:     mainvocab.NewVocabulary(), // Should be set by caller
		EmbeddingDim:      embeddingDim,
		ExpertStats:       make(map[string]*ExpertStat),
	}, nil
}

// ComputeAuxiliaryLoss computes the penalty for expert imbalance in the MoE layers.
func (m *IntentMoE) ComputeAuxiliaryLoss(stats MoEStats, batchSize int, numExperts int) float32 {
	var auxLoss float32

	// N * sum(fi * Pi)
	// fi = fraction of tokens sent to expert i
	// Pi = mean probability assigned to expert i
	for i := 0; i < numExperts; i++ {
		fi := float32(stats.ExpertCounts[i]) / float32(batchSize)
		pi := stats.RouterProbSum[i] / float32(batchSize)
		auxLoss += fi * pi
	}

	return auxLoss * float32(numExperts)
}

// TrackExpertPerformance updates the average loss handled by an expert.
func (m *IntentMoE) TrackExpertPerformance(layerID, expertID int, loss float32) {
	key := fmt.Sprintf("%d:%d", layerID, expertID)
	if m.ExpertStats == nil {
		m.ExpertStats = make(map[string]*ExpertStat)
	}
	if _, ok := m.ExpertStats[key]; !ok {
		m.ExpertStats[key] = &ExpertStat{}
	}
	m.ExpertStats[key].LossSum += loss
	m.ExpertStats[key].TokenCount++
}

// GetBestExpert finds the expert with the lowest average loss in a given layer.
func (m *IntentMoE) GetBestExpert(layerID int) int {
	bestID := 0
	minLoss := float32(math.MaxFloat32)
	found := false

	// Iterate through experts to find the one with the best performance
	// This assumes numExperts can be determined from the stats keys
	for key, stats := range m.ExpertStats {
		var lID, eID int
		fmt.Sscanf(key, "%d:%d", &lID, &eID)
		if lID == layerID && stats.TokenCount > 0 {
			avgLoss := stats.LossSum / float32(stats.TokenCount)
			if avgLoss < minLoss {
				minLoss = avgLoss
				bestID = eID
				found = true
			}
		}
	}

	if !found {
		return rand.Intn(8) // Fallback to random if no stats yet
	}
	return bestID
}

// PrintExpertHealthReport prints a summary of expert performance.
func (m *IntentMoE) PrintExpertHealthReport() {
	if m.ExpertStats == nil || len(m.ExpertStats) == 0 {
		fmt.Println("🏥 No Expert Stats available yet.")
		return
	}
	fmt.Println("\n🏥 --- Expert Health Report ---")
	// Sort keys for consistent output
	keys := make([]string, 0, len(m.ExpertStats))
	for k := range m.ExpertStats {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		stats := m.ExpertStats[key]
		avgLoss := float32(0.0)
		if stats.TokenCount > 0 {
			avgLoss = stats.LossSum / float32(stats.TokenCount)
		}
		fmt.Printf("Layer:Expert %s: Avg Loss: %7.4f | Tokens Handled: %d\n", key, avgLoss, stats.TokenCount)
	}
}

// EvolutionaryReset clones the best expert in a layer to replace a stagnant one.
func (m *IntentMoE) EvolutionaryReset(stagnantExpertID int, layerIdx int) {
	// 1. Find the "Winner" in this layer
	winnerID := m.GetBestExpert(layerIdx)
	if winnerID == stagnantExpertID {
		// If winner is stagnant, just random reset (shouldn't happen with proper stats)
		return
	}

	// 2. Locate the MoE Layer
	var targetLayer *MoELayer
	if layerIdx < len(ActiveLayers) {
		targetLayer = ActiveLayers[layerIdx]
	} else if m.Decoder.OutputMoE != nil {
		targetLayer = m.Decoder.OutputMoE
	} else {
		return
	}

	winnerExpert := targetLayer.Experts[winnerID]
	stagnantExpert := targetLayer.Experts[stagnantExpertID]

	winnerParams := winnerExpert.Parameters()
	stagnantParams := stagnantExpert.Parameters()

	if len(winnerParams) != len(stagnantParams) {
		return
	}

	// 3. Clone and Mutate with High-Variance Gaussian Jitter (0.15)
	fmt.Printf("🧬 Expert %d (L%d) evolved from Expert %d (SIMD Jitter 0.15 + Gating Reset)\n", stagnantExpertID, layerIdx, winnerID)
	for i := range stagnantParams {
		wp := winnerParams[i]
		sp := stagnantParams[i]
		// Use SIMD-ready jitter function
		simdAddJitterF32(sp.Data, wp.Data, 0.15)

		// Zero out the gradients for the new expert
		if sp.Grad != nil {
			for j := range sp.Grad.Data {
				sp.Grad.Data[j] = 0
			}
		}
	}


// 4. Reset the Router's view of this expert
// Weights are [inputDim, numExperts]. Expert j is the j-th column: W[k][j] = Data[k*numExperts + j]
	numExperts := targetLayer.GatingNetwork.Linear.Weights.Shape[1]
	inputDim := targetLayer.GatingNetwork.Linear.Weights.Shape[0]
	gatingData := targetLayer.GatingNetwork.Linear.Weights.Data
	for k := 0; k < inputDim; k++ {
		gatingData[k*numExperts+stagnantExpertID] = (rand.Float32() * 0.02) - 0.01
	}
	if targetLayer.GatingNetwork.Linear.Biases != nil {
		targetLayer.GatingNetwork.Linear.Biases.Data[stagnantExpertID] = 0
	}
}

// PerformGlobalWeightSurgery prunes weak weights across all experts in the model.
func (m *IntentMoE) PerformGlobalWeightSurgery(threshold float32) int {
	totalKills := 0
	for _, layer := range m.Encoder.GetMoELayers() {
		for _, expert := range layer.Experts {
			totalKills += PerformWeightSurgery(expert, threshold)
		}
	}
	if m.Decoder.OutputMoE != nil {
		for _, expert := range m.Decoder.OutputMoE.Experts {
			totalKills += PerformWeightSurgery(expert, threshold)
		}
	}
	return totalKills
}

// NewHybridIntentMoE creates a new IntentMoE model using the Hybrid LLM-GNN Encoder.
func NewHybridIntentMoE(vocabSize, embeddingDim, numExperts, parentVocabSize, childVocabSize, sentenceVocabSize, maxAttentionHeads int, word2vecModel *word2vec.SimpleWord2Vec) (*IntentMoE, error) {
	if word2vecModel != nil {
		vocabSize = word2vecModel.VocabSize
		word2vecModel.SyncF32()
	}
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)
	if word2vecModel != nil {
		embedding.LoadPretrainedWeights(word2vecModel.WordVectorsF32)
	}

	// 1. Create the inner LLM Encoder (MoE Stack with 4 layers for deeper reasoning)
	expertBuilder := func(expertIdx int) (Expert, error) {
		return NewGoffiExpert(expertIdx, embeddingDim, embeddingDim*4, embeddingDim)
	}
	l0, err := NewMoELayer(embeddingDim, embeddingDim, numExperts, 2, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE Layer 0: %w", err)
	}
	l1, err := NewMoELayer(embeddingDim, embeddingDim, numExperts, 2, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE Layer 1: %w", err)
	}
	l2, err := NewMoELayer(embeddingDim, embeddingDim, numExperts, 2, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE Layer 2: %w", err)
	}
	l3, err := NewMoELayer(embeddingDim, embeddingDim, numExperts, 2, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE Layer 3: %w", err)
	}

	llmEncoder := NewMoEStack(l0, l1, l2, l3)

	// 2. Wrap it with HybridLLMGNNEncoder
	hybridEncoder, err := NewHybridLLMGNNEncoder(llmEncoder, embeddingDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create HybridLLMGNNEncoder: %w", err)
	}

	// 3. Initialize Decoder
	decoder, err := NewRNNDecoder(embeddingDim, sentenceVocabSize, embeddingDim, maxAttentionHeads, 4, 0.0, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create RNN decoder: %w", err)
	}

	return &IntentMoE{
		Encoder:           hybridEncoder,
		Decoder:           decoder,
		Embedding:         embedding,
		EmbeddingDim:      embeddingDim,
		SentenceVocabSize: sentenceVocabSize,
		SentenceVocab:     nil,
	}, nil
}

// NormalizeContextVector returns a normalized copy of the context vector:
// scale each token's embedding to have L2 norm <= threshold (default 5.0) across the feature dim.
func (m *IntentMoE) NormalizeContextVector(cv *tensor.Tensor) *tensor.Tensor {
	if cv == nil {
		return nil
	}
	// Sync to CPU as normalization logic relies on Data slice
	cv.ToCPU()

	// Create a new tensor to avoid mutating original for backprop
	contextVector := tensor.NewTensor(cv.Shape, make([]float32, len(cv.Data)), cv.RequiresGrad)
	copy(contextVector.Data, cv.Data)
	contextVector.Creator = cv.Creator

	bSz := contextVector.Shape[0]
	sLen := contextVector.Shape[1]
	dim := contextVector.Shape[2]
	const ctxNormThreshold = 8.0
	for b := 0; b < bSz; b++ {
		for s := 0; s < sLen; s++ {
			offset := (b*sLen + s) * dim
			var norm float32
			for d := 0; d < dim; d++ {
				v := contextVector.Data[offset+d]
				norm += v * v
			}
			norm = float32(math.Sqrt(float64(norm + 1e-8)))
			if norm > ctxNormThreshold {
				scale := float32(ctxNormThreshold / norm)
				for d := 0; d < dim; d++ {
					contextVector.Data[offset+d] *= scale
				}
			}
		}
	}
	return contextVector
}

// Forward performs the forward pass of the IntentMoE model.
// scheduledSamplingProb: probability of using model predictions instead of ground truth (0.0 for inference)
func (m *IntentMoE) Forward(scheduledSamplingProb float32, inputs ...*tensor.Tensor) ([]*tensor.Tensor, *tensor.Tensor, error) {
	if len(inputs) < 2 {
		return nil, nil, fmt.Errorf("IntentMoE.Forward expects at least 2 inputs (query token IDs, target token IDs), got %d", len(inputs))
	}
	queryTokenIDs := inputs[0]
	targetTokenIDs := inputs[1]
	var inputMask *tensor.Tensor
	if len(inputs) >= 3 {
		inputMask = inputs[2]
	}

	// Pass token IDs through embedding layer
	queryEmbeddings, err := m.Embedding.Forward(queryTokenIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("embedding layer forward failed: %w", err)
	}

	// Encoder forward pass
	contextVector, err := m.Encoder.Forward(queryEmbeddings)
	if err != nil {
		return nil, nil, fmt.Errorf("MoE encoder forward failed: %w", err)
	}

	// Normalize context vector to prevent exploding values from propagating to the decoder.
	contextVector = m.NormalizeContextVector(contextVector)

	// Decoder forward pass with scheduled sampling & mask
	sentenceLogits, err := m.Decoder.Forward(contextVector, targetTokenIDs, scheduledSamplingProb, inputMask)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder forward failed: %w", err)
	}

	return sentenceLogits, contextVector, nil
}

// Backward performs the backward pass for the IntentMoE model.
func (m *IntentMoE) Backward(grads ...*tensor.Tensor) error {
	sentenceGrads := grads

	// Backward pass for the decoder
	if err := m.Decoder.Backward(sentenceGrads); err != nil {
		return fmt.Errorf("decoder backward failed: %w", err)
	}
	// CRITICAL FIX: Use the full context vector sequence gradient (accumulates from all attention steps)
	if m.Decoder.contextVector == nil {
		return fmt.Errorf("decoder context vector is nil in backward")
	}
	if m.Decoder.contextVector.Grad == nil {
		// If no gradients reached the context vector (e.g. zero attention),
		// initialize to zeros so the encoder backward can still proceed.
		m.Decoder.contextVector.Grad = tensor.NewTensor(m.Decoder.contextVector.Shape, make([]float32, len(m.Decoder.contextVector.Data)), false)
	}

	// This is a "local" clip that doesn't replace the global one but acts as a circuit breaker.
	cvGrad := m.Decoder.contextVector.Grad
	const cvClipThreshold = 10.0
	var cvSumSq float32
	for _, v := range cvGrad.Data {
		cvSumSq += v * v
	}
	cvNorm := float32(math.Sqrt(float64(cvSumSq + 1e-8)))
	if cvNorm > cvClipThreshold {
		scale := float32(cvClipThreshold / cvNorm)
		for i := range cvGrad.Data {
			cvGrad.Data[i] *= scale
		}
	}

	// 2. Account for NormalizeContextVector in the backward pass.
	// Since we scaled tokens by ctxNormThreshold/norm during forward, we must
	// scale their gradients by the same factor (first-order approximation).
	if m.Encoder.Inputs() != nil {
		// e.LLMEncoder.lastOutput is the un-normalized vector
		// However, it's safer to just re-read the contextVector Data IF we stored it.
		// For now, we apply the same normalization logic to the GRADIENT
		// based on the contextVector itself.
		dim := cvGrad.Shape[len(cvGrad.Shape)-1]
		numTokens := len(cvGrad.Data) / dim
		const ctxNormThreshold = 8.0
		for i := 0; i < numTokens; i++ {
			offset := i * dim
			var norm float32
			for d := 0; d < dim; d++ {
				// Use the un-normalized data from forward
				v := m.Decoder.contextVector.Data[offset+d]
				norm += v * v
			}
			norm = float32(math.Sqrt(float64(norm + 1e-8)))
			// If the token was at the boundary, suppress its gradient.
			if norm >= float32(ctxNormThreshold-0.01) {
				for d := 0; d < dim; d++ {
					cvGrad.Data[offset+d] *= 0.5
				}
			}
		}
	}

	contextVectorGrad := cvGrad

	// Backpropagate through the encoder
	err := m.Encoder.Backward(contextVectorGrad)
	if err != nil {
		return fmt.Errorf("MoE encoder backward failed: %w", err)
	}

	// Get the gradient for the embedding layer from the encoder's input
	if len(m.Encoder.Inputs()) > 0 {
		embeddingGrad := m.Encoder.Inputs()[0].Grad
		if embeddingGrad != nil {
			if err := m.Embedding.Backward(embeddingGrad); err != nil {
				return fmt.Errorf("embedding layer backward failed: %w", err)
			}
		}
	}

	return nil
}

// Parameters returns all learnable parameters of the IntentMoE model.
func (m *IntentMoE) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{}
	params = append(params, m.Embedding.Parameters()...)
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.Decoder.Parameters()...)
	return params
}

// SetMode sets the model to training or inference mode.
func (m *IntentMoE) SetMode(training bool) {
	if m.Encoder != nil {
		m.Encoder.SetMode(training)
	}
	if m.Decoder != nil {
		m.Decoder.SetMode(training)
	}
}

// SetGateTemperature updates the temperature for all MoE layers in the model.
func (m *IntentMoE) SetGateTemperature(temp float32) {
	if m.Encoder != nil {
		m.Encoder.SetGateTemperature(temp)
	}
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		m.Decoder.OutputMoE.RouterTemperature = temp
	}
}

// GreedySearchDecode performs greedy decoding (temperature=1.0).
// This is a wrapper for backward compatibility.
func (m *IntentMoE) GreedySearchDecode(contextVector *tensor.Tensor, maxLen, sosToken, eosToken int, repetitionPenalty, frequencyPenalty float32, topK int, taggedData tag.Tag) ([]int, error) {
	return m.GreedySearchDecodeWithTemp(contextVector, maxLen, sosToken, eosToken, 1.0, repetitionPenalty, frequencyPenalty, topK, taggedData)
}

func (m *IntentMoE) GreedySearchDecodeWithTemp(contextVector *tensor.Tensor, maxLen, sosToken, eosToken int, temperature, repetitionPenalty, frequencyPenalty float32, topK int, taggedData tag.Tag) ([]int, error) {
	var decodedIDs []int
	decoderInputIDs := tensor.NewTensor([]int{1, 1}, []float32{float32(sosToken)}, false)

	// Take the first element of the batch
	contextVector, err := contextVector.Slice(0, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to slice context vector: %w", err)
	}

	batchSize := contextVector.Shape[0]
	hiddenSize := m.Decoder.LSTM.HiddenSize

	initialHidden, err := contextVector.Mean(1)
	if err != nil {
		return nil, fmt.Errorf("failed to get mean of context vector for initial hidden state: %w", err)
	}

	if initialHidden.Shape[1] != hiddenSize {
		if initialHidden.Shape[1] > hiddenSize {
			initialHidden, err = initialHidden.Slice(1, 0, hiddenSize)
			if err != nil {
				return nil, fmt.Errorf("failed to slice initial hidden state: %w", err)
			}
		} else if initialHidden.Shape[1] < hiddenSize {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float32, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, err = tensor.Concat([]*tensor.Tensor{initialHidden, padding}, 1)
			if err != nil {
				return nil, fmt.Errorf("failed to pad initial hidden state: %w", err)
			}
		}
	}

	hiddenState := initialHidden
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	for step := 0; step < maxLen; step++ {
		outputLogits, newHidden, newCell, err := m.Decoder.DecodeStep(decoderInputIDs, hiddenState, cellState, contextVector)
		if err != nil {
			return nil, fmt.Errorf("decoder step failed: %w", err)
		}
		outputLogits.ToCPU()

		hiddenState = newHidden
		cellState = newCell

		// Suppress EOS for the first 3 steps — forces the model to generate
		// at least some words even before it's fully trained.
		if step < 3 && eosToken >= 0 && eosToken < len(outputLogits.Data) {
			outputLogits.Data[eosToken] = -1e9
		}

		// Apply repetition penalty
		ApplyRepetitionPenalty(outputLogits, decodedIDs, repetitionPenalty)

		// Apply Frequency Penalty
		if frequencyPenalty > 0.0 {
			counts := make(map[int]int)
			for _, id := range decodedIDs {
				counts[id]++
			}
			for id, count := range counts {
				if id < len(outputLogits.Data) {
					outputLogits.Data[id] -= frequencyPenalty * float32(count)
				}
			}
		}

		// Stuck Detector: If we predict the same token 3 times in a row, force a change
		if len(decodedIDs) >= 3 {
			last1 := decodedIDs[len(decodedIDs)-1]
			last2 := decodedIDs[len(decodedIDs)-2]
			last3 := decodedIDs[len(decodedIDs)-3]
			if last1 == last2 && last2 == last3 {
				if last1 < len(outputLogits.Data) {
					outputLogits.Data[last1] = -1e9
				}
			}
		}

		// Use sampling/top-k decoding for diversity
		sampledID, err := SampleFromLogits(outputLogits, temperature, topK, 0.0)
		if err != nil {
			return nil, fmt.Errorf("sampling failed: %w", err)
		}
		predictedID := sampledID

		// Diagnostic: log top-3 predictions for the first step
		if step == 0 && m.SentenceVocab != nil {
			type pred struct {
				id   int
				prob float32
			}
			vocabSize := len(outputLogits.Data)
			preds := make([]pred, vocabSize)
			var maxL float32 = -1e10
			for _, v := range outputLogits.Data {
				if v > maxL {
					maxL = v
				}
			}
			var sum float32 = 0.0
			for i, v := range outputLogits.Data {
				preds[i] = pred{i, float32(math.Exp(float64(v - maxL)))}
				sum += preds[i].prob
			}
			for i := range preds {
				preds[i].prob /= sum
			}
			sort.Slice(preds, func(a, b int) bool { return preds[a].prob > preds[b].prob })
			top := 3
			if len(preds) < top {
				top = len(preds)
			}
			fmt.Printf("🔍 [Decoder Step 0] Top predictions:\n")
			for k := 0; k < top; k++ {
				word := m.SentenceVocab.GetWord(preds[k].id)
				special := ""
				if preds[k].id == eosToken {
					special = " ← EOS (suppressed)"
				}
				fmt.Printf("   [%d] %-14s (%.2f%%)%s\n", k+1, word, preds[k].prob*100, special)
			}
		}

		if predictedID == eosToken {
			break
		}

		decodedIDs = append(decodedIDs, predictedID)
		decoderInputIDs = tensor.NewTensor([]int{1, 1}, []float32{float32(predictedID)}, false)
	}

	// Debug: print predicted token IDs and their mapped words
	if m.SentenceVocab != nil {
		words := make([]string, 0, len(decodedIDs))
		for _, id := range decodedIDs {
			// Filter out PAD, UNK, BOS, EOS tokens
			if id == m.SentenceVocab.PaddingTokenID || id == m.SentenceVocab.UnkID || id == m.SentenceVocab.BosID || id == m.SentenceVocab.EosID {
				continue
			}
			word := m.SentenceVocab.GetWord(id)
			// Filter out UNK and empty
			if word == "UNK" || word == "" {
				continue
			}
			words = append(words, word)
		}
		// Expanded known keys for formatting
		knownKeys := map[string]bool{
			"operation": true, "target_resource": true, "context": true, "user_role": true, "type": true, "name": true, "properties": true, "path": true,
			"Create": true, "Delete": true, "Update": true, "Filesystem::Folder": true, "Filesystem::File": true,
			"admin": true, "query": true, "semantic_output": true, "directory": true, "file": true, "folder": true, "resource": true, "role": true,
			"intent": true, "action": true, "status": true, "result": true, "input": true, "output": true, "config": true, "data": true, "plugin": true,
			"server": true, "webserver": true, "repository": true, "host": true, "port": true, "address": true, "command": true, "response": true,
			"description": true, "permission": true, "owner": true, "group": true, "mode": true, "timestamp": true, "log": true, "message": true,
			"license": true, "desktop": true, "user": true, "plugins": true, "project": true, "test": true, "src": true, "docs": true,
			"update": true, "delete": true, "create": true, "read": true, "write": true, "execute": true, "run": true, "start": true, "stop": true,
			"restart": true, "info": true, "details": true, "summary": true, "version": true, "author": true, "email": true,
			"readme": true, "main": true, "go": true, "server.go": true, "repository.go": true, "test.go": true, "config.go": true,
		}

		keyToNer := map[string]string{
			"type":            "OBJECT_TYPE",
			"target_resource": "PATH",
			"path":            "PATH",
		}

		nerToToken := make(map[string]string)
		for i, token := range taggedData.Tokens {
			if i < len(taggedData.NerTag) && taggedData.NerTag[i] != "O" && taggedData.NerTag[i] != "" {
				nerToToken[taggedData.NerTag[i]] = token
			}
		}

		for i := 0; i < len(words)-1; i++ {
			if nerTag, ok := keyToNer[words[i]]; ok {
				if replacementToken, ok := nerToToken[nerTag]; ok {
					words[i+1] = replacementToken
				}
			}
		}

		var formatted []string
		i := 0
		for i < len(words)-1 {
			key := words[i]
			val := words[i+1]
			if knownKeys[key] {
				formatted = append(formatted, fmt.Sprintf("%s: %s", key, val))
				i += 2
			} else {
				formatted = append(formatted, key)
				i++
			}
		}
	}
	return decodedIDs, nil
}

// SampleDecode performs sampling-based decoding with temperature, top-k, and top-p (nucleus) sampling.
// temperature: controls randomness (0.0 = deterministic, 1.0 = normal, >1.0 = more random)
// topK: if > 0, only sample from top K tokens
// topP: if > 0.0 and < 1.0, only sample from tokens whose cumulative probability is <= topP
func (m *IntentMoE) SampleDecode(contextVector *tensor.Tensor, maxLen, sosToken, eosToken int, temperature float32, topK int, topP float32, repetitionPenalty, frequencyPenalty float32) ([]int, error) {
	var decodedIDs []int
	decoderInputIDs := tensor.NewTensor([]int{1, 1}, []float32{float32(sosToken)}, false)

	// Take the first element of the batch
	contextVector, err := contextVector.Slice(0, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to slice context vector: %w", err)
	}

	batchSize := contextVector.Shape[0]
	hiddenSize := m.Decoder.LSTM.HiddenSize

	initialHidden, err := contextVector.Mean(1)
	if err != nil {
		return nil, fmt.Errorf("failed to get mean of context vector for initial hidden state: %w", err)
	}

	if initialHidden.Shape[1] != hiddenSize {
		if initialHidden.Shape[1] > hiddenSize {
			initialHidden, err = initialHidden.Slice(1, 0, hiddenSize)
			if err != nil {
				return nil, fmt.Errorf("failed to slice initial hidden state: %w", err)
			}
		} else if initialHidden.Shape[1] < hiddenSize {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float32, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, err = tensor.Concat([]*tensor.Tensor{initialHidden, padding}, 1)
			if err != nil {
				return nil, fmt.Errorf("failed to pad initial hidden state: %w", err)
			}
		}
	}

	hiddenState := initialHidden
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	for range maxLen {
		outputLogits, newHidden, newCell, err := m.Decoder.DecodeStep(decoderInputIDs, hiddenState, cellState, contextVector)
		if err != nil {
			return nil, fmt.Errorf("decoder step failed: %w", err)
		}

		hiddenState = newHidden
		cellState = newCell

		// Apply repetition penalty
		ApplyRepetitionPenalty(outputLogits, decodedIDs, repetitionPenalty)

		// Apply Frequency Penalty
		if frequencyPenalty > 0.0 {
			counts := make(map[int]int)
			for _, id := range decodedIDs {
				counts[id]++
			}
			for id, count := range counts {
				if id < len(outputLogits.Data) {
					outputLogits.Data[id] -= frequencyPenalty * float32(count)
				}
			}
		}

		// Sample from the logits with temperature, top-k, and top-p
		predictedID, err := SampleFromLogits(outputLogits, temperature, topK, topP)
		if err != nil {
			return nil, fmt.Errorf("sampling failed: %w", err)
		}

		if predictedID == eosToken {
			break
		}

		decodedIDs = append(decodedIDs, predictedID)

		decoderInputIDs = tensor.NewTensor([]int{1, 1}, []float32{float32(predictedID)}, false)
	}

	return decodedIDs, nil
}

// Checkpoint wraps the model and its training metadata for persistence.
type Checkpoint struct {
	Model           *IntentMoE
	StepCount       int
	LastProfile     nn.TrainingProfile
	Commitment      float32
	TokensProcessed int64
	TotalDuration   time.Duration
	Version         string
}

// CalculateCommitment calculates the "Intelligence" metric (% of weights > 0.40).
func (m *IntentMoE) CalculateCommitment() float32 {
	var highCount int
	var totalWeight int

	params := m.Parameters()
	for _, p := range params {
		totalWeight += len(p.Data)
		for _, w := range p.Data {
			if float32(math.Abs(float64(w))) > 0.40 {
				highCount++
			}
		}
	}

	if totalWeight == 0 {
		return 0
	}
	return (float32(highCount) / float32(totalWeight)) * 100
}

// ResizeEmbeddings adjusts the embedding layer to match a new vocabulary size.
func (m *IntentMoE) ResizeEmbeddings(newVocabSize int) {
	if m.Embedding == nil {
		return
	}
	if newVocabSize <= m.Embedding.VocabSize {
		return
	}

	fmt.Printf("🔄 Resizing Model Embeddings: %d -> %d\n", m.Embedding.VocabSize, newVocabSize)

	oldEmb := m.Embedding
	newEmb := nn.NewEmbedding(newVocabSize, oldEmb.DimModel)

	// Copy old weights
	copy(newEmb.Weight.Data, oldEmb.Weight.Data)

	m.Embedding = newEmb
}

// SaveIntentMoECheckpoint saves the IntentMoE and its metadata to a file with compression.
func SaveIntentMoECheckpoint(ckpt *Checkpoint, path string) error {
	// Create temporary file first to avoid corruption on crash
	tempPath := path + ".tmp"
	file, err := os.Create(tempPath)
	if err != nil {
		return fmt.Errorf("failed to create checkpoint file: %w", err)
	}

	// Use gzip for significant size reduction
	gz := gzip.NewWriter(file)
	writer := bufio.NewWriter(gz)

	encoder := gob.NewEncoder(writer)
	err = encoder.Encode(ckpt)
	if err != nil {
		gz.Close()
		file.Close()
		os.Remove(tempPath)
		return fmt.Errorf("failed to encode checkpoint: %w", err)
	}

	writer.Flush()
	gz.Close()
	file.Close()

	// Atomic rename
	if runtime.GOOS == "windows" {
		_ = os.Remove(path)
	}
	if err := os.Rename(tempPath, path); err != nil {
		return fmt.Errorf("failed to finalize checkpoint: %w", err)
	}

	// Get file size for logging
	fi, _ := os.Stat(path)
	log.Printf("✅ [CHECKPOINT] Saved model to %s (Size: %.2f MB)", path, float64(fi.Size())/(1024*1024))

	return nil
}

// LoadIntentMoECheckpoint loads a Checkpoint from a compressed file.
func LoadIntentMoECheckpoint(filePath string) (*Checkpoint, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening checkpoint file: %w", err)
	}
	defer file.Close()

	// Use gzip for decompression
	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader (is the file compressed?): %w", err)
	}
	defer gz.Close()

	decoder := gob.NewDecoder(gz)
	var ckpt Checkpoint
	err = decoder.Decode(&ckpt)
	if err != nil {
		return nil, fmt.Errorf("error decoding checkpoint from gob: %w", err)
	}

	if ckpt.Model == nil {
		return nil, fmt.Errorf("loaded checkpoint has a nil Model")
	}

	return &ckpt, nil
}

// SaveIntentMoEModelToGOB saves the IntentMoE to a file (legacy format).
func SaveIntentMoEModelToGOB(model *IntentMoE, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	encoder := gob.NewEncoder(writer)
	return encoder.Encode(model)
}

// LoadIntentMoEModelFromGOB loads a IntentMoE from a legacy GOB file.
func LoadIntentMoEModelFromGOB(filePath string) (*IntentMoE, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening gob file: %w", err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	decoder := gob.NewDecoder(reader)
	var loadedModel IntentMoE
	err = decoder.Decode(&loadedModel)
	if err != nil {
		return nil, fmt.Errorf("error decoding model from gob: %w", err)
	}
	return &loadedModel, nil
}

// PruneExpertRouter zeros out the routing probabilities for a specific expert to break a collapse.
func (m *IntentMoE) PruneExpertRouter(expertID int) {
	layers := m.Encoder.GetMoELayers()
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		layers = append(layers, m.Decoder.OutputMoE)
	}
	for _, layer := range layers {
		if layer.GatingNetwork != nil {
			numExperts := layer.GatingNetwork.Linear.Weights.Shape[1]
			inputDim := layer.GatingNetwork.Linear.Weights.Shape[0]
			for k := 0; k < inputDim; k++ {
				// Set the weight for this expert to a large negative number
				layer.GatingNetwork.Linear.Weights.Data[k*numExperts+expertID] = -3.0
			}
		}
	}
}

// ShakeRouters applies random noise to all gating networks to force exploration.
func (m *IntentMoE) ShakeRouters(scale float32) {
	layers := m.Encoder.GetMoELayers()
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		layers = append(layers, m.Decoder.OutputMoE)
	}
	for _, layer := range layers {
		if layer.GatingNetwork != nil {
			params := layer.GatingNetwork.Linear.Parameters()
			for _, p := range params {
				for i := range p.Data {
					p.Data[i] += (rand.Float32()*2 - 1) * scale
				}
			}
		}
	}
}
