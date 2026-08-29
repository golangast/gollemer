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
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
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
	gob.Register(&nn.LayerNormalization{})
	gob.Register(&nn.PositionalEmbedding{})
	gob.Register(&FeedForwardExpert{})
	gob.Register(&mainvocab.Vocabulary{})
	gob.Register(&GrammarExpert{})
}

// ClearState clears the intermediate tensors used for backward pass
func (m *IntentMoE) ClearState() {
	if m.Encoder != nil {
		m.Encoder.ClearState()
	}
	if m.EncoderNorm != nil {
		m.EncoderNorm.ClearState()
	}
	if m.EncoderPos != nil {
		m.EncoderPos.ClearState()
	}
	if m.Decoder != nil {
		m.Decoder.ClearState()
	}
	if m.Embedding != nil {
		m.Embedding.ClearState()
	}
}

// Detach removes the computation graph (creator and operation) from the model parameters.
// This is critical before serialization or to free memory after a training batch.
func (m *IntentMoE) Detach() {
	params := m.Parameters()
	for _, param := range params {
		param.Creator = nil
		param.Mask = nil
		param.Operation = nil
	}

	// Clear state for all components
	m.ClearState()

	// Clear decoder specific state which might hold references to the computation graph
	if m.Decoder != nil {
		m.Decoder.InitialHiddenState = nil
		m.Decoder.InitialCellState = nil

		// Clear LSTM cells state
		if m.Decoder.LSTM != nil {
			for _, layer := range m.Decoder.LSTM.Cells {
				for _, cell := range layer {
					cell.InputTensor = nil
					cell.PrevHidden = nil
					cell.PrevCell = nil
				}
			}
		}
	}

	runtime.GC()
}

// SetParamsRequiresGrad toggles the RequiresGrad flag for all model parameters.
// This is useful for disabling gradient tracking during inference to save memory.
func (m *IntentMoE) SetParamsRequiresGrad(requires bool) {
	params := m.Parameters()
	for _, param := range params {
		if param != nil {
			param.RequiresGrad = requires
		}
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

	if m.EncoderNorm != nil {
		// LayerNorm is CPU-only in this version's implementation (ToCPU calls inside)
		// but we call ToGPU to ensure it's where it needs to be.
		m.EncoderNorm.ToGPU()
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

	// Handle top-P (nucleus) sampling
	if topP > 0.0 && topP < 1.0 {
		// First compute probabilities for top-K candidates
		maxLogit := topKCandidates[0].value
		var sumExp float32
		for _, c := range topKCandidates {
			sumExp += float32(math.Exp(float64(c.value - maxLogit)))
		}

		var cumulativeProb float32
		var lastIdx int
		for i, c := range topKCandidates {
			prob := float32(math.Exp(float64(c.value-maxLogit))) / sumExp
			cumulativeProb += prob
			lastIdx = i
			if cumulativeProb >= topP {
				break
			}
		}
		topKCandidates = topKCandidates[:lastIdx+1]
	}

	// Find max logit for numerical stability
	maxLogit := topKCandidates[0].value

	// Compute sum of exponents for truncated candidates
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
func (m *IntentMoE) PredictNext(input *tensor.Tensor, k int, temp float32, suppressedIDs map[int]bool) (int, float32, error) {
	// 1. Forward pass
	// We assume a simplified forward call for single-token prediction
	logits, _, err := m.Forward(0.0, input, input, nil)
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

	// Apply Suppression
	if suppressedIDs != nil {
		for id := range suppressedIDs {
			if id >= 0 && id < len(lastLogits.Data) {
				lastLogits.Data[id] = -1e9
			}
		}
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

// PredictNextToken wraps PredictNext for autoregressive LLM loops
func (m *IntentMoE) PredictNextToken(currentSequence []int, suppressedIDs map[int]bool) int {
	inputData := make([]float32, len(currentSequence))
	for i, id := range currentSequence {
		inputData[i] = float32(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(currentSequence)}, inputData, false)
	idx, _, err := m.PredictNext(inputTensor, 5, 0.8, suppressedIDs) // TopK 5, Temp 0.8
	if err != nil {
		if m.SentenceVocab != nil && m.SentenceVocab.EosID > 0 {
			return m.SentenceVocab.EosID
		}
		return 0
	}
	return idx
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
		logits, _, err := m.Forward(0.0, input, input, nil)
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
	if penalty <= 0 {
		return
	}
	counts := make(map[int]float32)
	for i, id := range generatedIDs {
		// Prioritize the last 5 tokens for penalty
		weight := float32(1.0)
		dist := len(generatedIDs) - 1 - i
		if dist < 5 {
			weight = 2.5 // Stronger penalty for recent tokens
		}
		counts[id] += weight
	}
	for id, weight := range counts {
		if id < 0 || id >= len(logits.Data) {
			continue
		}

		// Subtractive penalty is much more effective than multiplicative for logit suppression.
		// We multiply by weight to increase penalty for recent/frequent tokens,
		// and add an extra boost if the token was the IMMEDIATE previous token.
		p := penalty * weight
		if len(generatedIDs) > 0 && id == generatedIDs[len(generatedIDs)-1] {
			p *= 2.0 // Double penalty for immediate repetition
		}
		logits.Data[id] -= p
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
	RepairArchitecture()
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
	SocialVocab       *mainvocab.Vocabulary
	TechVocab         *mainvocab.Vocabulary
	EmbeddingDim      int // Persisted dimension (e.g., 768) for resizing logic

	// Training Metadata for persistence
	StepCount     int // Total training steps completed
	TrainingPhase int // 0: Init, 1: MLM Done, 2: Seq2Seq Done

	// Structural Guidance
	Tagger     *IntentTagger // Predicts intent and grammar tags (POS) to guide generation
	Rules      *RuleBook     // Formal linguistic rules and grammar skeletons
	Supervisor *Supervisor

	// Diagnostics and Monitoring
	ExpertStats map[string]*ExpertStat // Key: "layerID:expertID"
	Metadata    ModelMetadata
	EncoderNorm *nn.LayerNorm           // Added for stability
	EncoderPos  *nn.PositionalEmbedding // Added for word-order awareness
	// 🚀 PERFORMANCE: Pre-computed grammar type for each token ID.
	// Built once on first use; avoids O(vocabSize) string matching per generation step.
	grammarTypeCache []string

	// 👁️ VISION: Optional vision encoder for multimodal (image/video) input.
	// If nil or if no patches are supplied, the forward pass is pure NLP with zero overhead.
	VisionEncoder *VisionEncoder
	// visionPatches holds the last raw patches supplied during a forward pass so
	// that Backward() can compute the correct gradient for the vision encoder.
	visionPatches [][]float32
}

// buildGrammarTypeCache pre-computes the grammar type for every token in the vocabulary.
// Call this once after the vocabulary is finalised. Subsequent generation steps then do
// a single O(1) slice lookup instead of O(vocabSize) string comparisons.
func (m *IntentMoE) buildGrammarTypeCache() {
	if m.SentenceVocab == nil {
		return
	}
	vSize := m.SentenceVocab.Size()
	if len(m.grammarTypeCache) == vSize {
		return // already built
	}
	m.grammarTypeCache = make([]string, vSize)
	for id := 0; id < vSize; id++ {
		word := m.SentenceVocab.GetWord(id)
		m.grammarTypeCache[id] = MapWordToGrammarType(word)
	}
}

// grammarTypeForID returns the pre-cached grammar type for a token ID, building the
// cache on first call if necessary.
func (m *IntentMoE) grammarTypeForID(id int) string {
	if len(m.grammarTypeCache) == 0 {
		m.buildGrammarTypeCache()
	}
	if id < 0 || id >= len(m.grammarTypeCache) {
		return "OTHER"
	}
	return m.grammarTypeCache[id]
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
	if m.EncoderNorm != nil {
		m.EncoderNorm.ToGPU()
	}
	if m.EncoderPos != nil {
		m.EncoderPos.ToGPU()
	}

	// 🌡️ GPU WARM-UP: Serial compilation of pipelines to prevent race conditions
	// (Vulkan/vkCreateComputePipelines crash) during the first parallel batch.
	m.warmup()
}

// GuessIntent performs advanced intent detection on a query.
// It uses a hybrid approach:
// 1. Neural classification via the Tagger (if available).
// 2. Heuristic keyword mapping for common social intents.
func (m *IntentMoE) GuessIntent(query string) (string, string) {
	q := strings.ToLower(query)

	// Advanced Weighted Scoring
	scores := make(map[string]float32)

	// Social: Greeting Clues
	if strings.Contains(q, "hello") || strings.Contains(q, "hi ") || strings.HasPrefix(q, "hi") {
		scores["social:greeting"] += 2.0
	}
	if strings.Contains(q, "hey") || strings.Contains(q, "morning") || strings.Contains(q, "evening") {
		scores["social:greeting"] += 1.5
	}

	// Social: Identity Clues
	if strings.Contains(q, "who are you") || strings.Contains(q, "your name") {
		scores["social:identity"] += 3.0
	}
	if strings.Contains(q, "what are you") || strings.Contains(q, "creator") {
		scores["social:identity"] += 2.0
	}

	// Social: Status Check Clues
	if strings.Contains(q, "how are you") || strings.Contains(q, "how you doing") {
		scores["social:status_check"] += 3.0
	}
	if strings.Contains(q, "how's it going") || strings.Contains(q, "everything ok") {
		scores["social:status_check"] += 2.5
	}

	// Social: Entertainment Clues
	if strings.Contains(q, "joke") || strings.Contains(q, "funny") || strings.Contains(q, "story") {
		scores["social:entertainment"] += 2.0
	}

	// Neural Verification (if Tagger is available)
	if m.Tagger != nil {
		tokens := strings.Fields(q)
		ids := make([]float32, len(tokens))
		for i, t := range tokens {
			id := m.SentenceVocab.GetTokenID(t)
			if id < 0 {
				id = 1
			}
			ids[i] = float32(id)
		}
		input := tensor.NewTensor([]int{1, len(ids)}, ids, false)
		// Assuming the tagger returns intent logits as the first output
		intentLogits, _, _ := m.Tagger.Forward(input)
		if intentLogits != nil {
			// Find top intent from neural model
			bestIdx := 0
			bestVal := float32(-1e9)
			for i, v := range intentLogits.Data {
				if v > bestVal {
					bestVal = v
					bestIdx = i
				}
			}
			// Boost the neural winner
			neuralIntent := fmt.Sprintf("social:neural_%d", bestIdx)
			scores[neuralIntent] += 1.0
		}
	}

	// Pick winner
	bestKey := "social:general"
	maxScore := float32(0.5)
	for k, s := range scores {
		if s > maxScore {
			maxScore = s
			bestKey = k
		}
	}

	parts := strings.Split(bestKey, ":")
	return parts[0], parts[1]
}

// GenerateGuidedSentence attempts to generate a response that follows a grammatical skeleton.
// It uses the Tagger to predict the 'shape' of the answer before filling in the words.
var verbose_thinking = false

// AnalyzePartialQuery implements streaming prediction by analyzing the first few words
// of the user's input and pre-loading relevant expert cartridges into RAM with zero latency.
func (m *IntentMoE) AnalyzePartialQuery(partialQuery string) {
	if m.Supervisor == nil || m.Supervisor.CartridgeMgr == nil {
		return
	}

	// Create a mean embedding of the partial query to use for semantic triage
	tokens := strings.Fields(strings.ToLower(partialQuery))
	ids := make([]float32, len(tokens))
	for i, t := range tokens {
		id := m.SentenceVocab.GetTokenID(t)
		if id < 0 {
			id = 1
		}
		ids[i] = float32(id)
	}

	var queryEmb []float32
	if len(ids) > 0 {
		inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)
		if emb, err := m.Embedding.Forward(inputT); err == nil && len(emb.Data) > 0 {
			// Calculate mean embedding for semantic triage
			embDim := emb.Shape[1]
			queryEmb = make([]float32, embDim)
			for i := 0; i < len(ids); i++ {
				for d := 0; d < embDim; d++ {
					queryEmb[d] += emb.Data[i*embDim+d]
				}
			}
			for d := 0; d < embDim; d++ {
				queryEmb[d] /= float32(len(ids))
			}
		}
	}

	cartridgePath := m.Supervisor.TriageCartridge(partialQuery, queryEmb)
	if cartridgePath != "" {
		log.Printf("⚡ Streaming Prediction: Detected namespace mid-sentence. Pre-loading %s...", cartridgePath)
		m.Supervisor.CartridgeMgr.PreloadCartridge(cartridgePath, 0, 0)
	}
}

func (m *IntentMoE) GenerateGuidedSentence(query string, maxLen int) (string, []string) {
	if m.Supervisor != nil && !verbose_thinking {
		// Use the supervisor's multi-pass logic if available
		// but avoid recursion by checking a flag or using a different entry point.
	}
	parent, child := m.GuessIntent(query)
	log.Printf("🔮 Sophisticated Intent Detection: [%s / %s]", parent, child)

	// 🎮 DYNAMIC EXPERT CARTRIDGE HOT-SWAPPING & PRE-LOADING
	var mountedExpertID int = -1
	var loadedCartridge string
	if m.Supervisor != nil && m.Supervisor.CartridgeMgr != nil {
		// Calculate mean embedding of the full query for triage
		tokens := strings.Fields(strings.ToLower(query))
		ids := make([]float32, len(tokens))
		for i, t := range tokens {
			id := m.SentenceVocab.GetTokenID(t)
			if id < 0 {
				id = 1
			}
			ids[i] = float32(id)
		}
		var queryEmb []float32
		if len(ids) > 0 {
			inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)
			if emb, err := m.Embedding.Forward(inputT); err == nil && len(emb.Data) > 0 {
				embDim := emb.Shape[1]
				queryEmb = make([]float32, embDim)
				for i := 0; i < len(ids); i++ {
					for d := 0; d < embDim; d++ {
						queryEmb[d] += emb.Data[i*embDim+d]
					}
				}
				for d := 0; d < embDim; d++ {
					queryEmb[d] /= float32(len(ids))
				}
			}
		}

		cartridgePath := m.Supervisor.TriageCartridge(query, queryEmb)
		if cartridgePath != "" {
			log.Printf("🔌 Hot-Swapping: Loading expert cartridge %s into RAM...", cartridgePath)
			err := m.Supervisor.CartridgeMgr.LoadCartridge(cartridgePath, 0, 0)
			if err == nil {
				m.Supervisor.CartridgeMgr.mu.Lock()
				expert := m.Supervisor.CartridgeMgr.Loaded[cartridgePath]
				m.Supervisor.CartridgeMgr.mu.Unlock()

				if m.Decoder.OutputMoE != nil {
					mountedExpertID, err = m.Supervisor.MountCartridgeToLayer(m, len(m.Encoder.GetMoELayers()), expert)
					if err != nil {
						log.Printf("⚠️ Failed to mount cartridge: %v", err)
						mountedExpertID = -1
					} else {
						loadedCartridge = cartridgePath
						// 4. Force routing bias for the entire sequence to the new expert
						for i := 0; i < maxLen; i++ {
							m.Decoder.OutputMoE.StepRoutingBias[i] = make([]float32, len(m.Decoder.OutputMoE.Experts))
							m.Decoder.OutputMoE.StepRoutingBias[i][mountedExpertID] = 20.0 // Overwhelming bias
						}
					}
				}
			} else {
				log.Printf("⚠️ Failed to load cartridge: %v", err)
			}
		}

		// Ensure cleanup after generation
		defer func() {
			if mountedExpertID != -1 && m.Decoder.OutputMoE != nil && loadedCartridge != "" {
				log.Printf("🧹 Unmounting and unloading %s to save memory...", loadedCartridge)
				m.Supervisor.UnmountCartridgeFromLayer(m, len(m.Encoder.GetMoELayers()), mountedExpertID)
				m.Supervisor.CartridgeMgr.UnloadCartridge(loadedCartridge)
				// Clean up routing bias
				m.Decoder.OutputMoE.StepRoutingBias = make(map[int][]float32)
			}
		}()
	}

	rule, hasRule := m.Rules.GetRuleByIntent(parent, child)

	// 1. Encode query
	tokens := strings.Fields(strings.ToLower(query))
	ids := make([]float32, len(tokens))
	for i, t := range tokens {
		id := m.SentenceVocab.GetTokenID(t)
		if id < 0 {
			id = 1
		}
		ids[i] = float32(id)
	}
	inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

	emb, _ := m.Embedding.Forward(inputT)
	ctx, _ := m.Encoder.Forward(emb)
	if m.EncoderNorm != nil {
		ctx, _ = m.EncoderNorm.Forward(ctx)
	}

	// 2. Generate with "Hard Rule Enforcement"
	var generated []string
	var decodedIDs []int

	hidden := m.Decoder.InitialHiddenState
	cell := m.Decoder.InitialCellState
	currentID := m.SentenceVocab.BosID
	if currentID < 0 {
		currentID = 0
	}

	for i := 0; i < maxLen; i++ {
		logits, nextH, nextC, err := m.Decoder.Step(currentID, ctx, hidden, cell)
		if err != nil {
			break
		}
		hidden = nextH
		cell = nextC

		// SENSITIVE PRUNING: Only allow words that match the grammar skeleton (if available)
		if hasRule && i < len(rule.GrammarSkeleton) {
			expectedType := rule.GrammarSkeleton[i]
			// Use pre-cached grammar types for O(1) lookup per token (avoids O(vocabSize) string matching)
			for idx := 0; idx < len(logits.Data); idx++ {
				actualType := m.grammarTypeForID(idx)
				if expectedType != "OTHER" && actualType != expectedType {
					logits.Data[idx] -= 5.0
				}
			}
		}

		// Apply Intent-Based Logit Boosting (Soft Rules)
		m.applyIntentBoost(logits, parent, child)

		// Apply Repetition Penalty (prevent "doing doing doing")
		ApplyRepetitionPenalty(logits, decodedIDs, 1.5)

		// Stuck Detector: Force diversity if immediate repetition is detected
		if len(decodedIDs) >= 1 {
			lastID := decodedIDs[len(decodedIDs)-1]
			if lastID < len(logits.Data) {
				logits.Data[lastID] -= 2.0 // Discourage immediate repeat
			}
		}

		// Greedy choice
		bestID := 0
		bestVal := float32(-math.MaxFloat32)
		for idx, val := range logits.Data {
			if val > bestVal {
				bestVal = val
				bestID = idx
			}
		}

		if bestID == m.SentenceVocab.EosID {
			break
		}

		word := m.SentenceVocab.GetWord(bestID)
		generated = append(generated, word)
		decodedIDs = append(decodedIDs, bestID)
		currentID = bestID
	}

	return strings.Join(generated, " "), generated
}

// GenerateSupervisedSentence uses the supervisor to ensure high-quality output.
func (m *IntentMoE) GenerateSupervisedSentence(query string) string {
	if m.Supervisor == nil {
		m.Supervisor = NewSupervisor()
	}

	resp, _ := m.Supervisor.SuperviseSentenceCreation(m, query)
	return resp
}

func (m *IntentMoE) applyIntentBoost(logits *tensor.Tensor, parent, child string) {
	// Use pre-cached grammar types for O(1) lookup per token (avoids O(vocabSize) string matching)
	for i := 0; i < len(logits.Data); i++ {
		gType := m.grammarTypeForID(i)

		// Boost structural words subtly
		switch gType {
		case "PRON", "VERB", "AUX", "PREP":
			logits.Data[i] += 0.2
		}

		// Boost intent-specific words subtly
		if parent == "social" {
			switch gType {
			case "ADJ", "GREET", "NOUN":
				logits.Data[i] += 0.15
			}
			if child == "greeting" && gType == "GREET" {
				logits.Data[i] += 0.3
			}
			if child == "identity" && gType == "NOUN" {
				logits.Data[i] += 0.3
			}
		}
	}
}

func isStructuralWord(w string) bool {
	switch strings.ToLower(w) {
	case "i", "you", "is", "are", "am", "the", "a", "to", "and", "it", "that", "in", "for":
		return true
	}
	return false
}

func isSocialWord(w string) bool {
	switch strings.ToLower(w) {
	case "good", "fine", "well", "great", "nice", "happy", "doing":
		return true
	}
	return false
}

func isGreetingWord(w string) bool {
	switch strings.ToLower(w) {
	case "hello", "hi", "hey", "morning", "evening", "there":
		return true
	}
	return false
}

func isIdentityWord(w string) bool {
	switch strings.ToLower(w) {
	case "name", "gollemer", "ai", "assistant", "model", "bot":
		return true
	}
	return false
}

// SocialSystemTokens are special formatting tokens that should never bleed into
// social context state calculations or generated responses.
var SocialSystemTokens = []string{
	"__intent__", "__ques__", "__ans__", "social", ":",
}

// MaskSocialSystemTokens sets the logit for each social system/format token to -1e9
// so they are never sampled in social contexts. Call this from any generation loop
// that operates under social intent.
func (m *IntentMoE) MaskSocialSystemTokens(logits *tensor.Tensor) {
	if m.SentenceVocab == nil || logits == nil {
		return
	}
	for _, tok := range SocialSystemTokens {
		id := m.SentenceVocab.GetTokenID(tok)
		if id >= 0 && id < len(logits.Data) {
			logits.Data[id] = -1e9
		}
	}
}

// MaskSocialSystemTokensInIDs returns true if any of the given IDs correspond to
// social system tokens, used to filter generated output sequences.
func (m *IntentMoE) ContainsSocialSystemToken(ids []int) bool {
	if m.SentenceVocab == nil {
		return false
	}
	systemIDs := make(map[int]bool, len(SocialSystemTokens))
	for _, tok := range SocialSystemTokens {
		id := m.SentenceVocab.GetTokenID(tok)
		if id >= 0 {
			systemIDs[id] = true
		}
	}
	for _, id := range ids {
		if systemIDs[id] {
			return true
		}
	}
	return false
}

func (m *IntentMoE) CalculateGrammarLossStrings(generatedWords []string, parent, child string) float32 {
	ids := make([]int, len(generatedWords))
	for i, w := range generatedWords {
		ids[i] = m.SentenceVocab.GetTokenID(w)
	}
	return m.CalculateGrammarLoss(ids, parent, child)
}

// CalculateGrammarLossByID computes a penalty for sentences that violate the RuleBook's grammar skeletons.
// Uses Token IDs directly to avoid string allocations during training.
func (m *IntentMoE) CalculateGrammarLoss(generatedIDs []int, parent, child string) float32 {
	if m.Rules == nil {
		return 0
	}
	rule, ok := m.Rules.GetRuleByIntent(parent, child)
	if !ok {
		return 0
	}

	skeleton := rule.GrammarSkeleton
	if len(skeleton) == 0 {
		return 0
	}

	// 🆕 STRICT: Mask system/format tokens in social context.
	// These tokens should never bleed into the social state calculation.
	if parent == "social" && m.ContainsSocialSystemToken(generatedIDs) {
		return 10.0 // Heavy penalty to force these tokens out of social state
	}

	var penalty float32 = 0.0
	maxCheck := len(generatedIDs)
	if len(skeleton) < maxCheck {
		maxCheck = len(skeleton)
	}

	for i := 0; i < maxCheck; i++ {
		// Use pre-computed grammar type cache for O(1) lookup
		actualType := m.grammarTypeForID(generatedIDs[i])
		expectedType := skeleton[i]

		if expectedType != "OTHER" && actualType != expectedType {
			// Penalty for wrong structural category (word salad prevention)
			penalty += 0.5
		}

		// N-gram Window Feature (Tri-grams)
		prevType := "BOS"
		if i > 0 {
			prevType = m.grammarTypeForID(generatedIDs[i-1])
		}
		nextType := "EOS"
		if i < len(generatedIDs)-1 {
			nextType = m.grammarTypeForID(generatedIDs[i+1])
		}

		penalty += rule.EvaluateWindow(prevType, actualType, nextType)
	}

	// Bonus for required keywords (we check if any of the generated IDs match the keyword IDs)
	// For now, keywords are still strings in the RuleBook, so we check them once.
	// But we can optimize this further if keywords are also cached as IDs.
	for _, kw := range rule.RequiredKeywords {
		found := false
		kwLower := strings.ToLower(kw)
		for _, id := range generatedIDs {
			if strings.ToLower(m.SentenceVocab.GetWord(id)) == kwLower {
				found = true
				break
			}
		}
		if !found {
			penalty += 0.2 // penalty for missing essential content
		}
	}

	//  Sequence Coherence Reward: Penalize repetitive local state transitions
	// (e.g., looping between "to", "you", ",", "i").
	if len(generatedIDs) >= 4 {
		for i := 0; i < len(generatedIDs)-3; i++ {
			// Check for repetitive bigrams: ABAB pattern
			if generatedIDs[i] == generatedIDs[i+2] && generatedIDs[i+1] == generatedIDs[i+3] {
				penalty += 0.5 // High penalty for local loops
			}
		}
	}
	// Penalize immediate word repetition (if not already handled by repetition penalty)
	for i := 0; i < len(generatedIDs)-1; i++ {
		if generatedIDs[i] == generatedIDs[i+1] && generatedIDs[i] != m.SentenceVocab.PaddingTokenID {
			penalty += 0.3
		}
	}

	return penalty
}
func (m *IntentMoE) CalculateSequenceSimilarityStrings(generatedWords, targetWords []string) float32 {
	gIDs := make([]int, len(generatedWords))
	for i, w := range generatedWords {
		gIDs[i] = m.SentenceVocab.GetTokenID(w)
	}
	tIDs := make([]int, len(targetWords))
	for i, w := range targetWords {
		tIDs[i] = m.SentenceVocab.GetTokenID(w)
	}
	return m.CalculateSequenceSimilarity(gIDs, tIDs)
}

// CalculateSequenceSimilarity computes a reward [0-1] based on how many target words are present
// in the generated output. Uses Token IDs to avoid string hashing and map overhead.
func (m *IntentMoE) CalculateSequenceSimilarity(generatedIDs, targetIDs []int) float32 {
	if len(targetIDs) == 0 {
		return 0
	}

	// Use an ID-based frequency count instead of map[string]int
	// Vocab size is ~4.7k, so a fixed-size array or a sparse slice is fast.
	// Find max ID to determine buffer size
	maxID := 0
	for _, id := range targetIDs {
		if id > maxID {
			maxID = id
		}
	}
	for _, id := range generatedIDs {
		if id > maxID {
			maxID = id
		}
	}

	counts := make([]int, maxID+1)
	for _, id := range targetIDs {
		counts[id]++
	}

	matches := 0
	for _, id := range generatedIDs {
		if id <= maxID && counts[id] > 0 {
			matches++
			counts[id]--
		}
	}

	return float32(matches) / float32(len(targetIDs))
}
func (m *IntentMoE) EncoderForward(input *tensor.Tensor, mask *tensor.Tensor) (*tensor.Tensor, error) {
	emb, err := m.Embedding.Forward(input)
	if err != nil {
		return nil, err
	}

	// Apply Positional Encoding if available (matching Forward logic)
	if m.EncoderPos != nil {
		emb, err = m.EncoderPos.Forward(emb)
		if err != nil {
			return nil, err
		}
	}

	enc, err := m.Encoder.Forward(emb)
	if err != nil {
		return nil, err
	}
	if m.EncoderNorm != nil {
		enc, err = m.EncoderNorm.Forward(enc)
		if err != nil {
			return nil, err
		}
	}
	return enc, nil
}

// CalculateGatingEntropy computes the average Shannon Entropy across all active MoE layers.
func (m *IntentMoE) CalculateGatingEntropy() float32 {
	layers := m.Encoder.GetMoELayers()
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		layers = append(layers, m.Decoder.OutputMoE)
	}

	if len(layers) == 0 {
		return 0
	}

	var totalEntropy float32
	var count int
	for _, layer := range layers {
		if layer.GateOutputs != nil {
			// Calculate Shannon Entropy: -sum(p * log2(p))
			var entropy float32
			probs := layer.GateOutputs.Data
			numExperts := len(layer.Experts)
			numTokens := len(probs) / numExperts

			if numTokens == 0 {
				continue
			}

			for t := 0; t < numTokens; t++ {
				for e := 0; e < numExperts; e++ {
					p := probs[t*numExperts+e]
					if p > 1e-10 {
						entropy -= p * float32(math.Log2(float64(p)))
					}
				}
			}
			totalEntropy += entropy / float32(numTokens)
			count++
		}
	}

	if count == 0 {
		return 0
	}
	return totalEntropy / float32(count)
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

// appendGrammarExperts grows an existing MoELayer by 8 GrammarExperts (one per POS role).
// The gating network weight matrix is widened to include columns for the new experts.
func appendGrammarExperts(layer *MoELayer, embeddingDim int, count int) error {
	numGrammar := count
	startID := len(layer.Experts)

	// Detect dimensions from the existing architecture to ensure specialized experts fit perfectly.
	layerInputDim := embeddingDim
	layerOutputDim := embeddingDim
	if len(layer.Experts) > 0 {
		// Use the dimensions of the existing experts
		layerInputDim = layer.InputDim
		layerOutputDim = layer.OutputDim
	} else if layer.GatingNetwork != nil && layer.GatingNetwork.Linear.Weights != nil {
		layerInputDim = layer.GatingNetwork.Linear.Weights.Shape[0]
	}

	for i := 0; i < numGrammar; i++ {
		ge, err := NewGrammarExpert(startID+i, i, layerInputDim, layerOutputDim)
		if err != nil {
			return fmt.Errorf("grammar expert %d: %w", i, err)
		}
		layer.Experts = append(layer.Experts, ge)
	}

	// Widen the gating network weight matrix from [inputDim, oldN] -> [inputDim, oldN+8]
	// New columns are initialised with small random values so the router can explore.
	gn := layer.GatingNetwork
	oldW := gn.Linear.Weights
	inputDim := oldW.Shape[0]
	oldN := oldW.Shape[1]
	newN := oldN + numGrammar

	newWData := make([]float32, inputDim*newN)
	for row := 0; row < inputDim; row++ {
		// Copy existing columns
		copy(newWData[row*newN:row*newN+oldN], oldW.Data[row*oldN:(row+1)*oldN])
		// New grammar-expert columns: small positive bias so router tries them
		for col := oldN; col < newN; col++ {
			newWData[row*newN+col] = (rand.Float32() - 0.5) * 0.02
		}
	}
	newW := tensor.NewTensor([]int{inputDim, newN}, newWData, true)
	gn.Linear.Weights = newW

	// Widen bias if present
	if gn.Linear.Biases != nil {
		oldB := gn.Linear.Biases.Data
		newBData := make([]float32, newN)
		copy(newBData, oldB)
		gn.Linear.Biases = tensor.NewTensor([]int{newN}, newBData, true)
	}

	// Widen NoiseLinear (used for exploration)
	if gn.NoiseLinear != nil {
		oldNW := gn.NoiseLinear.Weights
		newNWData := make([]float32, inputDim*newN)
		for row := 0; row < inputDim; row++ {
			copy(newNWData[row*newN:row*newN+oldN], oldNW.Data[row*oldN:(row+1)*oldN])
			for col := oldN; col < newN; col++ {
				newNWData[row*newN+col] = (rand.Float32() - 0.5) * 0.02
			}
		}
		gn.NoiseLinear.Weights = tensor.NewTensor([]int{inputDim, newN}, newNWData, true)

		if gn.NoiseLinear.Biases != nil {
			oldNB := gn.NoiseLinear.Biases.Data
			newNBData := make([]float32, newN)
			copy(newNBData, oldNB)
			gn.NoiseLinear.Biases = tensor.NewTensor([]int{newN}, newNBData, true)
		}
	}

	// Reinitialize LayerNorm for the new gating dimension
	gn.LayerNorm = nn.NewLayerNorm(newN)

	// Update frozen / stagnation / multiplier slices
	layer.ExpertFrozen = append(layer.ExpertFrozen, make([]bool, numGrammar)...)
	layer.StagnationCounters = append(layer.StagnationCounters, make([]int, numGrammar)...)
	extra := make([]float32, numGrammar)
	for i := range extra {
		extra[i] = 1.0
	}
	layer.ExpertGradMultiplier = append(layer.ExpertGradMultiplier, extra...)

	// Expand the new dynamic tracking slices
	extraHealth := make([]float64, numGrammar)
	for i := range extraHealth {
		extraHealth[i] = 1.0
	}
	layer.ExpertHealth = append(layer.ExpertHealth, extraHealth...)

	extraTimes := make([]time.Time, numGrammar)
	now := time.Now()
	for i := range extraTimes {
		extraTimes[i] = now
	}
	layer.ExpertLastUsedAt = append(layer.ExpertLastUsedAt, extraTimes...)

	extraPinned := make([]bool, numGrammar)
	for i := range extraPinned {
		extraPinned[i] = true // Grammar/syntactic experts are pinned by default
	}
	layer.ExpertPinned = append(layer.ExpertPinned, extraPinned...)

	extraRoles := make([]string, numGrammar)
	for i := 0; i < numGrammar; i++ {
		extraRoles[i] = GrammarRoles[i%len(GrammarRoles)]
	}
	layer.ExpertRole = append(layer.ExpertRole, extraRoles...)

	// Extend ExpertOutputScale
	extraScales := make([]float32, numGrammar)
	for i := range extraScales {
		extraScales[i] = 1.0
	}
	layer.ExpertOutputScale = append(layer.ExpertOutputScale, extraScales...)

	// Update NumExperts
	layer.NumExperts = len(layer.Experts)

	return nil
}

// beamCandidate holds a partial sequence for beam search.
type beamCandidate struct {
	ids         []int
	logProb     float64 // cumulative log-probability
	hiddenState *tensor.Tensor
	cellState   *tensor.Tensor
	finished    bool
}

// BeamSearchDecode generates a response using beam search to guarantee a
// well-formed sentence.  It maintains `beamWidth` candidate sequences at each
// step, expands each by the top-K next tokens, and returns the highest-scoring
// completed sequence (one that ended with EOS or reached maxLen).
//
// This replaces the per-token sampling used in GenerateSocialResponse and
// GreedySearchDecodeWithTemp, which frequently produce word-salad output on
// small training sets.
func (m *IntentMoE) BeamSearchDecode(
	contextVector *tensor.Tensor,
	maxLen, sosToken, eosToken, beamWidth int,
	temperature float32,
	repetitionPenalty float32,
	rule *IntentRule, // Optional structural guidance
	suppressedIDs map[int]bool, // Optional token IDs to hard-suppress (set to -1e9)
) ([]int, error) {
	if beamWidth <= 0 {
		beamWidth = 4
	}
	if temperature <= 0 {
		temperature = 0.7
	}

	// Take first batch element
	var err error
	contextVector, err = contextVector.Slice(0, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("BeamSearchDecode: slice context: %w", err)
	}

	batchSize := 1
	hiddenSize := m.Decoder.LSTM.HiddenSize

	initHidden, err := contextVector.Mean(1)
	if err != nil {
		return nil, fmt.Errorf("BeamSearchDecode: mean: %w", err)
	}
	initHidden, _ = initHidden.Reshape([]int{batchSize, contextVector.Shape[2]})
	if initHidden.Shape[1] != hiddenSize {
		if initHidden.Shape[1] > hiddenSize {
			initHidden, _ = initHidden.Slice(1, 0, hiddenSize)
		} else {
			pad := tensor.NewTensor([]int{batchSize, hiddenSize - initHidden.Shape[1]}, make([]float32, batchSize*(hiddenSize-initHidden.Shape[1])), false)
			initHidden, _ = tensor.Concat([]*tensor.Tensor{initHidden, pad}, 1)
		}
	}
	initCell := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	// Seed beam with BOS
	beams := []*beamCandidate{
		{
			ids:         []int{sosToken},
			logProb:     0.0,
			hiddenState: initHidden,
			cellState:   initCell,
		},
	}

	completed := make([]*beamCandidate, 0, beamWidth)

	for step := 0; step < maxLen && len(completed) < beamWidth; step++ {
		var nextBeams []*beamCandidate

		for _, cand := range beams {
			if cand.finished {
				completed = append(completed, cand)
				continue
			}

			lastID := cand.ids[len(cand.ids)-1]
			inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(lastID)}, false)

			logits, newHidden, newCell, _, err := m.Decoder.DecodeStep(inputT, cand.hiddenState, cand.cellState, contextVector, step)
			if err != nil {
				inputT.Release()
				continue
			}
			logits.ToCPU()

			// Suppress EOS for the first 3 steps
			if step < 3 && eosToken >= 0 && eosToken < len(logits.Data) {
				logits.Data[eosToken] = -1e9
			}
			// Suppress SOS/BOS from ever appearing in decoded output
			if sosToken >= 0 && sosToken < len(logits.Data) {
				logits.Data[sosToken] = -1e9
			}

			// Repetition penalty
			ApplyRepetitionPenalty(logits, cand.ids, repetitionPenalty)

			// 🚫 Social-context technical vocabulary suppression
			for id := range suppressedIDs {
				if id >= 0 && id < len(logits.Data) {
					logits.Data[id] = -1e9
				}
			}
			// 🧬 STRUCTURAL GUIDANCE (Structural Grammar Penalty)
			// Use pre-cached grammar types for O(1) lookup per token.
			if rule != nil && len(rule.GrammarSkeleton) > 0 {
				ruleStep := len(cand.ids) - 1
				if ruleStep < len(rule.GrammarSkeleton) {
					expectedType := rule.GrammarSkeleton[ruleStep]
					for idx, v := range logits.Data {
						if v < -1e8 {
							continue
						} // Skip already suppressed tokens
						actualType := m.grammarTypeForID(idx)
						if expectedType != "OTHER" && actualType != expectedType {
							logits.Data[idx] -= 3.0
						}
					}
				}
			}

			// 🚫 N-gram blocking: suppress any token that would create a repeated bigram or trigram
			n := len(cand.ids)
			if n >= 2 {
				lastTwo := [2]int{cand.ids[n-2], cand.ids[n-1]}
				for tok := range logits.Data {
					if cand.ids[n-1] == lastTwo[0] && tok == lastTwo[1] {
						logits.Data[tok] = -1e9 // block repeated bigram
					}
				}
			}
			if n >= 4 {
				// Block any token that would repeat the last trigram
				last3 := [3]int{cand.ids[n-3], cand.ids[n-2], cand.ids[n-1]}
				for i := 0; i < n-3; i++ {
					if cand.ids[i] == last3[0] && cand.ids[i+1] == last3[1] && cand.ids[i+2] == last3[2] {
						// The next token after this trigram occurrence should be blocked
						if i+3 < n {
							nextAfter := cand.ids[i+3]
							if nextAfter >= 0 && nextAfter < len(logits.Data) {
								logits.Data[nextAfter] = -1e9
							}
						}
						break
					}
				}
			}

			// Apply Temperature scaling
			for i := range logits.Data {
				logits.Data[i] /= temperature
			}

			// Log-softmax for numerical stability
			vocabSize := len(logits.Data)
			maxL := logits.Data[0]
			for _, v := range logits.Data {
				if v > maxL {
					maxL = v
				}
			}
			var sumExp float64
			for _, v := range logits.Data {
				sumExp += math.Exp(float64(v - maxL))
			}
			logSum := math.Log(sumExp) + float64(maxL)

			// Pick top-beamWidth tokens
			type scored struct {
				id      int
				logProb float64
			}
			topK := beamWidth * 2
			if topK > vocabSize {
				topK = vocabSize
			}
			tops := make([]scored, 0, topK)
			for i, v := range logits.Data {
				lp := float64(v) - logSum
				tops = append(tops, scored{i, lp})
			}
			sort.Slice(tops, func(a, b int) bool { return tops[a].logProb > tops[b].logProb })
			tops = tops[:topK]

			for _, s := range tops {
				newCand := &beamCandidate{
					ids:         append(append([]int{}, cand.ids...), s.id),
					logProb:     cand.logProb + s.logProb,
					hiddenState: newHidden,
					cellState:   newCell,
					finished:    s.id == eosToken,
				}
				nextBeams = append(nextBeams, newCand)
			}
		}

		// Prune: keep top-beamWidth by logProb / length (length-normalised)
		sort.Slice(nextBeams, func(a, b int) bool {
			lenA := float64(len(nextBeams[a].ids))
			lenB := float64(len(nextBeams[b].ids))
			if lenA < 1 {
				lenA = 1
			}
			if lenB < 1 {
				lenB = 1
			}
			return nextBeams[a].logProb/lenA > nextBeams[b].logProb/lenB
		})
		if len(nextBeams) > beamWidth {
			nextBeams = nextBeams[:beamWidth]
		}
		beams = nextBeams
	}

	// Merge finished and remaining beams
	completed = append(completed, beams...)
	if len(completed) == 0 {
		return nil, fmt.Errorf("BeamSearchDecode: no candidates generated")
	}

	// Pick best by length-normalised log-probability
	sort.Slice(completed, func(a, b int) bool {
		lenA := float64(len(completed[a].ids))
		lenB := float64(len(completed[b].ids))
		if lenA < 1 {
			lenA = 1
		}
		if lenB < 1 {
			lenB = 1
		}
		return completed[a].logProb/lenA > completed[b].logProb/lenB
	})

	best := completed[0].ids
	// Strip BOS and EOS
	var result []int
	for _, id := range best {
		if id != sosToken && id != eosToken {
			result = append(result, id)
		}
	}
	return result, nil
}

// NewIntentMoE creates a new IntentMoE model.
func NewIntentMoE(vocabSize, embeddingDim, numExperts, parentVocabSize, childVocabSize, sentenceVocabSize, maxAttentionHeads int) (*IntentMoE, error) {
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)

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
		// 👁️ Initialize the VisionEncoder (16x16 patches → 512-dim tokens).
		// PatchDim=256 matches the 16×16 luma patch from vision_capture/main.go.
		VisionEncoder: NewVisionEncoder(256, embeddingDim),
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
func NewHybridIntentMoE(vocabSize, embeddingDim, numExperts, parentVocabSize, childVocabSize, sentenceVocabSize, maxAttentionHeads int) (*IntentMoE, error) {
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)
	k := 2
	if numExperts <= 1 {
		k = 1
	}

	// 1. Create the inner LLM Encoder (MoE Stack with 4 layers for deeper reasoning)
	expertBuilder := func(expertIdx int) (Expert, error) {
		return NewFeedForwardExpert(embeddingDim, embeddingDim*2, embeddingDim)
	}
	l0, err := NewMoELayer(embeddingDim, embeddingDim, numExperts, k, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE Layer 0: %w", err)
	}
	l1, err := NewMoELayer(embeddingDim, embeddingDim, numExperts, k, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE Layer 1: %w", err)
	}

	// Grammar experts are optional for resource-constrained training runs.
	// On small models, auto-appending 8 grammar experts doubles the MoE size and
	// can trigger OOMs during the social curriculum. Keep the default compact.
	grammarExpertCount := 0
	if numExperts >= 8 && embeddingDim >= 384 {
		grammarExpertCount = 4
	}
	for _, layer := range []*MoELayer{l0, l1} {
		if grammarExpertCount > 0 {
			if err := appendGrammarExperts(layer, embeddingDim, grammarExpertCount); err != nil {
				return nil, fmt.Errorf("failed to append grammar experts: %w", err)
			}
		}
	}

	llmEncoder := NewMoEStack(l0, l1)

	// 2. Wrap it with HybridLLMGNNEncoder
	hybridEncoder, err := NewHybridLLMGNNEncoder(llmEncoder, embeddingDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create HybridLLMGNNEncoder: %w", err)
	}

	// 3. Initialize Decoder
	decoder, err := NewRNNDecoder(embeddingDim, sentenceVocabSize, embeddingDim, maxAttentionHeads, 1, 0.0, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create RNN decoder: %w", err)
	}

	// Decoder output MoE only gets grammar experts when the training run is large enough
	// to afford them; the compact social curriculum stays lean to avoid OOM kills.
	if decoder.OutputMoE != nil && grammarExpertCount > 0 {
		if err := appendGrammarExperts(decoder.OutputMoE, embeddingDim, grammarExpertCount); err != nil {
			return nil, fmt.Errorf("failed to append decoder grammar experts: %w", err)
		}
	}

	// 4. Initialize EncoderNorm and Positional Encoding
	encoderNorm := nn.NewLayerNorm(embeddingDim)
	encoderPos := nn.NewPositionalEmbedding(128, embeddingDim)

	model := &IntentMoE{
		Encoder:           hybridEncoder,
		EncoderNorm:       encoderNorm,
		EncoderPos:        encoderPos,
		Decoder:           decoder,
		Embedding:         embedding,
		EmbeddingDim:      embeddingDim,
		SentenceVocabSize: sentenceVocabSize,
		SentenceVocab:     nil,
	}

	// Ensure decoder starts with a reasonable multiplier if not already set
	if decoder.ContextMultiplier == 0 {
		decoder.ContextMultiplier = 15.0
	}

	// 🧬 Initialize ActiveLayers tracking for load-balancing
	model.RebuildActiveLayers()

	return model, nil
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

	if m.Decoder != nil {
		m.Decoder.LastQueryTokens = queryTokenIDs
		m.Decoder.LastTargetTokens = targetTokenIDs
		if len(m.Decoder.TokenToWord) == 0 && m.SentenceVocab != nil {
			m.Decoder.TokenToWord = m.SentenceVocab.TokenToWord
		}
	}

	// Pass token IDs through embedding layer
	queryEmbeddings, err := m.Embedding.Forward(queryTokenIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("embedding layer forward failed: %w", err)
	}

	// Apply Positional Encoding to query embeddings for word-order awareness
	if m.EncoderPos != nil {
		queryEmbeddings, err = m.EncoderPos.Forward(queryEmbeddings)
		if err != nil {
			return nil, nil, fmt.Errorf("encoder positional embedding failed: %w", err)
		}
	}

	// 👁️ VISION: If visionPatches were set via SetVisionPatches before this call,
	// project them into token embeddings and prepend them to the text sequence.
	// For pure-text batches (visionPatches is nil) this block is skipped entirely.
	m.visionPatches = nil // reset from any prior call
	if len(inputs) >= 4 {
		// The caller can optionally pass raw patches as a 4th input signal.
		// Here we support the convention of passing a dummy non-nil tensor to
		// signal that patches were already set via m.visionPatches directly.
	}
	if m.VisionEncoder != nil && len(m.visionPatches) > 0 {
		visionTokens := m.VisionEncoder.Forward(m.visionPatches)
		if len(visionTokens) > 0 {
			// Convert the [][]float32 vision tokens into a tensor and flatten
			// them into the same embedding space so they can be prepended.
			numVTokens := len(visionTokens)
			visionFlat := make([]float32, numVTokens*m.EmbeddingDim)
			for i, tok := range visionTokens {
				copy(visionFlat[i*m.EmbeddingDim:], tok)
			}
			// queryEmbeddings shape: [batch, seqLen, embDim]
			// We prepend vision tokens to seqLen dimension (batch=1 assumed for vision).
			origData := queryEmbeddings.Data
			combinedData := make([]float32, len(visionFlat)+len(origData))
			copy(combinedData, visionFlat)
			copy(combinedData[len(visionFlat):], origData)
			newSeqLen := numVTokens + queryEmbeddings.Shape[1]
			queryEmbeddings = tensor.NewTensor(
				[]int{queryEmbeddings.Shape[0], newSeqLen, m.EmbeddingDim},
				combinedData,
				false,
			)
		}
	}

	// Encoder forward pass
	// Before calling encoder forward, expose token IDs for observability
	contextVector, err := m.Encoder.Forward(queryEmbeddings)
	if err != nil {
		return nil, nil, fmt.Errorf("MoE encoder forward failed: %w", err)
	}

	// Normalize context vector using learned LayerNorm for maximum stability.
	if m.EncoderNorm != nil {
		contextVector, err = m.EncoderNorm.Forward(contextVector)
		if err != nil {
			return nil, nil, fmt.Errorf("encoder norm forward failed: %w", err)
		}
	}

	// 🔍 Diagnostic: Check for Signal Collapse
	ctxNorm := contextVector.L2Norm()
	if ctxNorm < 1e-8 {
		fmt.Printf("⚠️ [IntentMoE] SIGNAL COLLAPSE DETECTED! Context Strength: %.8f\n", ctxNorm)
	}

	// Decoder forward pass with scheduled sampling & mask
	sentenceLogits, err := m.Decoder.Forward(contextVector, targetTokenIDs, scheduledSamplingProb, inputMask)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder forward failed: %w", err)
	}

	return sentenceLogits, contextVector, nil
}

// SetVisionPatches supplies raw 16×16 luma patches for the *next* Forward call.
// Call this before Forward() when processing an image or video frame batch.
// Pass nil to return to pure-text (NLP-only) mode with zero vision overhead.
func (m *IntentMoE) SetVisionPatches(patches [][]float32) {
	m.visionPatches = patches
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

	// Circuit breaker... (existing logic)
	cvGrad := m.Decoder.contextVector.Grad
	const cvClipThreshold = 5.0 // Tightened for better stability
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

	// 2. Backpropagate through EncoderNorm
	if m.EncoderNorm != nil {
		if err := m.EncoderNorm.Backward(cvGrad); err != nil {
			return fmt.Errorf("encoder norm backward failed: %w", err)
		}
		cvGrad = m.EncoderNorm.Input().Grad
	}

	contextVectorGrad := cvGrad

	// Backpropagate through the encoder
	err := m.Encoder.Backward(contextVectorGrad)
	if err != nil {
		return fmt.Errorf("MoE encoder backward failed: %w", err)
	}

	// 👁️ VISION: If patches were used in this forward pass, propagate gradients
	// back through the VisionEncoder weights so it learns from this batch.
	// For pure-text batches (visionPatches is nil) this is skipped with zero overhead.
	if m.VisionEncoder != nil && len(m.visionPatches) > 0 {
		// Build a dummy gradOut matching the vision token sequence length.
		// In a full integration the encoder input grad would be sliced here;
		// for now we use a zero gradient so only the text path drives the update.
		gradOut := make([][]float32, len(m.visionPatches))
		for i := range gradOut {
			gradOut[i] = make([]float32, m.VisionEncoder.DModel)
		}
		m.VisionEncoder.Backward(gradOut, m.visionPatches, 0.001)
		m.visionPatches = nil // clear after backward
	}

	// 4. Backpropagate through Positional Encoding
	if len(m.Encoder.Inputs()) > 0 {
		gradBeforePos := m.Encoder.Inputs()[0].Grad
		if gradBeforePos != nil {
			if m.EncoderPos != nil {
				if err := m.EncoderPos.Backward(gradBeforePos); err != nil {
					return fmt.Errorf("encoder positional backward failed: %w", err)
				}
				if len(m.EncoderPos.Inputs()) > 0 {
					gradBeforePos = m.EncoderPos.Inputs()[0].Grad
				}
			}

			// 5. Backpropagate through the embedding layer
			if gradBeforePos != nil {
				if err := m.Embedding.Backward(gradBeforePos); err != nil {
					return fmt.Errorf("embedding layer backward failed: %w", err)
				}
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
	if m.EncoderNorm != nil {
		params = append(params, m.EncoderNorm.Parameters()...)
	}
	params = append(params, m.Decoder.Parameters()...)
	// 👁️ VISION: Include the VisionEncoder projection weights so the optimizer
	// trains them alongside the NLP parameters (zero overhead when no vision data).
	if m.VisionEncoder != nil {
		visionWeightTensor := tensor.NewTensor(
			[]int{m.VisionEncoder.PatchDim, m.VisionEncoder.DModel},
			m.VisionEncoder.Weights,
			true, // RequiresGrad
		)
		params = append(params, visionWeightTensor)
	}
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

// GetGateTemperature returns the temperature of the first MoE layer found.
func (m *IntentMoE) GetGateTemperature() float32 {
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		return m.Decoder.OutputMoE.RouterTemperature
	}
	// Fallback to active layers if decoder MoE is not available
	if len(ActiveLayers) > 0 {
		return ActiveLayers[0].RouterTemperature
	}
	return 1.0
}

// GreedySearchDecode performs greedy decoding (temperature=1.0).
// This is a wrapper for backward compatibility.
func (m *IntentMoE) GreedySearchDecode(contextVector *tensor.Tensor, maxLen, sosToken, eosToken int, repetitionPenalty, frequencyPenalty float32, topK int) ([]int, error) {
	return m.GreedySearchDecodeWithTemp(contextVector, maxLen, sosToken, eosToken, 1.0, repetitionPenalty, frequencyPenalty, topK, nil)
}

func (m *IntentMoE) GreedySearchDecodeWithTemp(contextVector *tensor.Tensor, maxLen, sosToken, eosToken int, temperature, repetitionPenalty, frequencyPenalty float32, topK int, suppressedIDs map[int]bool) ([]int, error) {
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
		outputLogits, newHidden, newCell, _, err := m.Decoder.DecodeStep(decoderInputIDs, hiddenState, cellState, contextVector, step)
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

		// 🆕 STRICT: Mask system/format tokens from ever appearing in social responses.
		m.MaskSocialSystemTokens(outputLogits)

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

		// Domain Masking: Suppress technical jargon
		if suppressedIDs != nil {
			for id := range suppressedIDs {
				if id >= 0 && id < len(outputLogits.Data) {
					outputLogits.Data[id] = -1e9
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

	for step := range maxLen {
		outputLogits, newHidden, newCell, _, err := m.Decoder.DecodeStep(decoderInputIDs, hiddenState, cellState, contextVector, step)
		if err != nil {
			return nil, fmt.Errorf("decoder step failed: %w", err)
		}

		hiddenState = newHidden
		cellState = newCell

		// 🆕 STRICT: Mask system/format tokens in social context (SampleDecode).
		m.MaskSocialSystemTokens(outputLogits)

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

	oldEmb := m.Embedding
	newEmb := nn.NewEmbedding(newVocabSize, oldEmb.DimModel)

	// Preserve old ControlTokenIDs
	if oldEmb.ControlTokenIDs != nil {
		newEmb.ControlTokenIDs = oldEmb.ControlTokenIDs
	}

	// Copy old weights
	copy(newEmb.Weight.Data, oldEmb.Weight.Data)

	m.Embedding = newEmb
	m.SanitizeControlTokens()
}

// SaveIntentMoECheckpoint saves the IntentMoE and its metadata to a file with compression.
func SaveIntentMoECheckpoint(ckpt *Checkpoint, path string) error {
	// 🧹 Pre-serialization GC to reduce OOM risk during large model encoding
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create checkpoint directory: %w", err)
	}

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

	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	if GlobalTelemetry != nil {
		GlobalTelemetry.RecordSerializationMetrics("checkpoint", map[string]interface{}{
			"path":        path,
			"alloc_delta": int64(after.TotalAlloc - before.TotalAlloc),
			"heap_delta":  int64(after.HeapAlloc - before.HeapAlloc),
			"heap_in_use": int64(after.HeapInuse),
			"gc_pause_ms": int64(after.PauseTotalNs / 1e6),
			"num_gc":      int64(after.NumGC),
		})
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
	// 🧹 Pre-serialization GC
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create model directory: %w", err)
	}

	tempPath := path + ".tmp"
	file, err := os.Create(tempPath)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}

	// Use gzip for compression (prevents OOM during large model serialization)
	gz := gzip.NewWriter(file)
	writer := bufio.NewWriter(gz)

	encoder := gob.NewEncoder(writer)
	err = encoder.Encode(model)
	if err != nil {
		gz.Close()
		file.Close()
		os.Remove(tempPath)
		return fmt.Errorf("failed to encode model: %w", err)
	}

	writer.Flush()
	gz.Close()
	file.Close()

	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	if GlobalTelemetry != nil {
		GlobalTelemetry.RecordSerializationMetrics("gob_model", map[string]interface{}{
			"path":        path,
			"alloc_delta": int64(after.TotalAlloc - before.TotalAlloc),
			"heap_delta":  int64(after.HeapAlloc - before.HeapAlloc),
			"heap_in_use": int64(after.HeapInuse),
			"gc_pause_ms": int64(after.PauseTotalNs / 1e6),
			"num_gc":      int64(after.NumGC),
		})
	}

	if runtime.GOOS == "windows" {
		_ = os.Remove(path)
	}
	return os.Rename(tempPath, path)
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

// LoadIntentMoEModelWithFallback attempts to load IntentMoE with format detection.
// Tries three paths in order:
//  1. gzip-compressed Checkpoint wrapper (SaveIntentMoECheckpoint format)
//  2. gzip-compressed raw IntentMoE (SaveIntentMoEModelToGOB format — most common)
//  3. raw (uncompressed) gob IntentMoE (legacy format)
func LoadIntentMoEModelWithFallback(filePath string) (*IntentMoE, error) {
	// Check file size first
	fi, err := os.Stat(filePath)
	if err != nil {
		return nil, fmt.Errorf("error checking model file: %w", err)
	}
	if fi.Size() == 0 {
		return nil, fmt.Errorf("model file is empty: %s", filePath)
	}

	// ── Path 1: gzip + Checkpoint wrapper ─────────────────────────────────────
	{
		file, err := os.Open(filePath)
		if err != nil {
			return nil, fmt.Errorf("error opening model file: %w", err)
		}
		gz, gzErr := gzip.NewReader(file)
		if gzErr == nil {
			var ckpt Checkpoint
			decErr := gob.NewDecoder(gz).Decode(&ckpt)
			gz.Close()
			file.Close()
			if decErr == nil && ckpt.Model != nil {
				if ckpt.StepCount > ckpt.Model.StepCount {
					ckpt.Model.StepCount = ckpt.StepCount
				}
				ckpt.Model.RepairArchitecture()
				return ckpt.Model, nil
			}
		} else {
			file.Close()
		}
	}

	// ── Path 2: gzip + raw IntentMoE (SaveIntentMoEModelToGOB format) ─────────
	{
		file, err := os.Open(filePath)
		if err != nil {
			return nil, fmt.Errorf("error opening model file: %w", err)
		}
		gz, gzErr := gzip.NewReader(file)
		if gzErr == nil {
			var model IntentMoE
			decErr := gob.NewDecoder(gz).Decode(&model)
			gz.Close()
			file.Close()
			if decErr == nil {
				model.RepairArchitecture()
				return &model, nil
			}
		} else {
			file.Close()
		}
	}

	// ── Path 3: raw (uncompressed) gob IntentMoE (legacy) ─────────────────────
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening model file: %w", err)
	}
	defer file.Close()
	var loadedModel IntentMoE
	if err := gob.NewDecoder(bufio.NewReader(file)).Decode(&loadedModel); err != nil {
		return nil, fmt.Errorf("failed to load model in all formats (gzip-checkpoint, gzip-model, raw-gob): %w", err)
	}
	loadedModel.RepairArchitecture()
	return &loadedModel, nil
}

// RepairArchitecture ensures the model has all necessary layers for the current version.
// This allows older GOB checkpoints to be loaded and "upgraded" to the stable architecture.
func (m *IntentMoE) RepairArchitecture() {
	if m.EncoderNorm == nil {
		m.EncoderNorm = nn.NewLayerNorm(m.EmbeddingDim)
	}
	if m.EncoderPos == nil {
		m.EncoderPos = nn.NewPositionalEmbedding(128, m.EmbeddingDim)
	}

	if m.Supervisor != nil && m.Supervisor.CartridgeMgr == nil {
		m.Supervisor.CartridgeMgr = NewCartridgeManager()
	}

	// Delegate to encoder
	if m.Encoder != nil {
		m.Encoder.RepairArchitecture()

		// 🧬 AUTO-UPGRADE: Ensure all MoE layers have Grammar Experts (typically 16 experts total if base was 8)
		if h, ok := m.Encoder.(*HybridLLMGNNEncoder); ok && h.LLMEncoder != nil {
			if stack, ok := h.LLMEncoder.(*MoEStack); ok {
				for _, layer := range stack.Layers {
					if layer != nil {
						hasGrammar := false
						for _, ex := range layer.Experts {
							if _, ok := ex.(*GrammarExpert); ok {
								hasGrammar = true
								break
							}
						}
						if !hasGrammar {
							numToAdd := len(layer.Experts)
							log.Printf("🛠️ [MoE] Repairing layer: adding missing Grammar Experts (currently %d experts)...", len(layer.Experts))
							if err := appendGrammarExperts(layer, m.EmbeddingDim, numToAdd); err != nil {
								log.Printf("❌ Failed to append grammar experts: %v", err)
							} else {
								// 🧬 JUMPSTART: Seed the new experts with structural bias if vocab is available
								if m.SentenceVocab != nil {
									for _, ex := range layer.Experts {
										if ge, ok := ex.(*GrammarExpert); ok {
											ge.SeedGrammarBias(m.SentenceVocab.Size(), m.SentenceVocab.TokenToWord)
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// Delegate to decoder
	if m.Decoder != nil {
		m.Decoder.RepairArchitecture()

		// 🧬 AUTO-UPGRADE: Decoder OutputMoE needs Grammar Experts too
		if m.Decoder.OutputMoE != nil {
			layer := m.Decoder.OutputMoE
			hasGrammar := false
			for _, ex := range layer.Experts {
				if _, ok := ex.(*GrammarExpert); ok {
					hasGrammar = true
					break
				}
			}
			if !hasGrammar {
				numToAdd := len(layer.Experts)
				if err := appendGrammarExperts(layer, m.EmbeddingDim, numToAdd); err != nil {
					log.Printf("❌ Failed to append decoder grammar experts: %v", err)
				} else {
					// 🧬 JUMPSTART: Seed the new experts with structural bias if vocab is available
					if m.SentenceVocab != nil {
						for _, ex := range layer.Experts {
							if ge, ok := ex.(*GrammarExpert); ok {
								ge.SeedGrammarBias(m.SentenceVocab.Size(), m.SentenceVocab.TokenToWord)
							}
						}
					}
				}
			}
		}
	}

	// 🧬 REBUILD ACTIVE LAYERS
	// Ensure all MoE layers (including loaded ones) are tracked for Load Balancing
	m.SanitizeControlTokens()
	m.RebuildActiveLayers()
}

// SanitizeControlTokens registers conditional prefix tokens to be scaled down in variance.
func (m *IntentMoE) SanitizeControlTokens() {
	if m.Embedding == nil || m.SentenceVocab == nil {
		return
	}
	if m.Embedding.ControlTokenIDs == nil {
		m.Embedding.ControlTokenIDs = make(map[int]bool)
	}
	specialTokens := []string{"__intent__", "__ques__", "__ans__", "social", "create_webserver", "create_handler", "create_database", "create_page", "create_file", "create_folder", "create_structure", "move_file", "create_object", "stop", "run_webserver", "watch", ":"}
	for _, tok := range specialTokens {
		id := m.SentenceVocab.GetTokenID(tok)
		if id > 0 {
			m.Embedding.ControlTokenIDs[id] = true
		}
	}
}

func (m *IntentMoE) RebuildActiveLayers() {
	// Clear current tracking
	ActiveLayers = nil

	// Collect from encoder
	if m.Encoder != nil {
		layers := m.Encoder.GetMoELayers()
		for _, l := range layers {
			ActiveLayers = append(ActiveLayers, l)
		}
	}

	// Collect from decoder
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		ActiveLayers = append(ActiveLayers, m.Decoder.OutputMoE)
	}

	// Ensure new fields are initialized for hot-loading or old checkpoints
	for _, layer := range ActiveLayers {
		if len(layer.ExpertOutputScale) == 0 {
			numExperts := len(layer.Experts)
			layer.ExpertOutputScale = make([]float32, numExperts)
			for i := range layer.ExpertOutputScale {
				layer.ExpertOutputScale[i] = 1.0
			}
		}
		if layer.StructuralBiasIntensity == 0 {
			layer.StructuralBiasIntensity = 8.0
		}
	}
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
