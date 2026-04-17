package chat

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/moe"
	neuralnn "github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/train"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	expertUtilization = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "moe_expert_usage_total",
			Help: "The total number of tokens processed by each expert.",
		},
		[]string{"layer", "expert_id"},
	)

	tokenLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "moe_token_generation_latency_ms",
			Help:    "Time taken to generate a single token.",
			Buckets: prometheus.LinearBuckets(10, 10, 10), // 10ms to 100ms
		},
	)
)

// Batch holds a pre-tokenized training batch.
type Batch struct {
	Input     *tensor.Tensor // Shape: [BatchSize, MaxInputLen]
	Target    *tensor.Tensor // Shape: [BatchSize, MaxTargetLen]
	Mask      []float32      // To tell the loss function to ignore <pad>
	InputMask *tensor.Tensor // Attention mask (0.0 for real, -1e9 for pad)
}

// TrainingMetric defines a data structure for monitoring training health, compatible with WASM dashboards.
type TrainingMetric struct {
	Step              int     `json:"step"`
	Loss              float32 `json:"loss"`
	LoadBalanceLoss   float32 `json:"lb_loss"`
	LearningRate      float32 `json:"lr"`
	ActiveExperts     []int   `json:"active_experts"`  // IDs of experts used in this batch
	IsCooling         bool    `json:"is_cooling"`      // Flag for the CoolingOptimizer state
	CircuitBreaker    bool    `json:"circuit_breaker"` // True if a shake was triggered
	Temperature       float32 `json:"temperature"`     // Current ThawScheduler temperature
	ThawedExpertCount int     `json:"thawed_count"`    // Number of active clusters/experts
}

// IsStuck checks for "stuttering" or punctuation loops in the generated tokens.
func IsStuck(tokens []string, threshold float32) bool {
	if len(tokens) < 10 {
		return false
	}

	// Check for "stuttering" - e.g., the same token repeating
	repeatCount := 0
	last := tokens[len(tokens)-1]
	for i := len(tokens) - 2; i >= max(0, len(tokens)-10); i-- {
		if tokens[i] == last {
			repeatCount++
		}
	}

	// If more than 60% (or specific threshold) of recent tokens are the same
	return float32(repeatCount) >= threshold
}

func ValidateModelHealth(model *moe.IntentMoE) bool {
	fmt.Println("🔍 Performing Pre-Flight Health Check...")
	isHealthy := true

	for i, param := range model.Parameters() {
		var maxVal float32 = -1e18
		var minVal float32 = 1e18
		nanCount := 0

		for _, v := range param.Data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				nanCount++
			}
			if v > maxVal {
				maxVal = v
			}
			if v < minVal {
				minVal = v
			}
		}

		// Check for NaNs
		if nanCount > 0 {
			fmt.Printf("❌ Param %d: Found %d NaN/Inf values!\n", i, nanCount)
			isHealthy = false
		}

		// Check for Weight Saturation
		if maxVal > 100.0 || minVal < -100.0 {
			fmt.Printf("⚠️  Param %d: High saturation detected (Range: %.2f to %.2f)\n", i, minVal, maxVal)
		}
	}

	if isHealthy {
		fmt.Println("✅ Model weights are within safe numerical bounds.")
	}
	return isHealthy
}

func InspectRouterWeights(model *moe.IntentMoE) {
	fmt.Println("🔬 Inspecting Router Integrity...")
	for i, layer := range moe.ActiveLayers {
		var weightSum float32 = 0.0
		for _, v := range layer.GatingNetwork.Linear.Weights.Data {
			weightSum += float32(math.Abs(float64(v)))
		}

		if weightSum == 0 {
			fmt.Printf("🚨 LAYER %d ALERT: Router weights are all ZEROS! (Inference will pin to E0)\n", i)
		} else {
			fmt.Printf("✅ Layer %d: Router weight magnitude is %.4f\n", i, weightSum)
		}
	}
}

func VerifyModelIntegrity(m *moe.IntentMoE) {
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		w := m.Decoder.OutputMoE.GatingNetwork.Linear.Weights.Data
		var sum float32 = 0.0
		for _, v := range w {
			sum += float32(math.Abs(float64(v)))
		}
		if sum == 0 {
			fmt.Println("🚨 CRITICAL: Decoder Router is empty! Resetting weights...")
			// Initialize with small random values to break the E0 tie
			for i := range w {
				w[i] = float32((rand.Float64() - 0.5) * 0.1)
			}
		}
	}
}

// Old ThawScheduler removed in favor of step-based CosineDecay scheduler in internal/ai/training/chat/thaw_scheduler.go

// findMoELayers extracts all MoE layers from a model for inspection/initialization
func findMoELayers(m *moe.IntentMoE) []*moe.MoELayer {
	if m == nil {
		return nil
	}
	layers := m.Encoder.GetMoELayers()
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		layers = append(layers, m.Decoder.OutputMoE)
	}
	return layers
}

// toFloat32 converts string tokens to float32 IDs for tensor creation
func toFloat32(tokens []string) []float32 {
	result := make([]float32, len(tokens))
	for i, t := range tokens {
		// Map token to a simple hash value
		hash := uint32(0)
		for _, ch := range t {
			hash = hash*31 + uint32(ch)
		}
		result[i] = float32(hash % 10000)
	}
	return result
}

func TrainChat(projectRoot string, rebalanceRequested bool, overfitMode bool, initialLR float32, weightDecay float32, autoHeal bool, maxGradNorm float32, useGPU bool, batchSize int, accumulationSteps int) {
	fmt.Println("--- 🗣️  Training Chat Model ---")

	if useGPU {
		fmt.Println("🚀 Using Global GPU Context for Chat Training...")
	}

	// 1. Load Word2Vec for embeddings
	w2vPath := filepath.Join(projectRoot, "data/models/gob_models/word2vec_model.gob")
	w2v, err := word2vec.LoadModel(w2vPath)
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}
	fmt.Println("✅ Loaded Word2Vec model")

	var chatPairs []struct{ Q, A, Intent string }

	// 2. Read conversing.csv (fallback or additional data)
	chatPath := filepath.Join(projectRoot, "data/training/trainingdata/conversing.csv")
	csvFile, err := os.Open(chatPath)
	if err != nil {
		log.Printf("⚠️  Failed to read conversing.csv: %v. Trying json...", err)
		// Try falling back to json if csv doesn't exist
		jsonPath := filepath.Join(projectRoot, "data/training/trainingdata/conversing.json")
		jsonData, err := os.ReadFile(jsonPath)
		if err != nil {
			log.Fatalf("Critical: Could not find data/training/trainingdata/conversing.csv or conversing.json!")
		}
		var jsonPairs []struct {
			Prompt   string `json:"prompt"`
			Response string `json:"response"`
		}
		if err := json.Unmarshal(jsonData, &jsonPairs); err != nil {
			log.Fatalf("Failed to unmarshal conversing.json: %v", err)
		}
		for _, p := range jsonPairs {
			for k := 0; k < 10; k++ {
				chatPairs = append(chatPairs, struct{ Q, A, Intent string }{p.Prompt, p.Response, "json_fallback"})
			}
		}
		// Skip CSV processing if we loaded JSON
		goto skipCSV
	}
	defer csvFile.Close()

	{
		reader := csv.NewReader(csvFile)
		records, err := reader.ReadAll()
		if err != nil {
			log.Fatalf("Failed to read conversing.csv: %v", err)
		}

		for i, record := range records {
			if i == 0 && strings.Contains(strings.ToLower(record[0]), "intent") {
				continue // Skip header
			}
			if len(record) >= 3 {
				// Intent = record[0], Pattern = record[1], Response = record[2]
				q := strings.Trim(record[1], "\" ")
				a := strings.Trim(record[2], "\" ")
				intent := strings.Trim(record[0], "\" ")
				if q == "" || a == "" {
					continue
				}

				// Train harder on this data by upsampling it 2x (reduced from 10x for speed)
				for k := 0; k < 2; k++ {
					chatPairs = append(chatPairs, struct{ Q, A, Intent string }{q, a, intent})
				}
			}
		}
	}
	log.Printf("📊 Loaded %d training pairs from conversing.csv", len(chatPairs))

skipCSV:

	// 3. Load human_chat.txt for social intent training
	humanChatPath := filepath.Join(projectRoot, "data/training/trainingdata/human_chat.txt")
	if humanChatData, err := os.ReadFile(humanChatPath); err == nil {
		humanChatLines := strings.Split(string(humanChatData), "\n")
		var currentQ, currentA string

		for _, line := range humanChatLines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			// Parse "Human 1: ..." or "Human 2: ..."
			if strings.HasPrefix(line, "Human 1:") {
				if currentQ != "" && currentA != "" {
					// Save the Q-A pair before starting new question
					chatPairs = append(chatPairs, struct{ Q, A, Intent string }{currentQ, currentA, "social"})
				}
				currentQ = strings.TrimPrefix(line, "Human 1:")
				currentQ = strings.TrimSpace(currentQ)
				currentA = ""
			} else if strings.HasPrefix(line, "Human 2:") {
				currentA = strings.TrimPrefix(line, "Human 2:")
				currentA = strings.TrimSpace(currentA)
			}
		}

		// Don't forget the last pair
		if currentQ != "" && currentA != "" {
			chatPairs = append(chatPairs, struct{ Q, A, Intent string }{currentQ, currentA, "social"})
		}

		log.Printf("📊 Loaded social intent pairs from human_chat.txt (total: %d pairs)", len(chatPairs))
	} else {
		log.Printf("⚠️  human_chat.txt not found at %s, skipping social intent data", humanChatPath)
	}

	// --- Pre-compute the final vocabulary size from training data ---
	// This lets us initialize a fresh model at the correct size, avoiding
	// the expensive ResizeOutputLayer call that is the primary OOM source.
	tmpVocab := mainvocab.NewVocabulary()
	tmpVocab.AddToken("<pad>")
	tmpVocab.AddToken("<s>")
	tmpVocab.AddToken("</s>")
	tmpVocab.AddToken("UNK")
	for _, pair := range chatPairs {
		for _, t := range cleanTokenize(pair.Q + " " + pair.A) {
			tmpVocab.AddToken(t)
		}
	}
	precomputedVocabSize := tmpVocab.Size()
	tmpVocab = nil // free immediately
	log.Printf("📐 Pre-computed final vocab size: %d", precomputedVocabSize)

	// Reset ActiveLayers to ensure we track only the current model's layers and prevent leaks
	moe.ActiveLayers = nil

	var intentModel *moe.IntentMoE
	moePath := filepath.Join(projectRoot, "data/models/gob_models/moe_classification_model.gob")
	bestMoePath := filepath.Join(projectRoot, "data/models/gob_models/moe_classification_model_best.gob")

	if _, err := os.Stat(moePath); err == nil {
		if loaded, err := moe.LoadIntentMoEModelFromGOB(moePath); err == nil {
			intentModel = loaded
			log.Printf("✅ Loaded existing MoE model from %s", moePath)

			// Re-register layers to ActiveLayers after loading
			moe.ActiveLayers = findMoELayers(intentModel)
			if len(moe.ActiveLayers) > 0 {
				log.Printf("📡 Re-registered %d MoE layers from loaded model", len(moe.ActiveLayers))
			}

			InspectRouterWeights(intentModel)
			VerifyModelIntegrity(intentModel)

			// Architecture compatibility check: if dims don't match new config, force fresh start.
			if intentModel.EmbeddingDim != 512 {
				log.Printf("⚠️  Found %dd model — expected 512d. Forcing fresh 512d start for new hardware config.", intentModel.EmbeddingDim)
				intentModel = nil
			}
		}
	}

	if intentModel != nil && useGPU {
		fmt.Println("🚀 Moving loaded model to GPU...")
		intentModel.ToGPU()
	}

	if intentModel == nil {
		hwInfo := "i5-12400F + 16GB RAM"
		if useGPU {
			hwInfo += " + GPU (Paragon/WebGPU)"
		}
		log.Printf("🚀 Initializing 512d MoE Transformer (8 Experts, 4-Layer Encoder, 4-Layer Decoder) for %s", hwInfo)
		// Use the pre-computed final vocab size so we never need to call
		// ResizeOutputLayer for a fresh model — this is the main OOM fix.
		freshVocab := precomputedVocabSize
		if freshVocab < 100 {
			freshVocab = 8000 // sanity floor if pre-computation didn't run
		}
		log.Printf("🔢 Initializing decoder with final vocab size: %d", freshVocab)
		// 512d: 8 experts + 4-layer Encoder + 4-layer Decoder
		// Reduced from 768d/16-experts which was consuming 14GB+ anon-RSS and getting OOM-killed.
		const modelDim = 512
		const numModelExperts = 8
		var err error
		intentModel, err = moe.NewHybridIntentMoE(
			freshVocab,      // vocabSize
			modelDim,        // embeddingDim
			numModelExperts, // numExperts
			modelDim,        // parentVocabSize
			modelDim,        // childVocabSize
			freshVocab,      // sentenceVocabSize
			8,               // maxAttentionHeads (must divide modelDim evenly)
			nil,             // word2vecModel
		)
		if err != nil {
			log.Fatalf("Failed to create MoE model: %v", err)
		}

		// 4-Layer LSTM decoder with 8 output experts per layer.
		intentModel.Decoder, _ = moe.NewRNNDecoder(modelDim, freshVocab, modelDim, 8, 4, 0.1, numModelExperts)

		if useGPU {
			fmt.Println("🚀 Moving fresh model to GPU...")
			intentModel.ToGPU()
		}

		// Phase 1: Robust initialization
		log.Println("🛠️ Phase 1: Robust init (He for experts, Orthogonal/High-Scale for router)...")
		allLayers := findMoELayers(intentModel)
		for _, layer := range allLayers {
			// Experts -> He Normal (variance-scaled)
			for _, expert := range layer.Experts {
				for _, p := range expert.Parameters() {
					InitializeHeNormal(p)
				}
			}
			// Router -> Sharp initial gating with anti-monopoly nudge
			InitializeRouterGating(layer.GatingNetwork.Linear.Weights, layer.GatingNetwork.Linear.Biases)
		}

		// Phase 2: LSTM Specialized Init
		log.Println("🛠️ Phase 2: Orthogonal init for LSTM weights + Forget-gate bias trick...")
		if intentModel.Decoder != nil && intentModel.Decoder.LSTM != nil {
			for _, layer := range intentModel.Decoder.LSTM.Cells {
				for _, cell := range layer {
					InitializeOrthogonal(cell.Wf, 0.8)
					InitializeOrthogonal(cell.Wi, 0.8)
					InitializeOrthogonal(cell.Wc, 0.8)
					InitializeOrthogonal(cell.Wo, 0.8)
					for i := range cell.Bi.Data {
						cell.Bi.Data[i] = 0
					}
					for i := range cell.Bc.Data {
						cell.Bc.Data[i] = 0
					}
					for i := range cell.Bo.Data {
						cell.Bo.Data[i] = 0
					}
					for i := range cell.Bf.Data {
						cell.Bf.Data[i] = 1.0
					}
				}
			}
		}

		// Signal Boosts removed for numerical stability (preventing 37M gradient norm)
		log.Println("⚡ Phase 3 & 4: Manual Signal Boosting disabled for stability.")
	}

	// Ensure ActiveLayers is synced with the model we are actually using
	moe.ActiveLayers = findMoELayers(intentModel)
	log.Printf("📡 Total Active MoE Layers: %d", len(moe.ActiveLayers))

	// 2. Health Check
	if !ValidateModelHealth(intentModel) {
		log.Println("⚠️ Model health check failed. Attempting to recover with rebalance...")
		rebalanceRequested = true
	}

	if rebalanceRequested {
		log.Println("⚖️ Manual Rebalance Triggered: Normalizing Expert weight distributions...")
		// Assuming we want to rebalance all MoE layers found
		for _, layer := range findMoELayers(intentModel) {
			layer.RebalanceExperts()
		}
	}

	// Adjust MoE settings for training
	for _, layer := range moe.ActiveLayers {
		layer.CapacityFactor = 1.5       // Increased from 1.25
		layer.LoadBalancingWeight = 0.05 // Reduced to let CrossEntropy lead more
		layer.RouterTemperature = 1.0    // Normal temperature
		layer.ExpertDropoutRate = 0.1    // Reduced dropout to prevent UNK collapse
		layer.SetMode(true)              // Enable training mode (noise)
	}
	log.Println("🔧 Adjusted MoE: Capacity=1.5, LBWeight=0.15, Temp=1.0, Dropout=0.1")

	// Initial Router Shake: If starting fresh or rebalancing, increase temperature briefly
	if rebalanceRequested {
		for _, layer := range findMoELayers(intentModel) {
			layer.RouterTemperature = 2.5 // "Shake" the router
		}
		log.Println("🔥 Initial Router Temperature set to 2.5 for exploration")
	}

	// Try to load vocab if nil
	vocabPath := filepath.Join(projectRoot, "data/models/gob_models/seq2seq_output_vocab.gob")
	if intentModel.SentenceVocab == nil {
		if v, err := mainvocab.LoadVocabulary(vocabPath); err == nil {
			intentModel.SentenceVocab = v
			log.Printf("✅ Loaded existing vocabulary from %s", vocabPath)
		}
	}

	// Ensure all parameters have RequiresGrad = true to prevent nil gradients during backward pass
	if intentModel != nil {
		for _, param := range intentModel.Parameters() {
			param.RequiresGrad = true
		}
	}

	if len(chatPairs) == 0 {
		log.Println("No chat pairs found.")
		return
	}

	// Expand Word2Vec vocabulary with missing words from chat data
	maxID := 0
	for _, id := range w2v.Vocabulary {
		if id > maxID {
			maxID = id
		}
	}
	if w2v.WordVectorsF32 == nil {
		w2v.WordVectorsF32 = make(map[int][]float32)
	}

	addedCount := 0
	for _, pair := range chatPairs {
		tokens := cleanTokenize(pair.Q)
		for _, t := range tokens {
			if _, ok := w2v.Vocabulary[t]; !ok {
				maxID++
				w2v.Vocabulary[t] = maxID
				// Initialize random vector
				vec := make([]float32, w2v.VectorSize)
				limit := float32(math.Sqrt(6.0 / float64(w2v.VectorSize)))
				for i := range vec {
					vec[i] = (rand.Float32() * 2 * limit) - limit
				}
				w2v.WordVectorsF32[maxID] = vec
				addedCount++
			}
		}
	}
	w2v.VocabSize = len(w2v.Vocabulary)
	if addedCount > 0 {
		log.Printf("✨ Expanded Word2Vec vocab with %d new words from training data. Total: %d", addedCount, w2v.VocabSize)
	}

	// --- ENSURE SENTENCEVOCAB EXISTS AND POPULATE IT FIRST ---
	if intentModel.SentenceVocab == nil {
		intentModel.SentenceVocab = mainvocab.NewVocabulary()
		intentModel.SentenceVocab.AddToken("<pad>")
		intentModel.SentenceVocab.PaddingTokenID = intentModel.SentenceVocab.GetTokenID("<pad>")
		intentModel.SentenceVocab.AddToken("<s>")
		intentModel.SentenceVocab.AddToken("</s>")
		intentModel.SentenceVocab.AddToken("UNK")
		intentModel.SentenceVocab.BosID = intentModel.SentenceVocab.GetTokenID("<s>")
		intentModel.SentenceVocab.EosID = intentModel.SentenceVocab.GetTokenID("</s>")
	}

	// Add EVERY word from the training data to the SentenceVocab
	for _, pair := range chatPairs {
		for _, t := range cleanTokenize(pair.Q + " " + pair.A) {
			intentModel.SentenceVocab.AddToken(t)
		}
	}
	log.Printf("✅ Final SentenceVocab Size: %d", intentModel.SentenceVocab.Size())

	// Ensure UNK is in vocab and get its ID
	unkID := intentModel.SentenceVocab.GetTokenID("UNK")
	if unkID == -1 {
		intentModel.SentenceVocab.AddToken("UNK")
		unkID = intentModel.SentenceVocab.GetTokenID("UNK")
	}

	// Resize Encoder Embedding if Vocabulary has grown
	currentVocabSize := intentModel.SentenceVocab.Size()
	if intentModel.Embedding.VocabSize != currentVocabSize {
		newEmb := neuralnn.NewEmbedding(currentVocabSize, intentModel.EmbeddingDim)

		// Fill with Word2Vec weights where possible
		for i := 0; i < currentVocabSize; i++ {
			word := intentModel.SentenceVocab.GetWord(i)
			if id, ok := w2v.Vocabulary[word]; ok {
				vec := w2v.WordVectorsF32[id]
				if len(vec) == intentModel.EmbeddingDim {
					copy(newEmb.Weight.Data[i*intentModel.EmbeddingDim:], vec)
				} else {
					// Handle dimension mismatch gracefully
					copyLen := min(len(vec), intentModel.EmbeddingDim)
					copy(newEmb.Weight.Data[i*intentModel.EmbeddingDim:], vec[:copyLen])
				}
			}
		}
		intentModel.Embedding = newEmb
	}

	// Move Word2Vec coverage check here before we free it
	hit := 0
	miss := 0
	for _, pair := range chatPairs {
		tokens := cleanTokenize(pair.Q)
		for _, t := range tokens {
			if _, ok := w2v.Vocabulary[t]; ok {
				hit++
			} else {
				miss++
			}
		}
	}
	log.Printf("Word2Vec Coverage: %d hits, %d misses (%.2f%%)", hit, miss, float64(hit)/float64(hit+miss)*100)

	// Free Word2Vec model from memory — it's no longer needed for weights
	log.Printf("🗑️  Freeing Word2Vec vectors from memory (%d vectors)...", w2v.VocabSize)
	w2v.WordVectors = nil
	// We keep w2v.Vocabulary if needed for other things, but most of them should move to SentenceVocab
	runtime.GC()
	debug.FreeOSMemory()
	log.Println("✅ Word2Vec heavy vectors freed.")

	// --- [Balanced Mixing Strategy] ---
	// Separate into Help (Technical) and Social/General
	var helpPairs, socialPairs []struct{ Q, A, Intent string }
	for _, p := range chatPairs {
		isHelp := strings.HasPrefix(p.Intent, "go_") ||
			strings.HasPrefix(p.Intent, "ml_") ||
			strings.HasPrefix(p.Intent, "moe_") ||
			strings.HasPrefix(p.Intent, "nlp_") ||
			strings.HasPrefix(p.Intent, "math_") ||
			strings.HasPrefix(p.Intent, "system_") ||
			strings.HasPrefix(p.Intent, "gollemer_")
		if isHelp {
			helpPairs = append(helpPairs, p)
		} else {
			socialPairs = append(socialPairs, p)
		}
	}

	log.Printf("⚖️ Data Distribution: Help=%d, Social=%d", len(helpPairs), len(socialPairs))

	// Create balanced set (50/50 mix)
	balancedPairs := make([]struct{ Q, A, Intent string }, 0, len(chatPairs))
	maxLen := max(len(helpPairs), len(socialPairs))
	for i := 0; i < maxLen; i++ {
		if i < len(helpPairs) {
			balancedPairs = append(balancedPairs, helpPairs[i])
		}
		if i < len(socialPairs) {
			balancedPairs = append(balancedPairs, socialPairs[i])
		}
	}
	chatPairs = balancedPairs
	// ------------------------------------

	// Shuffle and Split
	rand.Shuffle(len(chatPairs), func(i, j int) { chatPairs[i], chatPairs[j] = chatPairs[j], chatPairs[i] })
	splitIdx := int(float64(len(chatPairs)) * 0.9)
	trainPairs := chatPairs[:splitIdx]
	valPairs := chatPairs[splitIdx:]

	fmt.Printf("Data Split Pre-Limit (Balanced): %d Training, %d Validation\n", len(trainPairs), len(valPairs))

	// Word2Vec coverage already reported above

	// Resize Decoder if Vocabulary has grown (OR if architecture changed: total dim is now 1536)
	currentVocabSize = intentModel.SentenceVocab.Size()
	// Always force a resize if LayerNorm is the old size (e.g., 512) to handle the architecture upgrade
	needsResize := intentModel.Decoder.Embedding.VocabSize != currentVocabSize
	if intentModel.Decoder.LayerNorm != nil && intentModel.Decoder.LayerNorm.NormalizedShape != (intentModel.Decoder.LSTM.HiddenSize+intentModel.Decoder.Embedding.DimModel) {
		needsResize = true
		log.Printf("Forcing decoder resize due to architecture upgrade (LayerNorm %d -> %d)",
			intentModel.Decoder.LayerNorm.NormalizedShape,
			intentModel.Decoder.LSTM.HiddenSize+intentModel.Decoder.Embedding.DimModel)
	}

	if needsResize {
		log.Printf("Resizing Decoder from Vocab %d to %d", intentModel.Decoder.Embedding.VocabSize, currentVocabSize)
		intentModel.Decoder.ResizeOutputLayer(currentVocabSize)
		intentModel.SentenceVocabSize = currentVocabSize
	}

	// Clear any stale state from the loaded model
	DetachModel(intentModel)

	// ═══════════════════════════════════════════════════════════════
	// PHASE 0: MLM Pre-Training (Grammar Learning)
	// Teaches the encoder and embeddings word co-occurrence patterns
	// through fill-in-the-blank prediction before seq2seq training.
	// ═══════════════════════════════════════════════════════════════
	if !overfitMode {
		mlmSentences := ExtractMLMSentences(chatPairs)
		if len(mlmSentences) > 0 {
			// Add [MASK] token to vocab before MLM (this may grow vocab by 1)
			if intentModel.SentenceVocab.GetTokenID(MaskToken) == -1 {
				intentModel.SentenceVocab.AddToken(MaskToken)
				newVocabSize := intentModel.SentenceVocab.Size()
				if newVocabSize != intentModel.Decoder.Embedding.VocabSize {
					log.Printf("🔄 Resizing Decoder for [MASK] token: %d → %d", intentModel.Decoder.Embedding.VocabSize, newVocabSize)
					intentModel.Decoder.ResizeOutputLayer(newVocabSize)
					intentModel.SentenceVocabSize = newVocabSize
				}
			}

			mlmLR := initialLR * 2.0 // MLM can use a higher LR since it's a simpler task
			if mlmLR > 0.01 {
				mlmLR = 0.01
			}
			mlmEpochs := 5

			if err := RunMLMPreTraining(
				intentModel,
				mlmSentences,
				mlmEpochs,
				batchSize,
				mlmLR,
				maxGradNorm,
				useGPU,
				moePath, // Pass savePath
			); err != nil {
				log.Printf("⚠️ MLM Pre-Training failed (non-fatal): %v", err)
			}

			// Clear state and GC after MLM phase
			DetachModel(intentModel)
			runtime.GC()
			debug.FreeOSMemory()
		}
	}

	// Curriculum sort
	sort.Slice(chatPairs, func(i, j int) bool {
		return len(cleanTokenize(chatPairs[i].A)) < len(cleanTokenize(chatPairs[j].A))
	})
	log.Println("🎓 Curriculum active: Training starts with shortest sentences.")

	// Split data
	trainCount := int(float64(len(chatPairs)) * 0.95)
	trainPairs = chatPairs[:trainCount]
	valPairs = chatPairs[trainCount:]

	if overfitMode {
		trainPairs = trainPairs[:min(1, len(trainPairs))]
		log.Println("🎯 OVERFIT MODE ACTIVE: Training on single example only.")
	}

	log.Printf("Final Train Set Size: %d", len(trainPairs))

	// Use Iterator pattern to save memory and speed up training
	iterator := NewChatDataIterator(trainPairs, intentModel.SentenceVocab, unkID)
	if overfitMode {
		iterator.MaxLen = 768 // Match high-capacity architecture
	}

	// Extra cleanup
	w2v.Vocabulary = nil
	runtime.GC()
	debug.FreeOSMemory()

	// Training Loop
	epochs := 60

	// Optimizer initialization (Wrapped with Cooling Safety).
	// ClipThreshold is set to 0 (disabled) inside Adam because we perform a single,
	// authoritative global-L2 clip via train.ClipParamGrads before calling Step().
	// Having two clips at different thresholds silently overrides maxGradNorm.
	baseOptimizer := neuralnn.NewOptimizer(intentModel.Parameters(), initialLR, 0)
	optimizer := &neuralnn.CoolingOptimizer{
		Base: baseOptimizer,
	}

	// Learning rate settings
	var peakLR float32 = initialLR

	// OneCycle Scheduler
	scheduler := &OneCycle{
		MaxLR:      peakLR,
		MinLR:      peakLR * 0.01,
		TotalSteps: epochs * (len(trainPairs) / batchSize),
	}

	// Early stopping and metrics state
	patienceLimit := 10
	patienceCounter := 0
	globalStep := 0
	var epochLBLoss float32 = 0.0
	var bestPPL float32 = math.MaxFloat32

	if adam, ok := optimizer.Base.(*neuralnn.Adam); ok {
		adam.Lambda = weightDecay
	}

	// 🌡️ Annealer Setup as requested
	annealer := train.Annealer{
		StartTemp: 1.0,
		MinTemp:   0.1,
		Decay:     0.95,
		WarmUp:    4, // Match existing warmup logic
	}

	// Plateau & Stagnancy Tracker Setup
	pConfig := train.PlateauConfig{
		Patience:  3,
		Cooldown:  2,
		Factor:    0.5,
		TempDecay: 0.95,
		MinLR:     1e-6,
	}
	pState := train.PlateauState{BestPPL: 1000.0} // Initial PPL estimate
	stTracker := &train.StagnancyTracker{Epsilon: 1e-5}

	trainer := &moe.Trainer{CollapseCount: 0}
	// ExpertMonitor: session-level dispatch counter reset every epoch.
	// Tracks which experts actually receive tokens and computes load imbalance.
	numExperts := len(moe.ActiveLayers[0].Experts)
	epochMonitor := moe.NewExpertMonitor(numExperts)
	startTime := time.Now()
	var totalTokens int64
	var totalDuration time.Duration
	var learningRate float32
	var currentEpochTemp float32 = 1.0
	profile := neuralnn.GetProfile("standard") // Or create one from flags

	// Cross-Entropy Weights setup
	lossWeights := make([]float32, intentModel.SentenceVocab.Size())
	for i := range lossWeights {
		lossWeights[i] = 1.0
	}
	lossWeights[unkID] = 0.01
	lossWeights[intentModel.SentenceVocab.PaddingTokenID] = 0.0
	lossWeights[intentModel.SentenceVocab.EosID] = 0.1 // Discourage silence/premature EOS

	// Reduce-on-Plateau LR scheduler
	lrScheduler := &moe.LRScheduler{
		CurrentLR:   peakLR,
		DecayFactor: 0.5,
		Patience:    3,
		MinLR:       1e-7,
	}

	// --- [Curriculum & Data Integrity] ---
	type Curriculum struct {
		MaxSequenceLen      int
		MaxSequenceLenLimit int // Hard cap to prevent OOM
		MinPPLThreshold     float32
		GrowthFactor        int
	}
	curriculum := Curriculum{
		MaxSequenceLen:      64,
		MaxSequenceLenLimit: 128, // Cap at 128 tokens for 8GB VRAM safety
		MinPPLThreshold:     500,
		GrowthFactor:        5,
	}

	inspectData := func(batch *Batch) {
		if globalStep%100 != 0 {
			return
		}
		fmt.Println("🔍 [Data Integrity Check]")
		for i := 0; i < min(2, batch.Input.Shape[0]); i++ {
			var sumSq float32 = 0.0
			n := 1
			if len(batch.Input.Shape) > 1 {
				n = batch.Input.Shape[1]
			}
			for d := 0; d < min(32, n); d++ {
				idx := i*n + d
				if idx < len(batch.Input.Data) {
					v := batch.Input.Data[idx]
					sumSq += v * v
				}
			}
			norm := float32(math.Sqrt(float64(sumSq)))
			if norm < 0.1 {
				fmt.Printf("⚠️ Sequence %d: Potential SIGNAL LOSS (Token IDs near zero)\n", i)
			}
		}
	}
	// --------------------------------------

	fmt.Printf("Training on %d pairs for %d epochs (patience=%d)...\n", len(chatPairs), epochs, patienceLimit)

	// New Step-based ThawScheduler: manages which experts are frozen per step based on Cosine Decay.
	// FIX: Lowered LayerThresholds significantly — the model is at Step 20k / Epoch 6 and only
	// had 2/8 experts active. Old thresholds (0.85, 0.60, 0.35, 0.15) were too conservative.
	// With a StartTemp=1.0 -> MinTemp=0.1 cosine decay, temperature is ~0.97 at Step 20k,
	// which means the model is still very early in its thaw arc. The new thresholds ensure
	// all 8 experts are active by ~30% of total training steps.
	thawScheduler := &ThawScheduler{
		MaxSteps:  epochs * (len(trainPairs) / batchSize),
		StartTemp: 2.5,
		MinTemp:   1.2,
		LayerThresholds: []float32{
			2.4, // Thaw Cluster 0 (2 experts) immediately
			2.0, // Thaw Cluster 1 (4 experts)
			1.7, // Thaw Cluster 2 (6 experts)
			1.4, // Thaw Cluster 3 (All 8 experts)
		},
	}

	// TrainingLogger: CSV log for long-term trend analysis.
	logPath := filepath.Join(projectRoot, "logs/training_log.csv")
	trainingLogger, logErr := NewTrainingLogger(logPath)
	if logErr != nil {
		log.Printf("⚠️  Could not create training logger: %v", logErr)
	} else {
		log.Printf("📊 Training CSV log: %s", logPath)
		defer trainingLogger.Close()
	}

	// Periodic checkpoints directory (kept separate from best-model checkpoints)
	checkpointDir := filepath.Join(projectRoot, "data/models/checkpoints")
	if err := os.MkdirAll(checkpointDir, 0755); err != nil {
		log.Printf("⚠️  Warning: Could not create checkpoint directory: %v", err)
	}

	// 💾 INITIAL SAVE: Save the model immediately before training starts (Step 0)
	log.Printf("💾 [CHECKPOINT] Initial save at Step 0...")
	initialCkpt := &moe.Checkpoint{
		Model: intentModel, StepCount: 0, LastProfile: profile,
		Version: "gollemer-chat-v1.2-initial",
	}
	moe.SaveIntentMoECheckpoint(initialCkpt, filepath.Join(checkpointDir, "initial_0.gob"))

	for epoch := 0; epoch < epochs; epoch++ {
		epochStartTime := time.Now()
		var currentFrozenSet map[int]bool

		// (Cosine Decay ThawScheduler now updates per step instead of per epoch)

		// Curriculum shuffle logic
		if epoch > 2 {
			rand.Shuffle(len(trainPairs), func(i, j int) {
				trainPairs[i], trainPairs[j] = trainPairs[j], trainPairs[i]
			})
			log.Println("🔄 Shuffled training data for this epoch")
		}

		// Force diverse routing for the first 4 epochs to break out of mode collapse
		currentEpochTemp = float32(annealer.GetTemp(epoch))
		intentModel.SetGateTemperature(currentEpochTemp)
		log.Printf("🌡️ Epoch %d | Temperature: %.4f", epoch, currentEpochTemp)

		// (LB weight decay removed — LB weight is now small enough that it doesn't need decay)
		iterator.Reset()
		var totalLoss float32 = 0.0
		batches := 0
		// Reset utilization for each layer and the session monitor
		for _, l := range moe.ActiveLayers {
			l.ResetUtilizationStats()
		}
		epochMonitor.Reset() // Start fresh dispatch counts for this epoch

		iterator.MaxLen = curriculum.MaxSequenceLen
		if overfitMode {
			iterator.MaxLen = 512 // Don't filter the single sample we are trying to overfit!
		}

		// Prefetch tokenization: start a background goroutine that pre-produces
		// Batch structs into a buffered channel while the main goroutine
		// is busy with forward/backward. Buffer=64 keeps the main loop fed.
		// Prefetch tokenization
		prefetchCh := make(chan *Batch, 2)
		go func() {
			for iterator.HasNext() {
				prefetchCh <- iterator.NextBatch(batchSize)
			}
			close(prefetchCh)
		}()

		// Note: we use the maxGradNorm passed as an argument to TrainChat
		// so the linuxtrain.sh flags are respected.
		optimizer.ZeroGrad() // Initial zero out for accumulation

		for batch := range prefetchCh {
			if batch == nil || batch.Input == nil {
				continue
			}

			// 🌡️ Step-based Thaw Prediction
			currentTemp, thawedCount := thawScheduler.Next()
			// Assume thawedCount (1-4) controls expert clusters (2 experts each)
			// At thawedCount=0, at least 2 experts (E0, E1) are thawed for "Exploration"
			numExpertsToThaw := (thawedCount + 1) * 4 // 4 experts per cluster for 16-expert model
			if numExpertsToThaw > 16 {
				numExpertsToThaw = 16
			}

			frozenExperts := []int{}
			for i := numExpertsToThaw; i < 16; i++ {
				frozenExperts = append(frozenExperts, i)
			}

			frozenSet := make(map[int]bool, len(frozenExperts))
			for _, id := range frozenExperts {
				frozenSet[id] = true
			}
			currentFrozenSet = frozenSet

			if globalStep%100 == 0 {
				log.Printf("🌡️ Step %d: Temp=%.4f | Thawed Experts: %d/16", globalStep, currentTemp, numExpertsToThaw)
			}

			// optimizer.ZeroGrad() // Removed for accumulation
			inspectData(batch)
			if overfitMode && globalStep%10 == 0 {
				log.Printf("🎯 [Overfit] Step %d starting...", globalStep)
			}

			// Memory Management: GC occasionally; FreeOSMemory removed from hot path
			// (FreeOSMemory was returning heap to OS every 50 steps, causing page-fault storms)
			if globalStep%500 == 0 {
				runtime.GC()
			}

			// 🛡️ ENCODER WEIGHT DAMPENING: Every 100 steps, clip any encoder
			// parameter whose L2 norm exceeds 50 back to norm 10.
			// Moved from every 10 steps to every 100 — gradient clipping already
			// handles per-step stability; this is just a periodic sanity check.
			if globalStep%100 == 0 {
				for _, p := range intentModel.Encoder.Parameters() {
					var norm float32 = 0.0
					for _, v := range p.Data {
						norm += v * v
					}
					norm = float32(math.Sqrt(float64(norm)))
					if norm > 50.0 {
						scale := float32(10.0 / float64(norm))
						for i := range p.Data {
							p.Data[i] *= scale
						}
					}
				}
			}

			inputTensor := batch.Input
			targetTensor := batch.Target

			if useGPU {
				inputTensor.ToGPU()
				targetTensor.ToGPU()
				if batch.InputMask != nil {
					batch.InputMask.ToGPU()
				}
			}

			if targetTensor.Shape[1] < 2 {
				continue
			}

			var lr float32
			if overfitMode {
				lr = peakLR
			} else {
				lr = scheduler.GetNextLR()
			}
			optimizer.SetLearningRate(lr)
			learningRate = lr

			// 🛑 CIRCUIT BREAKER: Every 500 Batches
			isCircuitBreakerTriggered := false
			var check []string
			if globalStep%500 == 0 && globalStep > 0 {
				fmt.Println("\n🛑 Running Circuit Breaker Check...")
				// Critical Memory Safety: Generation is memory-hungry; clear existing garbage first.
				runtime.GC()
				debug.FreeOSMemory()

				check = GenerateTokens(intentModel, "how are you", 10, useGPU)

				// Reset model states populated during generation to free memory immediately.
				intentModel.ClearState()
				runtime.GC()
				debug.FreeOSMemory()

				// Use new IsStuck utility with 6-token repeat threshold
				if IsStuck(check, 6.0) {
					isCircuitBreakerTriggered = true
					fmt.Println("🚨 Punctuation/Stutter Loop Detected! Shaking experts and cooling...")

					// 1. Shake stagnant experts (intensity 0.05, scaled by current loss plateau ideally)
					for _, layer := range moe.ActiveLayers {
						layer.ShakeExperts(0.05, globalStep/1000+1)
						layer.RouterTemperature = 2.0 // Surge temperature to force exploration
					}

					// 2. Cooling Trigger: 250 steps, 20% of current LR
					optimizer.Trigger(250, 0.2)

					fmt.Printf("❄️ System Cooling Initiated at Step %d\n", globalStep)
				} else {
					fmt.Println("✅ Diversity Check Passed.")
					// Cool down the temperature slowly to normal if it was surging
					for _, layer := range moe.ActiveLayers {
						if layer.RouterTemperature > 0.7 {
							layer.RouterTemperature *= 0.95
						}
					}
				}
			}

			// Forward
			// Teacher Forcing Schedule:
			var samplingProb float32
			if epoch >= 1 {
				// Start sampling even earlier
				samplingProb = float32(math.Min(0.5, float64(epoch)*0.05))
			}
			logits, _, err := intentModel.Forward(samplingProb, inputTensor, targetTensor, batch.InputMask)
			if err != nil {
				log.Printf("Forward error: %v", err)
				continue
			}

			// Label Smoothing Schedule: 0.0 for first 5 epochs, then 0.1
			labelSmoothing := float32(0.1)
			if epoch < 5 {
				labelSmoothing = 0.0
			}

			// Loss
			var batchLoss float32 = 0.0
			var grads []*tensor.Tensor

			if len(logits) == 1 && len(logits[0].Shape) == 3 {
				// Vectorized 3D loss
				l := logits[0]
				// Target for loss is the sequence shifted by 1 (ignoring BOS at index 0)
				currentBatchSize := targetTensor.Shape[0]
				seqLen := targetTensor.Shape[1]
				targetSeqLen := seqLen - 1
				targets := make([]int, currentBatchSize*targetSeqLen)
				for b := 0; b < currentBatchSize; b++ {
					for t := 0; t < targetSeqLen; t++ {
						targets[b*targetSeqLen+t] = int(targetTensor.Data[b*seqLen+t+1])
					}
				}

				// Use the weighted version with the pre-defined weight slice
				loss, grad := WeightedCrossEntropy(l.ToCPU(), targets, lossWeights, labelSmoothing)
				if grad == nil {
					grad = tensor.NewTensor(l.Shape, make([]float32, len(l.Data)), false)
				}
				batchLoss = loss
				grads = []*tensor.Tensor{grad}
			} else {
				// Sequence of logits (Step-by-step path used for scheduled sampling)
				grads = make([]*tensor.Tensor, len(logits))
				currentBatchSize := targetTensor.Shape[0]
				seqLen := targetTensor.Shape[1]
				var stepLossTotal float32 = 0.0
				for t, logit := range logits {
					// Target for this step is AIDs[t+1]
					targets := make([]int, currentBatchSize)
					for b := 0; b < currentBatchSize; b++ {
						targets[b] = int(targetTensor.Data[b*seqLen+t+1])
					}
					l, g := WeightedCrossEntropy(logit.ToCPU(), targets, lossWeights, labelSmoothing)
					if g == nil {
						g = tensor.NewTensor(logit.Shape, make([]float32, len(logit.Data)), false)
					}
					stepLossTotal += l
					grads[t] = g
				}
				// Normalize step-by-step path by volume (consistent with vectorized path)
				div := float32(len(logits))
				batchLoss = stepLossTotal / div // Removed 2x boost — it was amplifying gradients
				for t := range grads {
					for i := range grads[t].Data {
						grads[t].Data[i] /= div // Propagate the normalized loss to gradients
					}
				}
			}

			// Check for NaN/Inf loss immediately
			if math.IsNaN(float64(batchLoss)) || math.IsInf(float64(batchLoss), 0) {
				log.Printf("⚠️ Batch %d loss is NaN/Inf. Skipping batch to prevent model corruption.", batches)
				continue
			}

			// Per-step loss log so training progress is visible
			if globalStep%10 == 0 {
				log.Printf("📈 Step %d | Loss: %.4f | LR: %.6f", globalStep, batchLoss, learningRate)
			}

			if overfitMode && globalStep%10 == 0 {
				log.Printf("🎯 [Overfit] Step %d | Final Loss: %.6f", globalStep, batchLoss)
			}

			// 7. Calculate MoE Stats for auxiliary loss and stability
			currentBatchLayers := intentModel.Encoder.GetMoELayers()
			if intentModel.Decoder.OutputMoE != nil {
				currentBatchLayers = append(currentBatchLayers, intentModel.Decoder.OutputMoE)
			}

			var lambda float32 = 0.25
			var batchZLoss float32
			var lbLoss float32
			for _, layer := range currentBatchLayers {
				batchZLoss += layer.RouterZLoss
				if layer.GateOutputs != nil {
					lbLoss += moe.CalculateImportanceLossTensor(layer.GateOutputs)
				}
			}
			if len(currentBatchLayers) > 0 {
				lbLoss /= float32(len(currentBatchLayers))
			}

			// Final total loss used for gradients and logging
			totalGradLoss := batchLoss + (lambda * lbLoss) + batchZLoss
			batchLoss = totalGradLoss
			epochLBLoss += (lambda * lbLoss)

			// Backward pass with Gradient Accumulation
			func() {
				defer func() {
					if r := recover(); r != nil {
						log.Printf("⚠️ Recovered from panic in Backward pass (batch skipped): %v", r)
					}
				}()

				if err := intentModel.Backward(grads...); err != nil {
					log.Printf("Backward failed: %v", err)
				} else {
					// 1. Track Expert Performance (Every Batch)
					for layerIdx, layer := range moe.ActiveLayers {
						selected := layer.GetSelectedExperts() // [TokenIdx][K]
						for _, tokensExperts := range selected {
							for _, expertID := range tokensExperts {
								intentModel.TrackExpertPerformance(layerIdx, expertID, batchLoss)
							}
						}
					}

					// 2. Zero out gradients for frozen experts (Every Batch)
					// This prevents them from accumulating any learning signal.
					if len(frozenSet) > 0 {
						for _, layer := range moe.ActiveLayers {
							for expertID, isFrozen := range frozenSet {
								if !isFrozen || expertID >= len(layer.Experts) {
									continue
								}
								for _, p := range layer.Experts[expertID].Parameters() {
									if p.Grad != nil {
										for j := range p.Grad.Data {
											p.Grad.Data[j] = 0.0
										}
									}
								}
							}
						}
					}

					// 3. Update weights every 'accumulationSteps' batches
					if (globalStep+1)%accumulationSteps == 0 {
						params := intentModel.Parameters()
						// --- Global L2 Gradient Clipping via train.ClipParamGrads ---
						// 1. First normalize accumulated grads by accumulation steps so the
						//    clip threshold applies to the effective single-sample gradient magnitude.
						var accScale float32 = 1.0 / float32(accumulationSteps)
						for _, p := range params {
							if p.Grad != nil {
								for i := range p.Grad.Data {
									p.Grad.Data[i] *= accScale
								}
							}
						}
						// 2. Clip on the normalized gradient
						paramGrads := make([][]float32, 0, len(params))
						for _, p := range params {
							if p.Grad != nil {
								paramGrads = append(paramGrads, p.Grad.Data)
							}
						}

						// --- Stability Probe: Identify the parameter contributing most BEFORE clipping ---
						var maxParamNorm float32 = 0.0
						hotParamName := "unknown"
						totalParams := len(params)
						for i, p := range params {
							if p.Grad != nil {
								var pn float32 = 0.0
								for _, g := range p.Grad.Data {
									pn += g * g
								}
								pn = float32(math.Sqrt(float64(pn)))
								if pn > maxParamNorm {
									maxParamNorm = pn
									// Heuristic to guess component
									component := "Expert/Inner"
									if p.IsRouter {
										component = "Router/Gating"
									}
									if i < 3 {
										component = "Embedding/Encoder"
									}
									if i > totalParams-15 {
										component = "Decoder/Projection"
									}

									hotParamName = fmt.Sprintf("%s [idx=%d, size=%d]", component, i, len(p.Data))
								}
							}
						}

						rawNorm, clipped := train.ClipParamGrads(paramGrads, float32(maxGradNorm))
						gradNorm := rawNorm

						if clipped && rawNorm > 20000.0 {
							log.Printf("🔥 [Stability Alert] Top Magnitude: %s | ParamNorm: %.2f | GlobalRawNorm: %.2f | MaxCap: %.1f", hotParamName, maxParamNorm, rawNorm, maxGradNorm)
						}

						// --- SWP: Update Stagnancy Masks ---
						for _, p := range params {
							if p.Grad != nil && p.RequiresGrad {
								if len(p.AccGrads()) == 0 {
									p.SetAccGrads(make([]float32, len(p.Data)))
								}
								p.SetTimidMask(stTracker.CalculateTimidMask(p.AccGrads(), p.Grad.Data))
							}
						}

						// Dynamic LR update using Cosine Decay with 10% Warmup
						totalSteps := epochs * (len(trainPairs) / batchSize)
						currentLR := getLR(globalStep, totalSteps, peakLR)
						optimizer.SetLearningRate(currentLR)

						optimizer.Step()
						if useGPU {
							for _, p := range intentModel.Parameters() {
								p.SyncToDevice()
							}
						}
						optimizer.ZeroGrad()

						clipIndicator := ""
						if clipped {
							clipIndicator = fmt.Sprintf(" [CLIPPED: raw=%.2f, cap=%.1f]", rawNorm, maxGradNorm)
						}
						log.Printf("📥 [Step %d] Weights updated | EffectiveNorm: %.4f%s | LR: %.8f", globalStep, gradNorm, clipIndicator, currentLR)
					}
				}
			}()

			// (Diversity, usage variance, and sparsity losses removed from training path for log clarity.
			// Cross-entropy and Router state losses are primary.)

			// Clear intermediate states to free memory
			intentModel.ClearState()

			// Critical Memory Safety: GC every 32 batches is enough to reclaim intermediates
			// without causing stop-the-world storms on every accumulation step.
			// (Old value of batches%4 caused severe fragmentation on 16GB systems.)
			if batches%32 == 0 {
				runtime.GC()
			}

			// Update metrics
			totalLoss += batchLoss
			batches++

			// Dashboard Reporting (Go-to-WASM Bridge)
			activeExperts := []int{}
			for _, layer := range moe.ActiveLayers {
				// Experts selected for each token in the current batch
				selected := layer.GetSelectedExperts() // [TokenIdx][K]
				for _, experts := range selected {
					activeExperts = append(activeExperts, experts...)
					epochMonitor.LogSelections(experts) // Feed into ExpertMonitor
				}
			}

			metric := &TrainingMetric{
				Step:              globalStep,
				Loss:              batchLoss,
				LoadBalanceLoss:   epochLBLoss / float32(batches),
				LearningRate:      learningRate,
				ActiveExperts:     activeExperts,
				IsCooling:         optimizer.IsActive,
				CircuitBreaker:    isCircuitBreakerTriggered,
				Temperature:       currentTemp,
				ThawedExpertCount: numExpertsToThaw,
			}
			metric.PushToJS()
			if trainingLogger != nil {
				trainingLogger.LogStep(*metric)
			}

			// Save latest metric to file for dashboard polling
			if globalStep%10 == 0 {
				latestJson, _ := json.Marshal(metric)
				os.WriteFile(filepath.Join(projectRoot, "logs/latest_metric.json"), latestJson, 0644)
			}

			// Loss Protection
			if math.IsNaN(float64(totalLoss)) || math.IsInf(float64(totalLoss), 0) {
				log.Fatalf("❌ Loss exploded to NaN/Inf at epoch %d, batch %d. Stopping training.", epoch, batches)
			}

			// Console Logging every 50 batches
			if batches%50 == 0 {
				elapsed := time.Since(epochStartTime).Seconds()
				batchesPerSec := float64(batches) / elapsed
				totalBatches := (len(chatPairs) + batchSize - 1) / batchSize
				log.Printf("Epoch %d, Batch %d/%d, Loss: %.4f (LB: %.4f, Step: %d, LR: %.7f) [%.2f b/s]",
					epoch, batches, totalBatches, batchLoss, epochLBLoss/float32(batches), globalStep, learningRate, batchesPerSec)

				// 🧩 Periodically print a Heatmap for the first expert of each layer
				if batches%200 == 0 {
					for i, layer := range moe.ActiveLayers {
						moe.PrintExpertHeatmap(fmt.Sprintf("L%d E0", i), layer.Experts[0], float32(0.05))
					}
				}
			}

			// 💾 PERIODIC SAVING: Every 200 batches.
			// Each save serialises the entire model, spiking RSS by ~1×model_size.
			// We now save only ONE file (timestamped) and skip the duplicate latest_periodic copy
			// to avoid holding two full serialised copies in memory simultaneously.
			if batches > 0 && batches%200 == 0 {
				log.Printf("💾 [CHECKPOINT] Starting periodic save at Step %d (Batch %d)...", globalStep, batches)
				fmt.Printf("💾 Periodic Saving: Step %d (Batch %d)\n", globalStep, batches)

				// Use a timestamped path for periodic savings
				timestamp := time.Now().Format("20060102_150405")
				periodicPath := filepath.Join(checkpointDir, fmt.Sprintf("ckpt_step%d_%s.gob", globalStep, timestamp))

				// Use Checkpoint struct for full metadata
				ckpt := &moe.Checkpoint{
					Model:           intentModel,
					StepCount:       globalStep,
					LastProfile:     profile,
					Commitment:      intentModel.CalculateCommitment(),
					TokensProcessed: totalTokens,
					TotalDuration:   totalDuration,
					Version:         "gollemer-chat-v1.2-periodic",
				}

				if err := moe.SaveIntentMoECheckpoint(ckpt, periodicPath); err != nil {
					log.Printf("⚠️  Failed to save periodic checkpoint: %v", err)
					fmt.Printf("⚠️  Periodic Save ERROR: %v\n", err)
				}

				// Nil the checkpoint immediately so the GC can reclaim the copy
				// before the next batch allocates more intermediate tensors.
				// NOTE: The duplicate save to latest_periodic.gob was removed — it was
				// serialising a second full model copy, pushing peak RSS 2×model_size.
				ckpt = nil
				runtime.GC()
				debug.FreeOSMemory()
			}

			// Memory safety: Clear computation graph.
			// We only clear gradients if we've just completed an accumulation cycle.
			shouldClearGrads := (globalStep+1)%accumulationSteps == 0
			if shouldClearGrads {
				// Full detach including gradients
				for _, p := range intentModel.Parameters() {
					p.Grad = nil
				}
			}
			DetachModel(intentModel)

			if globalStep > 0 && globalStep%500 == 0 {
				log.Printf("🧪 Periodic Generation Check [Step %d]", globalStep)
				s1, e1 := runTestSentence("Identity", "Who are you?", intentModel)
				s2, e2 := runTestSentence("Help", "How can you help?", intentModel)
				avgScore := (s1 + s2) / 2.0

				// 🧩 Expert Diversity Analysis
				expertCounts := make(map[int]int)
				for _, eid := range append(e1, e2...) {
					expertCounts[eid]++
				}
				mostUsed := -1
				maxC := 0
				for eid, count := range expertCounts {
					if count > maxC {
						maxC = count
						mostUsed = eid
					}
				}
				totalTokens := len(e1) + len(e2)
				dominance := 0.0
				if totalTokens > 0 {
					dominance = float64(maxC) / float64(totalTokens)
				}

				// 🚀 Closed-Loop Adaptive Control
				if avgScore < 8.0 {
					currentEpochTemp += 0.05
					intentModel.SetGateTemperature(currentEpochTemp)
					log.Printf("⚠️  Low Quality (%.1f) -> 🌡️  Increasing Temperature to %.4f", avgScore, currentEpochTemp)
				} else if avgScore > 13.0 {
					peakLR *= 0.8
					log.Printf("🏆  High Quality (%.1f) -> 📉  Slowing Learning Rate to %.8f", avgScore, peakLR)
				}

				if dominance > 0.8 && totalTokens > 3 {
					log.Printf("⚖️  Expert Collapse Detected (E%d = %.1f%%) -> 🎲 Shaking Routers to force exploration", mostUsed, dominance*100)
					intentModel.ShakeRouters(0.08)
					currentEpochTemp += 0.1
					intentModel.SetGateTemperature(currentEpochTemp)

					if dominance > 0.95 && globalStep > 800 {
						log.Printf("🏴‍☠️  MUTINY: Expert %d is too dominant. Reducing its router weights to zero to force other experts to wake up...", mostUsed)
						intentModel.PruneExpertRouter(mostUsed)
					}
				}

				// 🛡️ Load Balance Governor: Prevents LB loss from breaking training
				avgLBLoss := epochLBLoss / float32(batches)
				if avgLBLoss > 1.5 {
					log.Printf("🛑 LB Loss Alert (%.2f) -> Reducing LR to prevent divergence", avgLBLoss)
					peakLR *= 0.7
				}
			}

			globalStep++
			intentModel.ClearState()
		}
		// End of Epoch: log final batch count, print utilization, clear computation graph.
		if batches > 0 {
			log.Printf("Epoch %d, Batch %d/%d, AvgLoss: %.4f (Avg LB: %.4f, Step: %d)", epoch, batches, len(chatPairs), totalLoss/float32(batches), epochLBLoss/float32(batches), globalStep)
		}
		// Visualize Aggregate Utilization
		fmt.Printf("--- 📊 Aggregate Expert Utilization (Epoch %d) ---\n", epoch+1)
		InspectExpertStats(intentModel)

		// Collect all MoE layers (Encoder + Decoder)
		allLayers := intentModel.Encoder.GetMoELayers()
		if intentModel.Decoder.OutputMoE != nil {
			allLayers = append(allLayers, intentModel.Decoder.OutputMoE)
		}

		for layerIdx, layer := range allLayers {
			fmt.Printf("Layer %d Expert Utilization (Capacity Factor: %.2f):\n", layerIdx, layer.CapacityFactor)
			totalTokens := 0
			for i := 0; i < len(layer.Experts); i++ {
				// Use the layer's internal accumulated utilization
				totalTokens += layer.AccumulatedUtilization[i]
			}

			for i := 0; i < len(layer.Experts); i++ {
				count := layer.AccumulatedUtilization[i]
				var percent float32 = 0.0
				if totalTokens > 0 {
					percent = float32(count) / float32(totalTokens) * 100
				}
				bar := strings.Repeat("#", int(percent/2))
				fmt.Printf("  Expert %d: %8d (%5.1f%%) %s\n", i, count, percent, bar)

				// Use internal stagnation counters and call automated reset
				// (The layer's internal metrics are updated in Forward)
			}

			// Automated Evolutionary Reset based on the layer's internal tracking
			layer.EvolutionaryReset(2) // stagnationThreshold=2 epochs (down from 5; epoch-1 saw L4/E3 collapse to 0.6%)

			// After utilization tracking: detect and reset stagnant experts
			// Stagnant = used <1% of the time and not in frozen set (already being forced to learn)
			totalTokensFlt := float32(totalTokens)
			if totalTokensFlt > 0 {
				for i := 0; i < len(layer.Experts); i++ {
					if currentFrozenSet != nil && currentFrozenSet[i] {
						continue // Skip: deliberately frozen by ThawScheduler
					}
					usage := float32(layer.AccumulatedUtilization[i]) / totalTokensFlt
					if usage < 0.01 && epoch > 5 { // Only after warmup (5 epochs)
						fmt.Printf("♻️  Layer %d Expert %d is stagnant (%.2f%% usage). Triggering Evolutionary Reset...\n",
							layerIdx, i, usage*100)
						intentModel.EvolutionaryReset(i, layerIdx)
					}
				}
			}

			// Dominant Expert Freezing: If one expert does > 40% of the work, freeze it
			for i := 0; i < len(layer.Experts); i++ {
				count := layer.AccumulatedUtilization[i]
				usage := 0.0
				if totalTokens > 0 {
					usage = float64(count) / float64(totalTokens)
				}
				if usage > 0.40 {
					fmt.Printf("⚠️ Layer %d Expert %d is dominant (%.1f%%). Freezing for next epoch.\n", layerIdx, i, usage*100)
					layer.SetExpertFreeze(i, true)
				} else {
					// We only unfreeze if it's not already frozen by another mechanism
					// For now, simple toggling
					layer.SetExpertFreeze(i, false)
				}
			}

			// Update expert multipliers based on utilization
			layer.UpdateExpertMultipliers()

			// Reset utilization for the next epoch
			layer.ResetUtilizationStats()
		}
		DetachModel(intentModel)
		avgLoss := float32(0.0)
		if batches > 0 {
			avgLoss = totalLoss / float32(batches)
		}
		fmt.Printf("Epoch %d: Avg Loss %.4f in %.1fs\n", epoch+1, avgLoss, time.Since(epochStartTime).Seconds())
		// Print ExpertMonitor report: imbalance + per-expert counts
		epochMonitor.Report()
		log.Printf("⚖️  [Epoch %d] Load Imbalance (MSE): %.4f | Max Skew: %.1f%%",
			epoch+1, epochMonitor.LoadLoss(), epochMonitor.MaxImbalance()*100)

		// End of epoch memory cleanup
		runtime.GC()
		debug.FreeOSMemory()

		// Update total duration
		totalDuration = time.Since(startTime)

		// Validation
		valPPL := ValidateChat(intentModel, valPairs, useGPU)
		log.Printf("📉 Validation Perplexity: %.2f", valPPL)

		// 🧪 Generation Validation
		runTestSentence("Identity", "Who are you?", intentModel)
		runTestSentence("Help", "How can you help me today?", intentModel)

		// --- [AutoHeal & Health Tracking] ---
		// Find max dominance from the first encoder layer as a proxy for model health
		maxUsage := 0.0
		var l0Counts []int
		if len(allLayers) > 0 {
			l0 := allLayers[0]
			usage := l0.AccumulatedUtilization
			l0Counts = make([]int, len(usage))
			copy(l0Counts, usage)
			total := 0
			maxC := 0
			for _, c := range usage {
				total += c
				if c > maxC {
					maxC = c
				}
			}
			if total > 0 {
				maxUsage = float64(maxC) / float64(total)
			}
		}

		stats := moe.TrainingStats{
			Epoch:          epoch,
			CurrentLoss:    avgLoss,
			Perplexity:     valPPL,
			BestPerplexity: bestPPL,
			Layer0Counts:   l0Counts,
			MaxDominance:   float32(maxUsage),
			StepConfidence: 0.25, // Placeholder
		}

		if autoHeal {
			if adam, ok := optimizer.Base.(*neuralnn.Adam); ok {
				trainer.AutoHeal(intentModel, adam, stats)
			}
		}

		// Log Metrics
		moe.LogWeightStretch(intentModel)
		moe.CheckSaturation(intentModel, epoch)
		// --------------------------------------

		// Reduce-on-Plateau: update scheduler and adjust peakLR
		newLR := lrScheduler.Update(valPPL)
		if newLR != peakLR {
			peakLR = newLR
		}

		// Apply simple Step Decay every 5 epochs as well
		oldPeakLR := peakLR
		peakLR = moe.ApplyStepDecay(peakLR, epoch+1, 5, 0.5)
		if peakLR != oldPeakLR {
			log.Printf("📉 Step Decay applied: peakLR %.8f -> %.8f", oldPeakLR, peakLR)
		}

		scheduler.MaxLR = peakLR
		scheduler.MinLR = peakLR * 0.01

		// ⚖️ Plateau Monitor (as requested)
		plateauMsg := pState.Update(valPPL, pConfig, &peakLR, &currentEpochTemp)
		log.Printf("⚖️ Plateau Monitor: %s", plateauMsg)

		// ⚡ [SWP Trigger] Nudging "Timid" units if training has flatlined
		if pState.BadEpochs >= 5 {
			log.Printf("⚡ [SWP Trigger] Plateau severe (%d epochs). Nudging stagnant weights to reclaim gradient velocity...", pState.BadEpochs)
			params := intentModel.Parameters()
			for _, p := range params {
				if p.TimidMask() != nil {
					train.PerturbStagnantWeights(p.Data, p.TimidMask(), 0.005)
					optimizer.ResetStagnantMoments(p)
				}
			}
			pState.BadEpochs = 0 // Reset after perturbation to monitor fresh trajectory
		}

		// Curriculum Update
		if valPPL < curriculum.MinPPLThreshold && curriculum.MaxSequenceLen < curriculum.MaxSequenceLenLimit {
			curriculum.MaxSequenceLen += curriculum.GrowthFactor
			if curriculum.MaxSequenceLen > curriculum.MaxSequenceLenLimit {
				curriculum.MaxSequenceLen = curriculum.MaxSequenceLenLimit
			}
			curriculum.MinPPLThreshold *= 0.8
			log.Printf("🚀 CURRICULUM LEVEL UP: Max Sequence Length is now %d", curriculum.MaxSequenceLen)
		}

		// Log History
		logEpochHistory(projectRoot, epoch+1, float32(avgLoss), epochLBLoss/float32(batches), learningRate)
		ExportUtilizationCSV(epoch+1, globalStep)

		// periodic snapshots
		ckpt := &moe.Checkpoint{
			Model:           intentModel,
			StepCount:       globalStep,
			LastProfile:     profile,
			Commitment:      intentModel.CalculateCommitment(),
			TokensProcessed: totalTokens,
			TotalDuration:   totalDuration,
			Version:         "gollemer-chat-v1.2",
		}

		// overwriting the main file for compatibility
		moe.SaveIntentMoECheckpoint(ckpt, moePath)

		// GC between saves to avoid doubling peak RSS (two concurrent serializations)
		ckpt = nil
		runtime.GC()
		debug.FreeOSMemory()

		// Rebuild ckpt for the numbered snapshot
		ckpt = &moe.Checkpoint{
			Model:           intentModel,
			StepCount:       globalStep,
			LastProfile:     profile,
			Commitment:      intentModel.CalculateCommitment(),
			TokensProcessed: totalTokens,
			TotalDuration:   totalDuration,
			Version:         "gollemer-chat-v1.2",
		}

		// numbered checkpoint
		numberedPath := filepath.Join(checkpointDir, fmt.Sprintf("epoch_%03d.gob", epoch+1))
		moe.SaveIntentMoECheckpoint(ckpt, numberedPath)

		// Save Best Model
		if valPPL < bestPPL {
			bestPPL = valPPL
			patienceCounter = 0
			if err := moe.SaveIntentMoECheckpoint(ckpt, bestMoePath); err != nil {
				log.Printf("⚠️  Failed to save best MoE model: %v", err)
			} else {
				fmt.Printf("🏆 New Best Model! PPL: %.2f (Saved to %s)\n", bestPPL, bestMoePath)
				trainer.SaveGoldenCheckpoint(intentModel, stats, globalStep, profile, totalTokens, totalDuration)
			}
		} else {
			patienceCounter++
			log.Printf("⚠️  No improvement for %d/%d epochs (best PPL=%.2f, current=%.2f)", patienceCounter, patienceLimit, bestPPL, valPPL)
			if patienceCounter >= patienceLimit {
				log.Printf("🛑 Early stopping triggered after %d epochs without improvement.", patienceLimit)
				// Restore best model weights from disk before stopping
				if loaded, err := moe.LoadIntentMoEModelFromGOB(bestMoePath); err == nil {
					intentModel = loaded
					log.Printf("✅ Restored best model (PPL=%.2f) from %s", bestPPL, bestMoePath)
				} else {
					log.Printf("⚠️  Could not restore best model: %v", err)
				}
				break
			}
		}
	}

	fmt.Printf("✅ Trained on %d chat pairs\n", len(chatPairs))

	// Print the expert MVP/slacker report from the CSV log
	if trainingLogger != nil {
		trainingLogger.Close()
		GenerateProgressReport(logPath)
	}

	// Analyze expert specialization
	analyzeExpertSpecialization(intentModel)

	// 5. Save Vocabulary
	if err := intentModel.SentenceVocab.Save(vocabPath); err != nil {
		log.Printf("Failed to save vocabulary: %v", err)
	} else {
		fmt.Printf("💾 Saved vocabulary to %s\n", vocabPath)
	}
}

// TrainSocialChat trains a specialized model ONLY on human_chat.txt for pure social conversations
// It reuses TrainChat infrastructure but with social-only data
func TrainSocialChat(projectRoot string, overfitMode bool, initialLR float32, weightDecay float32, autoHeal bool, maxGradNorm float32, useGPU bool, batchSize int, accumulationSteps int) {
	log.Println("🎭 Starting SOCIAL-ONLY Chat Training (human_chat.txt only)")

	var chatPairs []struct{ Q, A, Intent string }
	var err error

	// Load ONLY human_chat.txt for social training
	humanChatPath := filepath.Join(projectRoot, "data/training/trainingdata/human_chat.txt")
	if humanChatData, err := os.ReadFile(humanChatPath); err == nil {
		humanChatLines := strings.Split(string(humanChatData), "\n")
		var currentQ, currentA string

		for _, line := range humanChatLines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			if strings.HasPrefix(line, "Human 1:") {
				if currentQ != "" && currentA != "" {
					chatPairs = append(chatPairs, struct{ Q, A, Intent string }{currentQ, currentA, "social"})
				}
				currentQ = strings.TrimPrefix(line, "Human 1:")
				currentQ = strings.TrimSpace(currentQ)
				currentA = ""
			} else if strings.HasPrefix(line, "Human 2:") {
				currentA = strings.TrimPrefix(line, "Human 2:")
				currentA = strings.TrimSpace(currentA)
			}
		}

		if currentQ != "" && currentA != "" {
			chatPairs = append(chatPairs, struct{ Q, A, Intent string }{currentQ, currentA, "social"})
		}

		log.Printf("📊 Loaded %d social conversation pairs from human_chat.txt", len(chatPairs))
	} else {
		log.Fatalf("❌ human_chat.txt not found at %s", humanChatPath)
	}

	if len(chatPairs) == 0 {
		log.Fatalf("❌ No social pairs loaded from human_chat.txt")
	}

	// Reuse TrainChat with social-only data by temporarily renaming model output
	// Call TrainChat with the social data
	oldChatPairs := chatPairs

	// Pre-compute vocabulary
	tmpVocab := mainvocab.NewVocabulary()
	tmpVocab.AddToken("<pad>")
	tmpVocab.AddToken("<s>")
	tmpVocab.AddToken("</s>")
	tmpVocab.AddToken("UNK")
	for _, pair := range chatPairs {
		for _, t := range cleanTokenize(pair.Q + " " + pair.A) {
			tmpVocab.AddToken(t)
		}
	}
	precomputedVocabSize := tmpVocab.Size()
	tmpVocab = nil
	runtime.GC()
	log.Printf("📐 Pre-computed final vocab size: %d", precomputedVocabSize)

	// 🏗️ Step 1: Model Loading or Initialization
	var intentModel *moe.IntentMoE
	socialModelPath := filepath.Join(projectRoot, "data/models/gob_models/moe_social_model.gob")
	socialVocabPath := filepath.Join(projectRoot, "data/models/gob_models/social_vocabulary.gob")

	if _, err := os.Stat(socialModelPath); err == nil {
		log.Printf("📥 Resuming training: Loading existing social model from %s", socialModelPath)
		intentModel, err = moe.LoadIntentMoEModelFromGOB(socialModelPath)
		if err != nil {
			log.Printf("⚠️ Failed to load existing model: %v. Starting fresh.", err)
			intentModel = nil
		}
	}

	if intentModel == nil {
		const modelDim = 256
		const numExperts = 4
		freshVocab := precomputedVocabSize
		if freshVocab < 100 {
			freshVocab = 2000
		}

		log.Printf("🚀 Initializing fresh social model: 256d, %d experts", numExperts)
		intentModel, err = moe.NewHybridIntentMoE(freshVocab, modelDim, numExperts, modelDim, modelDim, freshVocab, 8, nil)
		if err != nil {
			log.Fatalf("❌ Failed to create social model: %v", err)
		}
		intentModel.Decoder, _ = moe.NewRNNDecoder(modelDim, freshVocab, modelDim, 8, 4, 0.1, numExperts)

		// Standard initialization for fresh model
		allLayers := findMoELayers(intentModel)
		for _, layer := range allLayers {
			if layer.Experts != nil {
				for _, expert := range layer.Experts {
					for _, param := range expert.Parameters() {
						InitializeHeNormal(param)
					}
				}
			}
		}
	}

	if useGPU {
		intentModel.ToGPU()
	}

	// Shuffle and split
	rand.Shuffle(len(chatPairs), func(i, j int) { chatPairs[i], chatPairs[j] = chatPairs[j], chatPairs[i] })
	splitIdx := int(float64(len(chatPairs)) * 0.9)
	trainPairs := chatPairs[:splitIdx]
	valPairs := chatPairs[splitIdx:]

	log.Printf("📂 Data: %d training, %d validation", len(trainPairs), len(valPairs))

	if intentModel.SentenceVocab == nil || intentModel.SentenceVocab.Size() < 5 {
		// Try to load from social_vocabulary.gob first
		if v, err := mainvocab.LoadVocabulary(socialVocabPath); err == nil {
			log.Printf("🔤 Loaded existing vocabulary from %s: %d tokens", socialVocabPath, v.Size())
			intentModel.SentenceVocab = v
		} else {
			log.Println("🔤 Building fresh vocabulary from scratch...")
			sentenceVocab := mainvocab.NewVocabulary()
			sentenceVocab.AddToken("<pad>")
			sentenceVocab.AddToken("<s>")
			sentenceVocab.AddToken("</s>")
			sentenceVocab.AddToken("UNK")

			for _, pair := range trainPairs {
				tokens := cleanTokenize(pair.Q + " " + pair.A)
				for _, t := range tokens {
					sentenceVocab.AddToken(t)
				}
			}
			for _, pair := range valPairs {
				tokens := cleanTokenize(pair.Q + " " + pair.A)
				for _, t := range tokens {
					sentenceVocab.AddToken(t)
				}
			}

			intentModel.SentenceVocab = sentenceVocab
			intentModel.SentenceVocabSize = sentenceVocab.Size()

			// Set special token IDs
			sentenceVocab.BosID = sentenceVocab.GetTokenID("<s>")
			sentenceVocab.EosID = sentenceVocab.GetTokenID("</s>")
			sentenceVocab.PaddingTokenID = sentenceVocab.GetTokenID("<pad>")
			log.Printf("🔤 Built vocabulary: %d tokens (BOS=%d, EOS=%d, PAD=%d)", sentenceVocab.Size(), sentenceVocab.BosID, sentenceVocab.EosID, sentenceVocab.PaddingTokenID)
		}
	} else {
		log.Printf("🔤 Using existing model vocabulary: %d tokens", intentModel.SentenceVocab.Size())
	}

	sentenceVocab := intentModel.SentenceVocab
	// Set special token IDs (redundant but safe)
	sentenceVocab.BosID = sentenceVocab.GetTokenID("<s>")
	sentenceVocab.EosID = sentenceVocab.GetTokenID("</s>")
	sentenceVocab.PaddingTokenID = sentenceVocab.GetTokenID("<pad>")

	log.Printf("🔤 Vocabulary State: %d tokens (BOS=%d, EOS=%d, PAD=%d)", sentenceVocab.Size(), sentenceVocab.BosID, sentenceVocab.EosID, sentenceVocab.PaddingTokenID)

	moe.ActiveLayers = findMoELayers(intentModel)

	// Use iterator-based training (same as TrainChat)
	epochs := 60
	peakLR := initialLR
	if peakLR == 0 {
		peakLR = 0.0005
	}

	log.Printf("🎭 Training social model for %d epochs at peak LR=%.6f", epochs, peakLR)

	// Create loss weights for vocabulary
	lossWeights := make([]float32, intentModel.SentenceVocab.Size())
	for i := range lossWeights {
		lossWeights[i] = 1.0
	}
	unkID := intentModel.SentenceVocab.GetTokenID("UNK")
	if unkID >= 0 {
		lossWeights[unkID] = 0.01 // Reduce weight for unknown tokens
	}
	if intentModel.SentenceVocab.PaddingTokenID >= 0 {
		lossWeights[intentModel.SentenceVocab.PaddingTokenID] = 0.0 // No loss for padding
	}

	// Create optimizer (Wrapped with Cooling Safety)
	baseOptimizer := neuralnn.NewOptimizer(intentModel.Parameters(), peakLR, 1.0)
	optimizer := &neuralnn.CoolingOptimizer{
		Base: baseOptimizer,
	}

	globalStep := 0
	totalSteps := epochs * len(trainPairs)

	for epoch := 0; epoch < epochs; epoch++ {
		// Stop if requested (checked via PROJECT_ROOT/.stop)
		if _, err := os.Stat(filepath.Join(projectRoot, ".stop")); err == nil {
			log.Println("🛑 Stop signal detected. Saving and exiting...")
			os.Remove(filepath.Join(projectRoot, ".stop"))
			break
		}

		iterator := NewChatDataIterator(trainPairs, intentModel.SentenceVocab, unkID)
		epochLoss := float32(0)
		batchNum := 0

		for iterator.HasNext() {
			batch := iterator.NextBatch(batchSize)
			inputTensor, targetTensor := batch.Input, batch.Target
			inputMask := batch.InputMask
			if inputTensor == nil || targetTensor == nil {
				continue
			}

			if useGPU {
				inputTensor.ToGPU()
				targetTensor.ToGPU()
				if inputMask != nil {
					inputMask.ToGPU()
				}
			}

			if targetTensor.Shape[1] < 2 {
				continue
			}

			// Forward pass
			// Scheduled sampling gradually increases from teacher forcing (0%) to 30% model-generated
			samplingProb := float32(math.Min(0.3, float64(epoch)*0.01))
			logits, _, err := intentModel.Forward(samplingProb, inputTensor, targetTensor, inputMask)
			if err != nil {
				continue
			}

			// Compute loss
			var batchLoss float32 = 0.0
			var grads []*tensor.Tensor

			if len(logits) == 1 && len(logits[0].Shape) == 3 {
				l := logits[0]
				currentBatchSize := targetTensor.Shape[0]
				seqLen := targetTensor.Shape[1]
				targetSeqLen := seqLen - 1
				targets := make([]int, currentBatchSize*targetSeqLen)
				for b := 0; b < currentBatchSize; b++ {
					for t := 0; t < targetSeqLen; t++ {
						targets[b*targetSeqLen+t] = int(targetTensor.Data[b*seqLen+t+1])
					}
				}
				loss, grad := WeightedCrossEntropy(l.ToCPU(), targets, lossWeights, 0.0)
				if grad == nil {
					grad = tensor.NewTensor(l.Shape, make([]float32, len(l.Data)), false)
				}
				batchLoss = loss
				grads = []*tensor.Tensor{grad}
			} else {
				grads = make([]*tensor.Tensor, len(logits))
				currentBatchSize := targetTensor.Shape[0]
				seqLen := targetTensor.Shape[1]
				var stepLossTotal float32 = 0.0
				for t, logit := range logits {
					targets := make([]int, currentBatchSize)
					for b := 0; b < currentBatchSize; b++ {
						targets[b] = int(targetTensor.Data[b*seqLen+t+1])
					}
					l, g := WeightedCrossEntropy(logit.ToCPU(), targets, lossWeights, 0.0)
					if g == nil {
						g = tensor.NewTensor(logit.Shape, make([]float32, len(logit.Data)), false)
					}
					stepLossTotal += l
					grads[t] = g
				}
				div := float32(len(logits))
				batchLoss = stepLossTotal / div
				for t := range grads {
					for i := range grads[t].Data {
						grads[t].Data[i] /= div
					}
				}
			}

			if !math.IsNaN(float64(batchLoss)) && !math.IsInf(float64(batchLoss), 0) {
				// Backward pass
				if err := intentModel.Backward(grads...); err == nil {
					if (globalStep+1)%accumulationSteps == 0 {
						params := intentModel.Parameters()
						for _, p := range params {
							if p.Grad != nil {
								for i := range p.Grad.Data {
									p.Grad.Data[i] /= float32(accumulationSteps)
								}
							}
						}
						// Build paramGrads for clipping
						paramGrads := make([][]float32, 0, len(params))
						for _, p := range params {
							if p.Grad != nil {
								paramGrads = append(paramGrads, p.Grad.Data)
							}
						}
						train.ClipParamGrads(paramGrads, 1.0)

						// Update LR using Cosine Decay
						currentLR := getLR(globalStep, totalSteps, peakLR)
						optimizer.SetLearningRate(currentLR)

						optimizer.Step()
						if useGPU {
							for _, p := range params {
								p.SyncToDevice()
							}
						}
						optimizer.ZeroGrad()
					}
					epochLoss += batchLoss
				}
			}

			batchNum++
			globalStep++
			if batchNum%1 == 0 {
				avgLoss := epochLoss / float32(batchNum)
				log.Printf("🎭 Epoch %d/%d | Batch %d | Loss: %.6f | LR: %.8f", epoch+1, epochs, batchNum, avgLoss, optimizer.GetLearningRate())
			}

			// 💾 SAVE PROGRESS MID-EPOCH (As requested)
			if batchNum > 0 && batchNum%100 == 0 {
				ckptPath := filepath.Join(projectRoot, fmt.Sprintf("data/models/gob_models/moe_social_model_step_%d.gob", globalStep))
				log.Printf("💾 Saving periodic checkpoint at Step %d...", globalStep)
				if err := moe.SaveIntentMoEModelToGOB(intentModel, ckptPath); err != nil {
					log.Printf("❌ Mid-epoch save failed: %v", err)
				}
				// Also update the main social model file so restarts pick up latest progress
				_ = moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
			}

			DetachModel(intentModel)
		}

		log.Printf("✅ Epoch %d/%d completed | Avg Loss: %.6f", epoch+1, epochs, epochLoss/float32(batchNum))

		// 🧪 TEST GENERATION: See if it's learning sentences
		testPrompts := []string{"hello", "how are you", "what is your favorite movie"}
		for _, p := range testPrompts {
			response := StrictGenerate(intentModel, p, 20)
			log.Printf("🧪 Test '%s': %s", p, response)
		}

		// Save checkpoint at the end of every epoch
		checkpointPath := filepath.Join(projectRoot, fmt.Sprintf("data/models/gob_models/moe_social_model_epoch_%03d.gob", epoch+1))
		if err := moe.SaveIntentMoEModelToGOB(intentModel, checkpointPath); err != nil {
			log.Printf("❌ Failed to save checkpoint at epoch %d: %v", epoch+1, err)
		} else {
			log.Printf("💾 Saved checkpoint: Epoch %d/%d", epoch+1, epochs)
		}
		// Update latest main file
		_ = moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
	}

	// Save final social model
	if err := moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath); err != nil {
		log.Printf("❌ Failed to save social model: %v", err)
	} else {
		fmt.Printf("💾 Saved final social model to %s\n", socialModelPath)
	}

	// Save social vocabulary
	socialVocabPathFinal := filepath.Join(projectRoot, "data/models/gob_models/social_vocabulary.gob")
	if err := intentModel.SentenceVocab.Save(socialVocabPathFinal); err != nil {
		log.Printf("❌ Failed to save social vocabulary: %v", err)
	} else {
		fmt.Printf("💾 Saved social vocabulary to %s\n", socialVocabPathFinal)
	}

	log.Println("🎭 Social-only training complete!")
	_ = oldChatPairs // Keep reference
}

// StrictGenerate forces the model to generate a response without using UNK or PAD tokens.
func StrictGenerate(model *moe.IntentMoE, input string, maxLen int) string {
	// 1. Enter Eval Mode and set Router Temperature for stability
	oldTemps := make(map[*moe.MoELayer]float32)
	for _, layer := range moe.ActiveLayers {
		layer.SetMode(false)
		oldTemps[layer] = layer.RouterTemperature
		layer.RouterTemperature = 1.1 // Slightly higher for exploration during generation
	}
	if model.Decoder.OutputMoE != nil {
		oldTemps[model.Decoder.OutputMoE] = model.Decoder.OutputMoE.RouterTemperature
		model.Decoder.OutputMoE.RouterTemperature = 1.1
	}

	defer func() {
		for layer, temp := range oldTemps {
			layer.SetMode(true)
			layer.RouterTemperature = temp
		}
		if model.Decoder.OutputMoE != nil {
			model.Decoder.OutputMoE.SetMode(true)
			model.Decoder.OutputMoE.RouterTemperature = oldTemps[model.Decoder.OutputMoE]
		}
	}()

	// 1. Tokenize and Vectorize Input
	tokens := cleanTokenize(input)
	if len(tokens) == 0 {
		log.Printf("⚠️ Skip empty prompt: %s", input)
		return ""
	}
	inputIDs := make([]float32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	// 2. Get Encoder Context
	emb, err := model.Embedding.Forward(inputTensor)
	if err != nil {
		log.Printf("StrictGenerate Error (Embedding): %v", err)
		return ""
	}
	ctx, err := model.Encoder.Forward(emb)
	if err != nil {
		log.Printf("StrictGenerate Error (Encoder): %v", err)
		return ""
	}
	// Normalize context vector to match training scale
	ctx = model.NormalizeContextVector(ctx)

	// 🔍 Diagnostic: Importance Map for first layer in the stack
	if len(moe.ActiveLayers) > 0 {
		PrintImportanceMap(model.SentenceVocab, tokens, moe.ActiveLayers[0])
		moe.PrintLayerWeightHistogram("Encoder Layer 0", moe.ActiveLayers[0])
	}
	if ctx.Shape[1] == 0 {
		log.Printf("StrictGenerate Error: encoder produced empty sequence")
		return ""
	}

	// 3. Prepare Decoder States
	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize

	// Initial hidden state from CONTEXT MEAN (matching RNNDecoder.Forward logic)
	// This ensures consistency between training and generation.
	hiddenState, err := ctx.Mean(1)
	if err != nil {
		log.Printf("StrictGenerate Error (Initial Hidden Mean): %v", err)
		return ""
	}
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})

	// Projection if needed (copying logic from Decoder.Forward)
	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float32, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	// 4. Start Sequence with <s> (BOS)
	resIDs := []int{model.SentenceVocab.BosID}
	currentTokenID := model.SentenceVocab.BosID

	// Track counts for frequency penalty during generation
	counts := make(map[int]int)

	var path []string
	ctxNorm := ctx.L2Norm()
	fmt.Printf("📡 Encoder Context Strength: %.4f | Vector[0:3]: %.4f, %.4f, %.4f\n", ctxNorm, ctx.Data[0], ctx.Data[1], ctx.Data[2])

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(currentTokenID)}, false)

		oldHiddenNorm := hiddenState.L2Norm()
		// 1. Step-by-step decoding with expert tracking
		logits, nextHidden, nextCell, expertID, err := model.Decoder.DecodeStepWithExpert(inputT, hiddenState, cellState, ctx)
		if err != nil {
			log.Printf("StrictGenerate Error (DecodeStep): %v", err)
			break
		}
		newHiddenNorm := nextHidden.L2Norm()

		// Update states
		hiddenState = nextHidden
		cellState = nextCell

		// Diagnostic: Context Influence and Expert
		// The previous oldHiddenNorm and newHiddenNorm were for the *current* step's input.
		// This diagnostic is for the *change* in hidden state after the step.
		// So, oldHiddenNorm should be the state *before* the update, and newHiddenNorm *after*.
		// The variables were already declared above, so remove the `:=`
		hiddenDelta := float32(math.Abs(float64(newHiddenNorm - oldHiddenNorm)))
		fmt.Printf("🔍 Step %d | Context Influence: %.4f | Expert: E%d\n", i, hiddenDelta, expertID)

		// [Diagnostic] Log Top-3 predictions for Step 0 to debug "Still Silent"
		if i == 0 {
			LogTopPredictions(model, "Step 0 Generation", logits)
			// Mute EOS at Step 0 to force generation
			logits.Data[model.SentenceVocab.EosID] = -1e9
		}

		// 5. Apply Repetition and Frequency Penalty
		// Repetition Penalty (multiplicative)
		moe.ApplyRepetitionPenalty(logits, resIDs, 1.2) // Lowered from 1.8

		// Frequency Penalty (additive)
		const frequencyPenalty = 0.5
		for id, count := range counts {
			if id < len(logits.Data) {
				logits.Data[id] -= frequencyPenalty * float32(count)
			}
		}

		// 6. Mute UNK and <pad>
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		unkID := model.SentenceVocab.GetTokenID("UNK")
		if unkID != -1 {
			logits.Data[unkID] = -1e9
		}

		// 7. Pick Best Word (Fixed Temp/Top-P as requested)
		bestID, err := moe.SampleFromLogits(logits, 0.8, 1, 0.9)
		if err != nil {
			log.Printf("StrictGenerate Error (Sampling): %v", err)
			break
		}

		// 8. Early Stopping: If even the best choice is "noise"
		probs := tensor.Softmax(logits) // Declare probs here so it's available for both checks
		topProb := probs.Data[bestID]
		if i > 2 && topProb < 0.01 { // threshold 1%
			log.Printf("🛑 Early stop at step %d: Low confidence (%.2f%%)", i, topProb*100)
			break
		}

		if bestID == model.SentenceVocab.EosID {
			break
		}

		resIDs = append(resIDs, bestID)
		counts[bestID]++
		currentTokenID = bestID

		// Top-K Diagnostic summary
		topIndices, topValues := getTopK(probs, 5)
		fmt.Printf("🔍 Step %d | Top Choices:\n", i)
		for k := 0; k < 5; k++ {
			word := model.SentenceVocab.GetWord(topIndices[k])
			fmt.Printf("   [%d] %-12s (%.2f%%)\n", k+1, word, topValues[k]*100)
		}

		// Record the word and the expert that produced it
		word := model.SentenceVocab.GetWord(bestID)
		path = append(path, fmt.Sprintf("%s(E%d)", word, expertID))
	}

	// Convert IDs back to words
	var result []string
	for _, id := range resIDs[1:] { // Skip the BOS token
		word := model.SentenceVocab.GetWord(id)
		// Final check to not include special tokens in the final string
		if word != "<s>" && word != "</s>" && word != "<pad>" && word != "UNK" {
			result = append(result, word)
		}
	}

	// 3. Print the diagnostic expert path
	if len(path) > 0 {
		fmt.Printf("\n🧠 Expert Path: %s\n", strings.Join(path, " -> "))
	}

	return strings.Join(result, " ")
}

// StrictGenerateWithExperts is a variant of StrictGenerate that also returns the expert IDs used.
func StrictGenerateWithExperts(model *moe.IntentMoE, input string, maxLen int) (string, []int) {
	// 1. Enter Eval Mode and set Router Temperature for stability
	oldTemps := make(map[*moe.MoELayer]float32)
	for _, layer := range moe.ActiveLayers {
		layer.SetMode(false)
		oldTemps[layer] = layer.RouterTemperature
		layer.RouterTemperature = 1.1
	}
	if model.Decoder.OutputMoE != nil {
		oldTemps[model.Decoder.OutputMoE] = model.Decoder.OutputMoE.RouterTemperature
		model.Decoder.OutputMoE.RouterTemperature = 1.1
	}

	defer func() {
		for layer, temp := range oldTemps {
			layer.SetMode(true)
			layer.RouterTemperature = temp
		}
		if model.Decoder.OutputMoE != nil {
			model.Decoder.OutputMoE.SetMode(true)
			model.Decoder.OutputMoE.RouterTemperature = oldTemps[model.Decoder.OutputMoE]
		}
	}()

	tokens := cleanTokenize(input)
	if len(tokens) == 0 {
		return "", nil
	}
	inputIDs := make([]float32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	emb, err := model.Embedding.Forward(inputTensor)
	if err != nil {
		return "", nil
	}
	ctx, err := model.Encoder.Forward(emb)
	if err != nil {
		return "", nil
	}
	ctx = model.NormalizeContextVector(ctx)

	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	hiddenState, _ := ctx.Mean(1)
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})

	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float32, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	resIDs := []int{model.SentenceVocab.BosID}
	var usedExpertIDs []int
	currentTokenID := model.SentenceVocab.BosID
	counts := make(map[int]int)

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(currentTokenID)}, false)
		logits, nextHidden, nextCell, expertID, err := model.Decoder.DecodeStepWithExpert(inputT, hiddenState, cellState, ctx)
		if err != nil {
			break
		}
		hiddenState = nextHidden
		cellState = nextCell
		usedExpertIDs = append(usedExpertIDs, expertID)

		// 🛠️ Add extra noise if we are in a MUTINY aftermath (Temp > 1.2)
		if model.Encoder.GetMoELayers()[0].RouterTemperature > 1.2 {
			for idx := range logits.Data {
				logits.Data[idx] += float32((rand.Float64()*2 - 1) * 0.15)
			}
		}

		moe.ApplyRepetitionPenalty(logits, resIDs, 2.5) // Harsh penalty for validation
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		unkID := model.SentenceVocab.GetTokenID("UNK")
		if unkID != -1 {
			logits.Data[unkID] = -1e9
		}

		bestID, err := moe.SampleFromLogits(logits, 0.8, 1, 0.9)
		if err != nil {
			break
		}

		if i == 0 {
			logits.Data[model.SentenceVocab.EosID] = -1e9
		}
		if bestID == model.SentenceVocab.EosID {
			break
		}

		resIDs = append(resIDs, bestID)
		counts[bestID]++
		currentTokenID = bestID
	}

	var result []string
	for _, id := range resIDs[1:] {
		word := model.SentenceVocab.GetWord(id)
		if word != "<s>" && word != "</s>" && word != "<pad>" && word != "UNK" {
			result = append(result, word)
		}
	}
	return strings.Join(result, " "), usedExpertIDs
}

// runTestSentence is a helper to run test questions during training
func runTestSentence(label, input string, model *moe.IntentMoE) (float32, []int) {
	// 1. Generate the response using Strict Mode
	// (Internally it already logs the expert path)
	// We need to capture the expert IDs from the decoder steps.
	// We'll modify StrictGenerate to return used expert IDs.
	response, expertIDs := StrictGenerateWithExperts(model, input, 20)

	// 2. Clean up the output
	response = strings.TrimSpace(response)
	if response == "" {
		response = "[Still Silent]"
	}

	// 3. Score the sentence quality
	score := scoreSentenceHeuristic(response)

	// 4. Log the result with the performance score
	log.Printf("🧪 Test '%s' (%s): %s [Quality Score: %.1f]", input, label, response, score)

	return score, expertIDs
}

// LogTopPredictions analyzes the model output for a test prompt.
// It shows what the model is "thinking" even if it's not confident yet,
// by printing the top-3 candidates from the raw logit vector.
// This is useful for diagnosing [Still Silent] outputs during training.
func LogTopPredictions(model *moe.IntentMoE, testName string, logits *tensor.Tensor) {
	if model.SentenceVocab == nil || logits == nil {
		return
	}

	type prediction struct {
		token string
		prob  float64
	}

	// 1. Convert logits to probabilities via softmax
	vocabSize := logits.Shape[len(logits.Shape)-1]
	logitsFlat := logits.Data[len(logits.Data)-vocabSize:] // Use last row if batch/seq dim
	maxL := logitsFlat[0]
	for _, v := range logitsFlat {
		if v > maxL {
			maxL = v
		}
	}
	sum := float32(0.0)
	probs := make([]float32, vocabSize)
	for i, v := range logitsFlat {
		probs[i] = float32(math.Exp(float64(v - maxL)))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// 2. Collect and sort predictions
	results := make([]prediction, vocabSize)
	for i, p := range probs {
		results[i] = prediction{
			token: model.SentenceVocab.GetWord(i),
			prob:  float64(p),
		}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].prob > results[j].prob
	})

	// 3. Print top 3 contenders
	fmt.Printf("🧪 Test '%s':\n", testName)
	for i := 0; i < 3 && i < len(results); i++ {
		displayToken := results[i].token
		if displayToken == "" {
			displayToken = "<empty>"
		}
		if displayToken == " " {
			displayToken = "[SPACE]"
		}
		fmt.Printf("   [%d] %-12s (%.4f%%)\n", i+1, displayToken, results[i].prob*100)
	}
	fmt.Println("-----------------------------------")
}

// lookupVocab tries to find a token ID in the given vocabulary with fallbacks.
func lookupVocab(token string, vocab *mainvocab.Vocabulary) int {
	token = strings.ToLower(strings.TrimSpace(token))
	id := vocab.GetTokenID(token)
	if id != -1 && (id != 0 || token == "<pad>") {
		return id
	}
	// Try stripping trailing punctuation
	stripped := strings.TrimRight(token, ".,!?;:'\"")
	if stripped != token {
		id = vocab.GetTokenID(stripped)
		if id != -1 && (id != 0 || stripped == "<pad>") {
			return id
		}
	}
	// Fall back to UNK
	return vocab.GetTokenID("UNK")
}

// lookupW2V is deprecated in favor of lookupVocab but kept for backward compatibility if needed.
func lookupW2V(token string, w2v *word2vec.SimpleWord2Vec) float64 {
	token = strings.ToLower(strings.TrimSpace(token))
	if id, ok := w2v.Vocabulary[token]; ok {
		return float64(id)
	}
	return 0
}

// logEpochHistory saves epoch metrics to a CSV file
func logEpochHistory(projectRoot string, epoch int, loss float32, lbLoss float32, lr float32) {
	historyPath := filepath.Join(projectRoot, "logs/training_history.csv")
	file, err := os.OpenFile(historyPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("Warning: Could not open history file: %v", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Add header if new file
	if info, err := os.Stat(historyPath); err == nil && info.Size() == 0 {
		writer.Write([]string{"epoch", "avg_loss", "lb_loss", "learning_rate", "timestamp"})
	}

	writer.Write([]string{
		strconv.Itoa(epoch),
		fmt.Sprintf("%.4f", loss),
		fmt.Sprintf("%.4f", lbLoss),
		fmt.Sprintf("%.6f", lr),
		time.Now().Format(time.RFC3339),
	})
}

// ChatDataIterator implements the Iterator pattern for training data.
// It tokenizes data on-the-fly to save memory and supports shuffling.
type ChatDataIterator struct {
	pairs  []struct{ Q, A, Intent string }
	vocab  *mainvocab.Vocabulary
	unkID  int
	idx    int
	MaxLen int
}

func NewChatDataIterator(pairs []struct{ Q, A, Intent string }, vocab *mainvocab.Vocabulary, unkID int) *ChatDataIterator {
	// Shuffle pairs for better training
	rand.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	return &ChatDataIterator{
		pairs:  pairs,
		vocab:  vocab,
		unkID:  unkID,
		idx:    0,
		MaxLen: 80, // Default cap
	}
}

func (it *ChatDataIterator) HasNext() bool {
	return it.idx < len(it.pairs)
}

func (it *ChatDataIterator) Next() (*tensor.Tensor, *tensor.Tensor) {
	pair := it.pairs[it.idx]
	it.idx++

	// Query Tokenization (now also using SentenceVocab!)
	qTokens := cleanTokenize(pair.Q)
	qIDs := make([]float32, len(qTokens))
	for i, t := range qTokens {
		qIDs[i] = float32(lookupVocab(t, it.vocab))
	}

	if len(qIDs) == 0 {
		qIDs = []float32{0}
	}

	// Response Tokenization (SentenceVocab)
	aTokens := cleanTokenize(pair.A)
	aIDs := make([]float32, len(aTokens)+2) // +2 for BOS and EOS
	aIDs[0] = float32(it.vocab.BosID)
	idx := 1
	for _, t := range aTokens {
		id := it.vocab.GetTokenID(t)
		if id == -1 || (id == 0 && t != "<pad>") {
			aIDs[idx] = float32(it.unkID)
		} else {
			aIDs[idx] = float32(id)
		}
		idx++
	}
	aIDs[idx] = float32(it.vocab.EosID)

	inputTensor := tensor.NewTensor([]int{1, len(qIDs)}, qIDs, false)
	targetTensor := tensor.NewTensor([]int{1, len(aIDs)}, aIDs, false)
	return inputTensor, targetTensor
}

func (it *ChatDataIterator) NextBatch(batchSize int) *Batch {
	var inputs [][]float32
	var targets [][]float32
	maxIn, maxOut := 0, 0

	for i := 0; i < batchSize && it.HasNext(); i++ {
		inp, tgt := it.Next()
		// Sequence length constraint: respects curriculum limit
		if len(inp.Data) > it.MaxLen || len(tgt.Data) > it.MaxLen {
			continue
		}
		inputs = append(inputs, inp.Data)
		targets = append(targets, tgt.Data)
		if len(inp.Data) > maxIn {
			maxIn = len(inp.Data)
		}
		if len(tgt.Data) > maxOut {
			maxOut = len(tgt.Data)
		}
	}

	if len(inputs) == 0 {
		return &Batch{}
	}

	paddedIn := make([]float32, len(inputs)*maxIn)
	paddedOut := make([]float32, len(targets)*maxOut)
	mask := make([]float32, len(targets)*maxOut)
	inputLogitMask := make([]float32, len(inputs)*maxIn) // For attention: 0 for real, -1e9 for pad
	padID := float32(it.vocab.PaddingTokenID)

	for i := range inputs {
		for j := 0; j < maxIn; j++ {
			if j < len(inputs[i]) {
				paddedIn[i*maxIn+j] = inputs[i][j]
				inputLogitMask[i*maxIn+j] = 0.0
			} else {
				paddedIn[i*maxIn+j] = padID
				inputLogitMask[i*maxIn+j] = -1e9
			}
		}
		for j := 0; j < maxOut; j++ {
			if j < len(targets[i]) {
				paddedOut[i*maxOut+j] = targets[i][j]
				mask[i*maxOut+j] = 1.0
			} else {
				paddedOut[i*maxOut+j] = padID
				mask[i*maxOut+j] = 0.0
			}
		}
	}

	// Reshape InputMask for attention: [Batch, 1, 1, SeqLen]
	inputMaskTensor := tensor.NewTensor([]int{len(inputs), 1, 1, maxIn}, inputLogitMask, false)

	return &Batch{
		Input:     tensor.NewTensor([]int{len(inputs), maxIn}, paddedIn, false),
		Target:    tensor.NewTensor([]int{len(targets), maxOut}, paddedOut, false),
		Mask:      mask,
		InputMask: inputMaskTensor,
	}
}

func (it *ChatDataIterator) Reset() {
	it.idx = 0
	rand.Shuffle(len(it.pairs), func(i, j int) { it.pairs[i], it.pairs[j] = it.pairs[j], it.pairs[i] })
	log.Println("🔄 Shuffled training data for new epoch")
}

// DetachModel removes the computation graph from the model parameters and clears internal states.
func DetachModel(model *moe.IntentMoE) {
	if model == nil {
		return
	}
	for _, param := range model.Parameters() {
		if param != nil {
			param.Creator = nil
			param.Mask = nil
			param.Operation = nil
			// Temporary tensors used in forward pass are NOT parameters.
			// model.ClearState() will handle them.
		}
	}
	// Clear all intermediate tensors and cached states across all layers
	model.ClearState()
	// Optionally clear decoder MoE if it exists and is not in ActiveLayers
	if model.Decoder.OutputMoE != nil {
		model.Decoder.OutputMoE.ClearState()
	}
	// Aggressively trigger GC to free up VRAM associated with intermediate tensors
	runtime.GC()
}

func visualizeExpertUtilization() {
	for i, layer := range moe.ActiveLayers {
		fmt.Printf("Layer %d ", i)
		layer.VisualizeUtilization()
	}
}

func analyzeExpertSpecialization(model *moe.IntentMoE) {
	fmt.Println("\n--- 🧠 Expert Specialization Analysis ---")

	for layerIdx, layer := range moe.ActiveLayers {
		fmt.Printf("Layer %d:\n", layerIdx)

		// Map of expert index to map of token to count
		specialization := make(map[int]map[string]int)
		for i := 0; i < len(layer.Experts); i++ {
			specialization[i] = make(map[string]int)
		}

		// We need to look at the last forward pass's selected experts
		// and the tokens they processed. Since analyzeExpertSpecialization is called
		// at the end, we'll run a few sentences through to gather fresh stats.

		sampleSentences := []string{
			"how are you today",
			"what is your favorite movie",
			"i am going hiking this weekend",
			"tell me about your family",
			"do you like pizza",
			"where do you live",
			"the weather is nice today",
			"my cat is missing",
			"i like science and technology",
		}

		for _, s := range sampleSentences {
			tokens := cleanTokenize(s)
			ids := make([]float32, len(tokens))
			for i, t := range tokens {
				ids[i] = float32(lookupVocab(t, model.SentenceVocab))
			}
			input := tensor.NewTensor([]int{1, len(ids)}, ids, false)
			emb, _ := model.Embedding.Forward(input)

			// We need to reach the MoELayer via the encoder
			// If it's a Hybrid encoder, we unwrap it
			var moelayer *moe.MoELayer
			if ml, ok := model.Encoder.(*moe.MoELayer); ok {
				moelayer = ml
			} else if hybrid, ok := model.Encoder.(*moe.HybridLLMGNNEncoder); ok {
				if ml, ok := hybrid.LLMEncoder.(*moe.MoELayer); ok {
					moelayer = ml
				}
			}

			if moelayer != nil {
				moelayer.SetMode(false) // Evaluation mode
				moelayer.Forward(emb)

				selected := moelayer.GetSelectedExperts() // [seqLen][K]
				for i, experts := range selected {
					token := tokens[i]
					for _, expIdx := range experts {
						specialization[expIdx][token]++
					}
				}
			}
		}

		for i := 0; i < len(layer.Experts); i++ {
			type kv struct {
				K string
				V int
			}
			var list []kv
			for k, v := range specialization[i] {
				list = append(list, kv{k, v})
			}
			sort.Slice(list, func(a, b int) bool {
				return list[a].V > list[b].V
			})

			top := 10
			if len(list) < top {
				top = len(list)
			}

			fmt.Printf("  Expert %d Top Tokens: ", i)
			for j := 0; j < top; j++ {
				fmt.Printf("%s (%d) ", list[j].K, list[j].V)
			}
			fmt.Println()
		}
	}
	fmt.Println("-----------------------------------------")
}

func cleanTokenize(text string) []string {
	text = strings.ToLower(text)
	var tokens []string
	var currentWord strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) || r == '\'' {
			currentWord.WriteRune(r)
		} else {
			// Save the word built so far
			if currentWord.Len() > 0 {
				tokens = append(tokens, currentWord.String())
				currentWord.Reset()
			}
			// If it's punctuation (and not whitespace), make it a token
			if unicode.IsPunct(r) || r == '?' || r == '!' {
				tokens = append(tokens, string(r))
			}
		}
	}
	// Catch trailing word
	if currentWord.Len() > 0 {
		tokens = append(tokens, currentWord.String())
	}
	return tokens
}

func InitializeXavier(p *tensor.Tensor) {
	if p == nil || len(p.Shape) == 0 {
		return
	}
	fanIn := p.Shape[0]
	fanOut := 0
	if len(p.Shape) > 1 {
		fanOut = p.Shape[1]
	} else {
		fanOut = fanIn
	}
	limit := float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
	for i := range p.Data {
		p.Data[i] = (float32(rand.Float64()) * 2 * limit) - limit
	}
}

func InitializeHeNormal(p *tensor.Tensor) {
	if p == nil || len(p.Shape) == 0 {
		return
	}
	fanIn := float64(p.Shape[0])
	scale := float32(math.Sqrt(2.0 / fanIn))
	for i := range p.Data {
		p.Data[i] = float32(rand.NormFloat64()) * scale
	}
}

func InitializeRouterGating(weights, biases *tensor.Tensor) {
	if weights == nil {
		return
	}
	scale := float32(0.5)
	for i := range weights.Data {
		weights.Data[i] = (float32(rand.Float64())*2.0 - 1.0) * scale
	}
	if biases != nil {
		for i := range biases.Data {
			if i == 3 && len(biases.Data) > 3 {
				biases.Data[i] = -0.2
			} else {
				biases.Data[i] = 0.1
			}
		}
	}
}

// InspectExpertStats calculates min, max, mean, and stdDev for all experts.
func InspectExpertStats(model *moe.IntentMoE) {
	fmt.Println("\n🔍 --- Expert Parameter Inspection ---")
	allLayers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		allLayers = append(allLayers, model.Decoder.OutputMoE)
	}

	for lIdx, layer := range allLayers {
		fmt.Printf("Layer %d:\n", lIdx)
		for i, expert := range layer.Experts {
			params := expert.Parameters()
			var sum float32
			var sumSq float32
			var minVal, maxVal float32 = 1e9, -1e9
			count := 0
			for _, p := range params {
				for _, v := range p.Data {
					sum += v
					sumSq += v * v
					if v < minVal {
						minVal = v
					}
					if v > maxVal {
						maxVal = v
					}
					count++
				}
			}
			if count == 0 {
				continue
			}
			mean := sum / float32(count)
			variance := (sumSq / float32(count)) - (mean * mean)
			if variance < 0 {
				variance = 0
			}
			std := float32(math.Sqrt(float64(variance)))

			status := "Healthy"
			if std < 0.01 {
				status = "⚠️  CLUMPED"
			}
			if math.IsNaN(float64(std)) {
				status = "❌ NAN"
			}

			fmt.Printf("  Expert %d: Range [%.3f, %.3f] StdDev %.4f (%s)\n", i, minVal, maxVal, std, status)
		}
	}
}

// InitializeOrthogonal fills param with an orthonormal row basis using the Gram-Schmidt
// process, then scales by gain. This is the recommended initializer for LSTM weight
// matrices and prevents vanishing/exploding gradients from the start.
func InitializeOrthogonal(param *tensor.Tensor, gain float32) {
	if param == nil || len(param.Shape) < 2 {
		InitializeXavier(param)
		return
	}

	// 1. Fill with random normal distribution
	for i := range param.Data {
		param.Data[i] = float32(rand.NormFloat64())
	}

	rows := param.Shape[0]
	cols := len(param.Data) / rows
	if cols == 0 {
		return
	}

	// 2. Gram-Schmidt orthogonalization
	for i := 0; i < rows; i++ {
		rowI := param.Data[i*cols : (i+1)*cols]
		for j := 0; j < i; j++ {
			rowJ := param.Data[j*cols : (j+1)*cols]
			// Compute dot product of rowI and rowJ
			dot := 0.0
			for k := range rowI {
				dot += float64(rowI[k] * rowJ[k])
			}
			// Subtract projection: rowI -= dot * rowJ
			for k := range rowI {
				rowI[k] -= float32(dot) * rowJ[k]
			}
		}
		// Normalize the row
		norm := 0.0
		for _, v := range rowI {
			norm += float64(v * v)
		}
		norm = math.Sqrt(norm + 1e-8)
		for k := range rowI {
			rowI[k] = (rowI[k] / float32(norm)) * gain
		}
	}
}

// InitializeLSTMBias zeros the bias tensor then sets the forget-gate chunk to 1.0.
// The LSTM gates are ordered [f, i, c, o]; the forget gate is the 2nd chunk (index hiddenSize..2*hiddenSize).
// Setting it to 1.0 at init allows the LSTM to pass gradients back unimpeded from the beginning.
func InitializeLSTMBias(param *tensor.Tensor, hiddenSize int) {
	if param == nil {
		return
	}
	// Zero everything first
	for i := range param.Data {
		param.Data[i] = 0
	}
	// Set Forget Gate bias to 1.0 — the 2nd gate in the [f, i, c, o] ordering
	forgetStart := hiddenSize
	forgetEnd := 2 * hiddenSize
	if forgetEnd > len(param.Data) {
		forgetEnd = len(param.Data)
	}
	for i := forgetStart; i < forgetEnd; i++ {
		param.Data[i] = 1.0
	}
	// For BiLSTM: if bias is the backward pass (second half), set that forget gate too
	if len(param.Data) == 8*hiddenSize {
		backwardForgetStart := 5 * hiddenSize
		backwardForgetEnd := 6 * hiddenSize
		for i := backwardForgetStart; i < backwardForgetEnd; i++ {
			param.Data[i] = 1.0
		}
	}
}

// isLSTMWeight returns true when the tensor looks like an LSTM gate weight matrix
// (i.e. 2D and the shorter dim is consistent with LSTM gate structure).
// The LSTMCell stores separate Wf/Wi/Wc/Wo matrices shaped [inputSize+hiddenSize, hiddenSize].
func isLSTMWeight(param *tensor.Tensor) bool {
	if param == nil || len(param.Shape) != 2 {
		return false
	}
	// Heuristic: LSTM weight matrices are square or have a larger first dimension
	// (inputSize + hiddenSize > hiddenSize). They are never 1-row (bias) tensors.
	return param.Shape[0] > 1 && param.Shape[1] > 1
}

// isLSTMBias returns true when the tensor looks like an LSTM gate bias vector.
// The LSTMCell stores separate 1D or [1, hiddenSize] bias tensors.
func isLSTMBias(param *tensor.Tensor, hiddenSize int) bool {
	if param == nil {
		return false
	}
	size := len(param.Data)
	// LSTM bias is exactly hiddenSize elements (one bias per gate, 4 gates total if stacked)
	// or hiddenSize for individual gate biases (Bf/Bi/Bc/Bo each [1, hiddenSize]).
	return size == hiddenSize || size == 4*hiddenSize || size == 8*hiddenSize
}

func ValidateChat(model *moe.IntentMoE, valPairs []struct{ Q, A, Intent string }, useGPU bool) float32 {
	// 1. Enter Eval Mode (Disable Dropout/Noise)
	for _, layer := range moe.ActiveLayers {
		layer.SetMode(false)
	}
	defer func() {
		for _, layer := range moe.ActiveLayers {
			layer.SetMode(true) // Re-enable training mode
		}
	}()

	var totalLoss float64
	var tokenCount int

	// Ensure we have the UNK ID for validation
	unkID := model.SentenceVocab.GetTokenID("UNK")
	if unkID == -1 {
		unkID = 0 // Fallback
	}

	for _, pair := range valPairs {
		// Tokenize
		qTokens := cleanTokenize(pair.Q)
		qIDs := make([]float32, max(1, len(qTokens)))
		for i, t := range qTokens {
			id := lookupVocab(t, model.SentenceVocab)
			qIDs[i] = float32(id)
		}
		if len(qTokens) == 0 {
			qIDs[0] = 0 // Pad
		}

		aTokens := cleanTokenize(pair.A)
		aIDs := make([]int, len(aTokens)+2)
		aIDs[0] = model.SentenceVocab.BosID
		for i, t := range aTokens {
			id := model.SentenceVocab.GetTokenID(t)
			if id == -1 || (id == 0 && t != "<pad>") {
				aIDs[i+1] = unkID
			} else {
				aIDs[i+1] = id
			}
		}
		aIDs[len(aIDs)-1] = model.SentenceVocab.EosID

		inputT := tensor.NewTensor([]int{1, len(qIDs)}, qIDs, false)
		// Validation targets (all except BOS)
		targets := aIDs[1:]

		// Forward pass (inference mode)
		targetTIDs := make([]float32, len(aIDs))
		for i, id := range aIDs {
			targetTIDs[i] = float32(id)
		}
		targetT := tensor.NewTensor([]int{1, len(targetTIDs)}, targetTIDs, false)

		if useGPU {
			inputT.ToGPU()
			targetT.ToGPU()
		}

		logits, _, err := model.Forward(0.0, inputT, targetT, nil)
		if err != nil || len(logits) == 0 {
			continue
		}

		// Recreate loss weights for validation context
		valWeights := make([]float32, model.SentenceVocab.Size())
		for i := range valWeights {
			valWeights[i] = 1.0
		}
		valWeights[unkID] = 0.01
		valWeights[model.SentenceVocab.PaddingTokenID] = 0.0

		loss, _ := WeightedCrossEntropy(logits[0], targets, valWeights, 0.0)
		totalLoss += float64(loss)
		tokenCount++

		DetachModel(model)
	}

	if tokenCount == 0 {
		return 0.0
	}

	avgLoss := totalLoss / float64(tokenCount)
	perplexity := float32(math.Exp(avgLoss))
	return perplexity
}

func MuteUNKToken(model *moe.IntentMoE, unkID int) {
	// Access the Output Layer weights
	outLayer := model.Decoder.OutputLayer
	if outLayer == nil {
		if model.Decoder.OutputMoE != nil {
			log.Println("ℹ️  MuteUNKToken: Decoder uses MoE output. UNK muting will be handled during generation via logit filtering.")
		}
		return
	}
	hiddenSize := outLayer.Weights.Shape[0]
	vocabSize := outLayer.Weights.Shape[1]

	log.Printf("🛠️  Muting UNK token (ID %d) in output layer of size %dx%d...", unkID, hiddenSize, vocabSize)

	if unkID >= vocabSize {
		log.Printf("⚠️ MuteUNKToken: UNK ID %d is out of bounds for vocab size %d", unkID, vocabSize)
		return
	}

	for h := 0; h < hiddenSize; h++ {
		// We set the weights for the UNK index to a very low number
		// -100.0 effectively removes its probability during softmax even with high variance logits
		idx := h*vocabSize + unkID
		if idx < len(outLayer.Weights.Data) {
			outLayer.Weights.Data[idx] = -100.0
		}
	}
}

// PunctuationWeights based on vocabulary mapping.
// Penalize tokens that are "easy" exits to prevent punctuation loops.
var PunctuationWeights = map[int]float32{
	46: 0.05, // Period '.' - extremely low weight
	44: 0.1,  // Comma ','
	63: 0.1,  // Question mark '?'
	32: 0.2,  // Space ' '
}

func WeightedCrossEntropy(logits *tensor.Tensor, targets []int, weights []float32, labelSmoothing float32) (float32, *tensor.Tensor) {
	// Flatten batch and sequence dimensions to handle 3D tensors [Batch, Seq, Vocab]
	vocabSize := logits.Shape[len(logits.Shape)-1]
	numClasses := vocabSize
	numRows := len(logits.Data) / numClasses
	grad := tensor.NewTensor(logits.Shape, make([]float32, len(logits.Data)), false)

	var totalLoss float32
	var count float32
	softmax := make([]float32, numClasses) // Pre-allocate softmax buffer

	for i := 0; i < numRows; i++ {
		if i >= len(targets) {
			break
		}
		targetID := targets[i]

		// 1. Skip if weight is 0 (Padding)
		if weights[targetID] == 0.0 {
			continue
		}

		offset := i * numClasses
		row := logits.Data[offset : offset+numClasses]

		// 2. Softmax
		maxLogit := row[0]
		for _, v := range row {
			if v > maxLogit {
				maxLogit = v
			}
		}
		var sumExp float32 = 0.0
		for j, v := range row {
			softmax[j] = float32(math.Exp(float64(v - maxLogit)))
			sumExp += softmax[j]
		}
		invSumExp := 1.0 / sumExp

		// 3. Loss
		prob := softmax[targetID] * invSumExp
		loss := -float32(math.Log(float64(prob + 1e-12)))

		currentWeight := weights[targetID]
		// Apply Punctuation Penalty
		if puncWeight, ok := PunctuationWeights[targetID]; ok {
			currentWeight *= puncWeight
		}

		totalLoss += loss * currentWeight
		count++

		// 4. Gradient
		for j := 0; j < numClasses; j++ {
			sj := softmax[j] * invSumExp
			var targetProb float32 = 0.0
			if j == targetID {
				targetProb = 1.0
			}
			if labelSmoothing > 0 {
				targetProb = targetProb*(1.0-labelSmoothing) + (labelSmoothing / float32(numClasses))
			}

			g := (sj - targetProb) * currentWeight
			grad.Data[offset+j] = g
		}
	}

	if count > 0 {
		avgLoss := totalLoss / count
		// Normalize gradients by the same count factor
		for i := range grad.Data {
			grad.Data[i] /= count
		}
		return avgLoss, grad
	}
	return 0, grad
}

func isCollapsed(tokens []string, vocab *mainvocab.Vocabulary) bool {
	if len(tokens) < 5 {
		return false
	}

	// Count occurrences of the top token
	counts := make(map[string]int)
	maxCount := 0
	for _, t := range tokens {
		counts[t]++
		if counts[t] > maxCount {
			maxCount = counts[t]
		}
	}

	// If more than 60% of the response is the same token (usually '.' or 'UNK')
	ratio := float64(maxCount) / float64(len(tokens))
	return ratio > 0.60
}

func GenerateTokens(model *moe.IntentMoE, input string, maxLen int, useGPU bool) []string {
	// Quiet version for circuit breaker
	tokens := cleanTokenize(input)
	if len(tokens) == 0 {
		return nil
	}
	inputIDs := make([]float32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	if useGPU {
		inputTensor.ToGPU()
	}

	emb, _ := model.Embedding.Forward(inputTensor)
	ctx, _ := model.Encoder.Forward(emb)
	if ctx == nil || ctx.Shape[1] == 0 {
		return nil
	}
	// Normalize context vector to match training scale
	ctx = model.NormalizeContextVector(ctx)

	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	lastIdx := ctx.Shape[1] - 1
	hiddenState, _ := ctx.Slice(1, lastIdx, lastIdx+1)
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})

	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float32, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	resIDs := []int{model.SentenceVocab.BosID}
	currentTokenID := model.SentenceVocab.BosID

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(currentTokenID)}, false)
		logits, nextHidden, nextCell, _, err := model.Decoder.DecodeStepWithExpert(inputT, hiddenState, cellState, ctx)
		if err != nil {
			break
		}
		hiddenState = nextHidden
		cellState = nextCell

		// Mute UNK and PAD
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		unkID := model.SentenceVocab.GetTokenID("UNK")
		if unkID != -1 {
			logits.Data[unkID] = -1e9
		}

		bestID, _ := moe.SampleFromLogits(logits, 0.1, 1, 1.0)
		if bestID == model.SentenceVocab.EosID {
			break
		}
		resIDs = append(resIDs, bestID)
		currentTokenID = bestID
	}

	var result []string
	for _, id := range resIDs[1:] {
		result = append(result, model.SentenceVocab.GetWord(id))
	}
	return result
}

func getContextVector(model *moe.IntentMoE, query string) *tensor.Tensor {
	tokens := cleanTokenize(query)
	ids := make([]float32, len(tokens))
	for i, t := range tokens {
		ids[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	if len(ids) == 0 {
		ids = []float32{0}
	}
	input := tensor.NewTensor([]int{1, len(ids)}, ids, false)
	emb, _ := model.Embedding.Forward(input)
	ctx, _ := model.Encoder.Forward(emb)
	// Return the sequence-mean for similarity checks
	v, _ := ctx.Mean(1)
	return v
}

func DiagnosticEncoderSimilarity(model *moe.IntentMoE, q1, q2 string) {
	v1 := getContextVector(model, q1)
	v2 := getContextVector(model, q2)

	// Calculate Cosine Similarity
	dot := 0.0
	for i := range v1.Data {
		dot += float64(v1.Data[i] * v2.Data[i])
	}
	norm1 := v1.L2Norm()
	norm2 := v2.L2Norm()
	similarity := dot / (float64(norm1) * float64(norm2))

	fmt.Printf("📊 Similarity ['%s' vs '%s']: %.4f\n", q1, q2, similarity)
	if similarity > 0.98 {
		fmt.Println("⚠️  CRITICAL: Vectors are too similar! The Encoder is collapsing.")
	} else {
		fmt.Println("✅ Encoder is successfully differentiating between these intents.")
	}
}

type Hypothesis struct {
	IDs   []int
	Score float32
}

func convertToFloat(ids []int) []float32 {
	f := make([]float32, len(ids))
	for i, v := range ids {
		f[i] = float32(v)
	}
	return f
}

func getTopK(t *tensor.Tensor, k int) ([]int, []float32) {
	indices := make([]int, len(t.Data))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return t.Data[indices[i]] > t.Data[indices[j]]
	})
	if k > len(indices) {
		k = len(indices)
	}
	topIndices := indices[:k]
	topValues := make([]float32, k)
	for i, idx := range topIndices {
		topValues[i] = t.Data[idx]
	}
	return topIndices, topValues
}

func BeamSearchDecode(model *moe.IntentMoE, ctx *tensor.Tensor, beamSize int, maxLen int) []int {
	const repetitionPenalty = 1.2 // 1.0 = no penalty, 2.0 = very aggressive
	const alpha = 0.7             // Length penalty coefficient

	beams := []Hypothesis{{IDs: []int{model.SentenceVocab.BosID}, Score: 0.0}}

	for step := 0; step < maxLen; step++ {
		candidates := []Hypothesis{}

		for _, b := range beams {
			if len(b.IDs) > 0 && b.IDs[len(b.IDs)-1] == model.SentenceVocab.EosID {
				candidates = append(candidates, b)
				continue
			}

			// Standard Forward Pass to get Logits
			inputT := tensor.NewTensor([]int{1, len(b.IDs)}, convertToFloat(b.IDs), false)
			// The context `ctx` from the encoder must be passed to the decoder at each step.
			logits, err := model.Decoder.Forward(ctx, inputT, 0.0)
			if err != nil {
				continue // Skip beam if it fails
			}

			// Get Log-Probabilities
			probs := tensor.Softmax(logits[len(logits)-1])
			logProbs := tensor.NewTensor(probs.Shape, make([]float32, len(probs.Data)), false)
			for i, p := range probs.Data {
				logProbs.Data[i] = float32(math.Log(float64(p) + 1e-9))
			}

			// --- APPLY PENALTY ---
			seen := make(map[int]bool)
			for _, id := range b.IDs {
				seen[id] = true
			}

			for i := 0; i < len(logProbs.Data); i++ {
				if seen[i] {
					if logProbs.Data[i] < 0 {
						logProbs.Data[i] *= repetitionPenalty
					} else {
						logProbs.Data[i] /= repetitionPenalty
					}
				}
			}

			// Prevent EOS at step 0 to avoid empty responses
			if step == 0 && model.SentenceVocab.EosID < len(logProbs.Data) {
				logProbs.Data[model.SentenceVocab.EosID] = -math.MaxFloat32
			}

			topKIndices, topKProbs := getTopK(logProbs, beamSize)

			for i := 0; i < len(topKIndices); i++ {
				newIDs := append([]int{}, b.IDs...)
				newIDs = append(newIDs, topKIndices[i])
				candidates = append(candidates, Hypothesis{
					IDs:   newIDs,
					Score: b.Score + topKProbs[i],
				})
			}
		}

		sort.Slice(candidates, func(i, j int) bool {
			// Normalize score by length^alpha
			scoreI := float64(candidates[i].Score) / math.Pow(float64(len(candidates[i].IDs)), alpha)
			scoreJ := float64(candidates[j].Score) / math.Pow(float64(len(candidates[j].IDs)), alpha)
			return scoreI > scoreJ
		})

		if len(candidates) > beamSize {
			beams = candidates[:beamSize]
		} else {
			beams = candidates
		}
	}
	if len(beams) == 0 {
		return []int{}
	}
	return beams[0].IDs
}

// BeamSearchDecodeFiltered is a modified version of BeamSearchDecode that filters out specified token IDs.
func BeamSearchDecodeFiltered(model *moe.IntentMoE, ctx *tensor.Tensor, beamSize int, maxLen int, filteredIDs []int) []int {
	const repetitionPenalty = 1.2 // 1.0 = no penalty, 2.0 = very aggressive
	const alpha = 0.7             // Length penalty coefficient
	const temperature = 1.5       // Flatten distribution to encourage non-UNK tokens

	beams := []Hypothesis{{IDs: []int{model.SentenceVocab.BosID}, Score: 0.0}}

	// Create a map for quick lookup of filtered IDs
	filterMap := make(map[int]bool)
	for _, id := range filteredIDs {
		filterMap[id] = true
	}

	for step := 0; step < maxLen; step++ {
		candidates := []Hypothesis{}

		for _, b := range beams {
			if len(b.IDs) > 0 && b.IDs[len(b.IDs)-1] == model.SentenceVocab.EosID {
				candidates = append(candidates, b)
				continue
			}

			// Standard Forward Pass to get Logits
			inputT := tensor.NewTensor([]int{1, len(b.IDs)}, convertToFloat(b.IDs), false)
			// The context `ctx` from the encoder must be passed to the decoder at each step.
			logits, err := model.Decoder.Forward(ctx, inputT, 0.0)
			if err != nil {
				continue // Skip beam if it fails
			}

			lastLogit := logits[len(logits)-1]
			ApplyTemperature(lastLogit.Data, float32(temperature))

			// Get Log-Probabilities
			probs := tensor.Softmax(lastLogit)
			logProbs := tensor.NewTensor(probs.Shape, make([]float32, len(probs.Data)), false)
			for i, p := range probs.Data {
				logProbs.Data[i] = float32(math.Log(float64(p) + 1e-9))
			}

			// --- APPLY PENALTY ---
			seen := make(map[int]bool)
			for _, id := range b.IDs {
				seen[id] = true
			}

			for i := 0; i < len(logProbs.Data); i++ {
				if seen[i] {
					if logProbs.Data[i] < 0 {
						logProbs.Data[i] *= float32(repetitionPenalty)
					} else {
						logProbs.Data[i] /= float32(repetitionPenalty)
					}
				}
			}

			// --- NEW: FILTER OUT UNWANTED TOKENS ---
			for id := range filterMap {
				if id < len(logProbs.Data) {
					logProbs.Data[id] = -3.4028235e38 // Set to a very low value to avoid being picked
				}
			}

			topKIndices, topKProbs := getTopK(logProbs, beamSize)

			for i := 0; i < len(topKIndices); i++ {
				newIDs := append([]int{}, b.IDs...)
				newIDs = append(newIDs, topKIndices[i])
				candidates = append(candidates, Hypothesis{
					IDs:   newIDs,
					Score: b.Score + topKProbs[i],
				})
			}
		}

		sort.Slice(candidates, func(i, j int) bool {
			// Normalize score by length^alpha
			scoreI := float64(candidates[i].Score) / math.Pow(float64(len(candidates[i].IDs)), alpha)
			scoreJ := float64(candidates[j].Score) / math.Pow(float64(len(candidates[j].IDs)), alpha)
			return scoreI > scoreJ
		})

		if len(candidates) > beamSize {
			beams = candidates[:beamSize]
		} else {
			beams = candidates
		}
	}
	if len(beams) == 0 {
		return []int{}
	}
	return beams[0].IDs
}

// ConversationTurn holds structured data about a single turn in the conversation.
type ConversationTurn struct {
	Input    []float32         // The averaged embedding of the user's input
	RawInput string            // The original user input text
	Intent   string            // The resolved intent (e.g., "create_handler")
	Entities map[string]string // Any extracted names/urls
	Response string            // The bot's response text
}

// ChatSession manages the conversation history for sliding window memory and context.
type ChatSession struct {
	History       []ConversationTurn
	MaxHistory    int // Number of exchanges to remember
	ContextVector []float32
	mu            sync.Mutex
}

// NewChatSession creates a new chat session.
func NewChatSession(maxHistory int, vectorSize int) *ChatSession {
	return &ChatSession{
		History:       make([]ConversationTurn, 0),
		MaxHistory:    maxHistory,
		ContextVector: make([]float32, vectorSize),
	}
}

// AddToHistory adds a new turn and updates the context vector.
func (s *ChatSession) AddToHistory(turn ConversationTurn) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.History) >= s.MaxHistory {
		s.History = s.History[1:] // Slide the window
	}
	s.History = append(s.History, turn)

	s.updateContextVector()
}

// updateContextVector computes a weighted average of recent embeddings.
func (s *ChatSession) updateContextVector() {
	if len(s.History) == 0 {
		return
	}

	// Simple weighted average: newer turns have more weight.
	for i := range s.ContextVector {
		s.ContextVector[i] = 0
	}

	var totalWeight float32 = 0.0
	for i, turn := range s.History {
		weight := float32(i + 1) // Simple linear weight
		for j, val := range turn.Input {
			if j < len(s.ContextVector) {
				s.ContextVector[j] += val * weight
			}
		}
		totalWeight += weight
	}

	if totalWeight > 0 {
		for i := range s.ContextVector {
			s.ContextVector[i] /= totalWeight
		}
	}
}

// GetContextVector returns a copy of the current context vector.
func (s *ChatSession) GetContextVector() []float32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	ctxCopy := make([]float32, len(s.ContextVector))
	copy(ctxCopy, s.ContextVector)
	return ctxCopy
}

// injectContextualClues provides a verbose explanation of the bot's reasoning
// based on the previous turn, enhancing the conversational feel.
func injectContextualClues(session *ChatSession) {
	if len(session.History) == 0 {
		return
	}

	lastTurn := session.History[len(session.History)-1]
	// NOTE: This relies on the Intent and Entities fields being populated by a proper
	// intent classifier and NER system. They are currently placeholders in this chat loop.
	if lastTurn.Intent != "chat_response" && len(lastTurn.Entities) > 0 {
		if name, ok := lastTurn.Entities["name"]; ok {
			// Example of verbose output.
			fmt.Printf("🤖 [Reasoning: Based on our previous step involving '%s', I will generate the next response.]\n", name)
		}
	}
}

func StartChat(model *moe.IntentMoE) {
	// Start Prometheus metrics server on port 2112
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		http.ListenAndServe(":2112", nil)
	}()

	fmt.Println("📈 Metrics available at http://localhost:2112/metrics")

	session := NewChatSession(3, model.Embedding.DimModel)
	// 1. Define the "Core Identity"
	// Keep it short so it doesn't eat up the RNN's memory (hidden state)
	const systemPrompt = "System: You are a friendly, helpful assistant. Tone: Kind."

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("\n--- 🤖 MoE Chatbot (Stateful Memory Enabled) ---")

	for {
		fmt.Print("\nYou: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" {
			break
		}

		// NEW: Inject contextual clues for verbose output.
		injectContextualClues(session)

		// TODO: Implement full prompt chaining.
		// A full implementation would involve parsing the current input to identify
		// the intent and any missing entities. If entities are missing (e.g., user says
		// "create a file" without a name), and the input is a continuation ("for it", "do it"),
		// the system would look at `session.History` for the last relevant entity and
		// inject it into the current command's context before execution. This requires
		// a dialogue manager and an integrated NER component.

		// Sentiment Analysis & Emotional Steering
		sentiment := GetSentimentScore(input)
		isApologetic := false
		if sentiment < -0.5 {
			isApologetic = true
			fmt.Println("🤖 [System Note: Bot is in 'Apologetic Mode']")
			// for _, layer := range moe.ActiveLayers {
			// 	// Manually add a bias to the router's logits for Expert 7
			// 	// This makes it 5x more likely to be chosen for this specific turn
			// 	if len(layer.RouterBias) > 7 {
			// 		layer.RouterBias[7] += 2.0
			// 	}
			// }
		}

		// 1. Tokenize and embed current input
		tokens := cleanTokenize(input)
		ids := make([]float32, len(tokens))
		avgInputEmbedding := make([]float32, model.Embedding.DimModel)
		tokenCount := 0
		for i, t := range tokens {
			id := lookupVocab(t, model.SentenceVocab)
			ids[i] = float32(id)
			// Get actual embedding from the model's weights for the history history
			if id >= 0 && id < model.Embedding.VocabSize {
				start := id * model.Embedding.DimModel
				vec := model.Embedding.Weight.Data[start : start+model.Embedding.DimModel]
				for d := 0; d < model.Embedding.DimModel; d++ {
					avgInputEmbedding[d] += vec[d]
				}
				tokenCount++
			}
		}
		if tokenCount > 0 {
			for d := range avgInputEmbedding {
				avgInputEmbedding[d] /= float32(tokenCount)
			}
		}

		// 2. Combine with context vector
		contextVector := session.GetContextVector()
		const lambda = 0.3 // Context decay factor
		if len(contextVector) == model.Embedding.DimModel {
			for i := 0; i < len(ids); i++ {
				// This is a conceptual change. The actual implementation
				// would modify the embedding tensor, not the IDs.
				// This logic is now handled in the Reply/StreamReply methods.
			}
		}

		// 3. Standard Inference
		inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

		// Inference (Eval Mode)
		for _, l := range moe.ActiveLayers {
			l.SetMode(false)
		}

		emb, _ := model.Embedding.Forward(inputT)
		ctx, _ := model.Encoder.Forward(emb)

		// 4. Beam Search Decoding
		// We use BeamSize 5, MaxLen 50, and Repetition Penalty 1.2
		outIDs := BeamSearchDecodeFiltered(model, ctx, 5, 50, []int{model.SentenceVocab.GetTokenID("UNK")})

		// 5. Convert IDs back to Words
		var response []string
		for _, id := range outIDs {
			word := model.SentenceVocab.GetWord(id)
			if word != "<s>" && word != "</s>" && word != "<pad>" {
				response = append(response, word)
			}
		}
		botResponse := strings.Join(response, " ")

		// 6. Print Routing Insight
		fmt.Printf("Bot [%s]: %s\n", getExpertPath(), botResponse)

		// 7. Save this turn to memory
		newTurn := ConversationTurn{
			Input:    avgInputEmbedding,
			RawInput: input,                   // Save original input
			Intent:   "chat_response",         // Placeholder, would be resolved by classifier
			Entities: make(map[string]string), // Placeholder
			Response: botResponse,
		}
		session.AddToHistory(newTurn)

		// Reset Emotional Steering
		if isApologetic {
			// for _, layer := range moe.ActiveLayers {
			// 	if len(layer.RouterBias) > 7 {
			// 		layer.RouterBias[7] -= 2.0
			// 	}
			// }
		}

		// Cleanup memory for the next turn
		DetachModel(model)
	}
}

// Helper to see which expert "won" the routing for the last token
func getExpertPath() string {
	var paths []string
	for i, layer := range moe.ActiveLayers {
		stats := layer.UtilizationStats()
		bestExp := 0
		maxCount := -1
		for expIdx, count := range stats {
			if count > maxCount {
				maxCount = count
				bestExp = expIdx
			}
		}
		paths = append(paths, fmt.Sprintf("L%d:E%d", i, bestExp))
	}
	return strings.Join(paths, " | ")
}

func GetSentimentScore(input string) float32 {
	// Basic word-list approach (or use a library like 'go-sentiment')
	posWords := map[string]bool{"happy": true, "great": true, "thanks": true, "love": true}
	negWords := map[string]bool{"angry": true, "bad": true, "hate": true, "error": true, "stop": true}

	var score float32 = 0.0
	tokens := strings.Fields(strings.ToLower(input))
	for _, t := range tokens {
		if posWords[t] {
			score += 1.0
		}
		if negWords[t] {
			score -= 1.0
		}
	}
	return score
}

// MoEChatBot encapsulates the chatbot state and logic for concurrency.
type MoEChatBot struct {
	model        *moe.IntentMoE
	session      *ChatSession
	systemPrompt string
}

func NewMoEChatBot(model *moe.IntentMoE) *MoEChatBot {
	return &MoEChatBot{
		model:        model,
		session:      NewChatSession(5, model.Embedding.DimModel),
		systemPrompt: "System: You are a friendly, helpful assistant. Tone: Kind.",
	}
}

var modelMutex sync.Mutex

func (b *MoEChatBot) Reply(input string) string {
	modelMutex.Lock()
	defer modelMutex.Unlock()

	// Sentiment Analysis & Emotional Steering
	sentiment := GetSentimentScore(input)
	isApologetic := false
	if sentiment < -0.5 {
		isApologetic = true
		// fmt.Println("🤖 [System Note: Bot is in 'Apologetic Mode']")
		// for _, layer := range moe.ActiveLayers {
		// 	if len(layer.RouterBias) > 7 {
		// 		layer.RouterBias[7] += 2.0
		// 	}
		// }
	}

	// 1. Tokenize and embed current input
	tokens := cleanTokenize(input)
	ids := make([]float32, len(tokens))
	avgInputEmbedding := make([]float32, b.model.Embedding.DimModel)
	tokenCount := 0
	for i, t := range tokens {
		id := lookupVocab(t, b.model.SentenceVocab)
		ids[i] = float32(id)
		if id >= 0 && id < b.model.Embedding.VocabSize {
			start := id * b.model.Embedding.DimModel
			vec := b.model.Embedding.Weight.Data[start : start+b.model.Embedding.DimModel]
			for d := 0; d < b.model.Embedding.DimModel; d++ {
				avgInputEmbedding[d] += vec[d]
			}
			tokenCount++
		}
	}
	if tokenCount > 0 {
		for d := range avgInputEmbedding {
			avgInputEmbedding[d] /= float32(tokenCount)
		}
	}

	inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

	// Inference (Eval Mode)
	for _, l := range moe.ActiveLayers {
		l.SetMode(false)
	}

	emb, _ := b.model.Embedding.Forward(inputT)

	// 2. Combine with context vector
	contextVector := b.session.GetContextVector()
	const lambda = 0.3 // Context decay factor
	if len(contextVector) == b.model.Embedding.DimModel {
		for i := 0; i < emb.Shape[1]; i++ { // For each token in sequence
			offset := i * b.model.Embedding.DimModel
			for j := 0; j < b.model.Embedding.DimModel; j++ {
				emb.Data[offset+j] += contextVector[j] * lambda
			}
		}
	}

	ctx, _ := b.model.Encoder.Forward(emb)

	// 4. Beam Search Decoding
	outIDs := BeamSearchDecodeFiltered(b.model, ctx, 5, 50, []int{b.model.SentenceVocab.GetTokenID("UNK")})

	// 5. Convert IDs back to Words
	var response []string
	for _, id := range outIDs {
		word := b.model.SentenceVocab.GetWord(id)
		if word != "<s>" && word != "</s>" && word != "<pad>" {
			response = append(response, word)
		}
	}
	botResponse := strings.Join(response, " ")

	// 7. Save this turn to memory
	newTurn := ConversationTurn{
		Input:    avgInputEmbedding,
		RawInput: input,
		Intent:   "chat_response",         // Placeholder
		Entities: make(map[string]string), // Placeholder
		Response: botResponse,
	}
	b.session.AddToHistory(newTurn)

	// Reset Emotional Steering
	if isApologetic {
		// for _, layer := range moe.ActiveLayers {
		// 	if len(layer.RouterBias) > 7 {
		// 		layer.RouterBias[7] -= 2.0
		// 	}
		// }
	}

	// Cleanup memory for the next turn
	DetachModel(b.model)

	return botResponse
}

// StreamReply returns a channel that emits words one by one.
func (b *MoEChatBot) StreamReply(userInput string) <-chan string {
	wordChan := make(chan string)

	go func() {
		defer close(wordChan)
		modelMutex.Lock()
		defer modelMutex.Unlock()

		// Sentiment Analysis & Emotional Steering
		sentiment := GetSentimentScore(userInput)
		isApologetic := false
		if sentiment < -0.5 {
			isApologetic = true
			// for _, layer := range moe.ActiveLayers {
			// 	if len(layer.RouterBias) > 7 {
			// 		layer.RouterBias[7] += 2.0
			// 	}
			// }
		}

		// 1. Tokenize and embed current input
		tokens := cleanTokenize(userInput)
		ids := make([]float32, len(tokens))
		avgInputEmbedding := make([]float32, b.model.Embedding.DimModel)
		tokenCount := 0
		for i, t := range tokens {
			id := lookupVocab(t, b.model.SentenceVocab)
			ids[i] = float32(id)
			if id >= 0 && id < b.model.Embedding.VocabSize {
				start := id * b.model.Embedding.DimModel
				vec := b.model.Embedding.Weight.Data[start : start+b.model.Embedding.DimModel]
				for d := 0; d < b.model.Embedding.DimModel; d++ {
					avgInputEmbedding[d] += vec[d]
				}
				tokenCount++
			}
		}
		if tokenCount > 0 {
			for d := range avgInputEmbedding {
				avgInputEmbedding[d] /= float32(tokenCount)
			}
		}

		// 2. Combine with context vector
		contextVector := b.session.GetContextVector()
		const lambda = 0.3 // Context decay factor
		if len(contextVector) == b.model.Embedding.DimModel {
			// This logic will be applied to the embedding tensor below
		}
		inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

		// 3. Encode (Eval mode)
		for _, l := range moe.ActiveLayers {
			l.SetMode(false)
		}
		emb, _ := b.model.Embedding.Forward(inputT)

		// Apply context vector to embeddings
		if len(contextVector) == b.model.Embedding.DimModel {
			for i := 0; i < emb.Shape[1]; i++ {
				offset := i * b.model.Embedding.DimModel
				for j := 0; j < b.model.Embedding.DimModel; j++ {
					emb.Data[offset+j] += contextVector[j] * lambda
				}
			}
		}
		b.model.Encoder.Forward(emb)

		// 4. Decode Loop
		currIDs := []float32{float32(b.model.SentenceVocab.BosID)}
		var responseTokens []string

		for i := 0; i < 50; i++ {
			startToken := time.Now()

			decInputT := tensor.NewTensor([]int{1, len(currIDs)}, currIDs, false)
			logits, _, _ := b.model.Forward(0.0, nil, decInputT)

			// 2. LOG THE EXPERTS: Capture the routing decisions for this token
			for layerIdx, layer := range moe.ActiveLayers {
				// We peek at the last routing decision made by the Gating Network
				selected := layer.GetSelectedExperts()
				if len(selected) > 0 {
					if last := selected[len(selected)-1]; len(last) > 0 {
						winner := last[0]
						expertUtilization.WithLabelValues(
							fmt.Sprintf("%d", layerIdx),
							fmt.Sprintf("%d", winner),
						).Inc()
					}
				}
			}

			// 3. Measure Latency
			tokenLatency.Observe(float64(time.Since(startToken).Milliseconds()))

			lastLogit := logits[len(logits)-1]
			nextID := b.sampleNextToken(lastLogit)

			if nextID == b.model.SentenceVocab.EosID {
				break
			}

			word := b.model.SentenceVocab.GetWord(nextID)
			if word != "<s>" && word != "</s>" && word != "<pad>" {
				wordChan <- word
				responseTokens = append(responseTokens, word)
			}
			currIDs = append(currIDs, float32(nextID))
		}

		// Save to history
		newTurn := ConversationTurn{
			Input:    avgInputEmbedding,
			RawInput: userInput,
			Intent:   "chat_response",         // Placeholder
			Entities: make(map[string]string), // Placeholder
			Response: strings.Join(responseTokens, " "),
		}
		b.session.AddToHistory(newTurn)

		// Reset Emotional Steering
		if isApologetic {
			// for _, layer := range moe.ActiveLayers {
			// 	if len(layer.RouterBias) > 7 {
			// 		layer.RouterBias[7] -= 2.0
			// 	}
			// }
		}

		// Cleanup
		DetachModel(b.model)
	}()

	return wordChan
}

func (b *MoEChatBot) sampleNextToken(logit *tensor.Tensor) int {
	probs := tensor.Softmax(logit)

	// Simple Greedy for now:
	var maxVal float32 = -1.0
	bestID := 0
	for i, v := range probs.Data {
		if v > maxVal {
			maxVal = v
			bestID = i
		}
	}
	return bestID
}

func StressTestBot(model *moe.IntentMoE) {
	const numUsers = 50
	const messagesPerUser = 5

	var wg sync.WaitGroup
	startTime := time.Now()

	fmt.Printf("🚀 Starting Stress Test: %d Users, %d Messages each...\n", numUsers, messagesPerUser)

	for i := 0; i < numUsers; i++ {
		wg.Add(1)
		go func(userID int) {
			defer wg.Done()

			// Each user gets their own "Stateful Bot" instance
			// sharing the SAME underlying Model weights
			userBot := NewMoEChatBot(model)

			for m := 0; m < messagesPerUser; m++ {
				msg := fmt.Sprintf("User %d message %d: How are the experts doing?", userID, m)

				startMsg := time.Now()
				_ = userBot.Reply(msg)
				elapsed := time.Since(startMsg)

				if userID == 0 && m == 0 {
					fmt.Printf("⏱️ Sample Latency (User 0): %v\n", elapsed)
				}
			}
		}(i)
	}

	wg.Wait()
	totalTime := time.Since(startTime)
	totalMsgs := numUsers * messagesPerUser
	fmt.Printf("\n--- 🏁 Stress Test Results ---\n")
	fmt.Printf("Total Time:      %v\n", totalTime)
	fmt.Printf("Total Messages:  %d\n", totalMsgs)
	fmt.Printf("Throughput:      %.2f msgs/sec\n", float64(totalMsgs)/totalTime.Seconds())
}

// ApplyTemperature scales the logits by the temperature.
// temperature > 1.0 makes the model more creative/risky
// temperature < 1.0 makes it more confident/repetitive
func ApplyTemperature(logits []float32, temperature float32) {
	if temperature == 1.0 {
		return
	}
	for i := range logits {
		logits[i] /= temperature
	}
}

// ExportUtilizationCSV writes current expert utilization to a persistent log.
func ExportUtilizationCSV(epoch, step int) {
	filename := "logs/moe_utilization.csv"
	var f *os.File
	var err error

	if _, statErr := os.Stat(filename); os.IsNotExist(statErr) {
		f, err = os.Create(filename)
		if err == nil {
			f.WriteString("epoch,step,layer,expert,count\n")
		}
	} else {
		f, err = os.OpenFile(filename, os.O_APPEND|os.O_WRONLY, 0644)
	}

	if err != nil {
		log.Printf("⚠️ Failed to open utilization log: %v", err)
		return
	}
	defer f.Close()

	for lIdx, layer := range moe.ActiveLayers {
		stats := layer.UtilizationStats()
		for eIdx, count := range stats {
			f.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n", epoch, step, lIdx, eIdx, count))
		}
	}
}

// CompareCheckpoints calculates the Mean Absolute Deviation (MAD) between two saved models.
func CompareCheckpoints(pathA, pathB string) {
	modelA, err := moe.LoadIntentMoEModelFromGOB(pathA)
	if err != nil {
		log.Printf("Error loading A: %v", err)
		return
	}
	modelB, err := moe.LoadIntentMoEModelFromGOB(pathB)
	if err != nil {
		log.Printf("Error loading B: %v", err)
		return
	}

	fmt.Println("📊 --- Weight Delta Analysis (MAD) ---")
	paramsA := modelA.Parameters()
	paramsB := modelB.Parameters()

	if len(paramsA) != len(paramsB) {
		fmt.Printf("❌ Param count mismatch: %d vs %d\n", len(paramsA), len(paramsB))
		return
	}

	for i := range paramsA {
		pA := paramsA[i]
		pB := paramsB[i]

		var totalDelta float32 = 0.0
		for j := range pA.Data {
			totalDelta += float32(math.Abs(float64(pA.Data[j] - pB.Data[j])))
		}
		avgDelta := totalDelta / float32(len(pA.Data))
		fmt.Printf("Param %d (Size %v) MAD: %e\n", i, pA.Shape, avgDelta)
	}
}

// OneCycle implements a cyclic learning rate policy.
type OneCycle struct {
	MaxLR       float32
	MinLR       float32
	TotalSteps  int
	CurrentStep int
}

// CalculateCosineDecay implements a cosine learning rate decay.
func CalculateCosineDecay(step int, totalSteps int, startLR float32, minLR float32) float32 {
	if step >= totalSteps {
		return minLR
	}
	// Calculate progress (0.0 to 1.0)
	progress := float64(step) / float64(totalSteps)

	// Cosine decay formula
	cosOut := 0.5 * (1.0 + math.Cos(math.Pi*progress))

	return minLR + (startLR-minLR)*float32(cosOut)
}

func (oc *OneCycle) GetNextLR() float32 {
	oc.CurrentStep++
	pct := float32(oc.CurrentStep) / float32(oc.TotalSteps)

	// Phase 1: Ramp up (first 30% of training)
	if pct < 0.3 {
		return oc.MinLR + (oc.MaxLR-oc.MinLR)*(pct/0.3)
	}

	// Phase 2: Cool down (remaining 70%)
	decayPct := (pct - 0.3) / 0.7
	return oc.MaxLR * float32(math.Max(0.01, 1.0-float64(decayPct)))
}

// MonitorGradientFlow compares the L2 norm of gradients across model layers.
func MonitorGradientFlow(model *moe.IntentMoE) {
	var layer0Norm, layer1Norm float32
	activeLayers := moe.ActiveLayers
	if len(activeLayers) < 2 {
		return
	}

	// Calculate Norm for Layer 0 Experts
	for _, expert := range activeLayers[0].Experts {
		for _, p := range expert.Parameters() {
			if p.Grad != nil {
				layer0Norm += p.Grad.L2Norm()
			}
		}
	}

	// Calculate Norm for Layer 1 Experts
	for _, expert := range activeLayers[1].Experts {
		for _, p := range expert.Parameters() {
			if p.Grad != nil {
				layer1Norm += p.Grad.L2Norm()
			}
		}
	}

	ratio := layer1Norm / (layer0Norm + 1e-10)
	fmt.Printf("📉 Grad Flow Ratio (L1/L0): %.4f | L1 Strength: %.6f\n", ratio, layer1Norm)

	if ratio < 0.1 && layer0Norm > 1e-5 {
		fmt.Println("⚠️  WARNING: Vanishing Gradients detected in Layer 1. Consider increasing Residual Weight.")
	}
}

// PrintImportanceMap visualizes which experts are handling which tokens.
func PrintImportanceMap(vocab *mainvocab.Vocabulary, tokens []string, layer *moe.MoELayer) {
	fmt.Println("\n🧠 --- Token -> Expert Mapping ---")
	// Use the stored selected experts for the last sequence in the batch
	selected := layer.GetSelectedExperts() // [TokenIdx][K]
	if len(selected) == 0 {
		return
	}

	for i, token := range tokens {
		if i >= len(selected) {
			break
		}
		expertID := selected[i][0] // Take the top-1 expert

		// Visual Heatmap (stub, as we don't have per-token softmax weights readily available here without re-running)
		fmt.Printf("%-15s [E%d] %s\n", token, expertID, strings.Repeat("█", 5))
	}
}

// getLR calculates the current learning rate for the training step using
// Cosine Decay with a 10% Warmup phase. This avoids early gradient explosions
// and ensures smooth convergence towards the end of the session.
func getLR(currentStep, totalSteps int, baseLR float32) float32 {
	warmupSteps := totalSteps / 10 // 10% warmup
	if warmupSteps == 0 {
		warmupSteps = 1
	}

	if currentStep < warmupSteps {
		// Linear Warmup: Scale LR from 0 up to the baseLR
		return baseLR * float32(currentStep) / float32(warmupSteps)
	}

	// Cosine Decay: Scale LR from baseLR down towards zero
	if currentStep >= totalSteps {
		return baseLR * 0.01 // Maintain a very small floor LR
	}

	// Calculate progression through the decay phase (0.0 to 1.0)
	progress := float64(currentStep-warmupSteps) / float64(totalSteps-warmupSteps)

	// standard cosine decay formula: 0.5 * baseLR * (1 + cos(pi * progress))
	return 0.5 * baseLR * (1 + float32(math.Cos(math.Pi*progress)))
}

// scoreSentenceHeuristic provides a numeric quality metric for validation sentences.
func scoreSentenceHeuristic(text string) float32 {
	if text == "[Still Silent]" || text == "" {
		return 0
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return 0
	}

	var score float32 = 10.0

	// 1. Length Penalty
	if len(words) < 5 {
		score -= 3.0
	}

	// 2. Repetition Penalty
	seen := make(map[string]int)
	for _, w := range words {
		seen[w]++
	}
	var repeatCost float32 = 0.0
	for _, count := range seen {
		if count > 1 {
			repeatCost += float32(math.Pow(float64(count), 1.5))
		}
	}
	score -= repeatCost * 0.5

	// 3. Variety Reward
	score += float32(len(seen)) * 0.2

	// 4. Punctuation Reward
	if strings.ContainsAny(text, ".!?") {
		score += 2.0
	}

	if score < 0 {
		score = 0
	}
	return score
}
