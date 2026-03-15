package chat

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	randv1 "math/rand"
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

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/golangast/gollemer/neural/moe"
	neuralnn "github.com/golangast/gollemer/neural/nn"
	mainvocab "github.com/golangast/gollemer/neural/nnu/vocab"
	"github.com/golangast/gollemer/neural/nnu/word2vec"
	"github.com/golangast/gollemer/neural/tensor"
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
	Mask      []float64      // To tell the loss function to ignore <pad>
	InputMask *tensor.Tensor // Attention mask (0.0 for real, -1e9 for pad)
}

func ValidateModelHealth(model *moe.IntentMoE) bool {
	fmt.Println("🔍 Performing Pre-Flight Health Check...")
	isHealthy := true

	for i, param := range model.Parameters() {
		maxVal := -1e18
		minVal := 1e18
		nanCount := 0
		
		for _, v := range param.Data {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				nanCount++
			}
			if v > maxVal { maxVal = v }
			if v < minVal { minVal = v }
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
		weightSum := 0.0
		for _, v := range layer.GatingNetwork.Linear.Weights.Data {
			weightSum += math.Abs(v)
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
        sum := 0.0
        for _, v := range w { sum += math.Abs(v) }
        if sum == 0 {
            fmt.Println("🚨 CRITICAL: Decoder Router is empty! Resetting weights...")
            // Initialize with small random values to break the E0 tie
            for i := range w { w[i] = (rand.Float64() - 0.5) * 0.1 }
        }
    }
}

func TrainChat(projectRoot string, rebalanceRequested bool) {
	fmt.Println("--- 🗣️  Training Chat Model ---")

	// Pre-declare helper to find MoE layers
	findMoELayers := func(m *moe.IntentMoE) []*moe.MoELayer {
		var layers []*moe.MoELayer
		if m == nil {
			return layers
		}
		// Check Encoder
		if ml, ok := m.Encoder.(*moe.MoELayer); ok {
			layers = append(layers, ml)
		} else if hybrid, ok := m.Encoder.(*moe.HybridLLMGNNEncoder); ok {
			if ml, ok := hybrid.LLMEncoder.(*moe.MoELayer); ok {
				layers = append(layers, ml)
			}
		}
		// Check Decoder's OutputMoE
		if m.Decoder != nil && m.Decoder.OutputMoE != nil {
			layers = append(layers, m.Decoder.OutputMoE)
		}
		return layers
	}

	// 1. Load Word2Vec for embeddings
	w2vPath := filepath.Join(projectRoot, "gob_models/word2vec_model.gob")
	w2v, err := word2vec.LoadModel(w2vPath)
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}
	fmt.Println("✅ Loaded Word2Vec model")

	// 2. Read human_chat.txt
	chatPath := filepath.Join(projectRoot, "trainingdata/human_chat.txt")
	file, err := os.Open(chatPath)
	if err != nil {
		log.Fatalf("Failed to open chat data: %v", err)
	}
	defer file.Close()

	// Reset ActiveLayers to ensure we track only the current model's layers and prevent leaks
	moe.ActiveLayers = nil

	var intentModel *moe.IntentMoE
	moePath := filepath.Join(projectRoot, "gob_models/moe_classification_model.gob")
	bestMoePath := filepath.Join(projectRoot, "gob_models/moe_classification_model_best.gob")

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
		} else {
			log.Printf("⚠️  Failed to load existing MoE model: %v. Starting fresh.", err)
		}
	}

	if intentModel == nil {
		log.Println("🚀 Initializing compact MoE (256d Encoder, 128d Decoder, 8 Encoder Experts, Linear Output)")
		// IMPORTANT: Pass nil for w2v here to avoid allocating a 60k+ word embedding table.
		// W2V weights are injected selectively after the SentenceVocab is built.
		intentModel, _ = moe.NewHybridIntentMoE(
			5000,    // small placeholder, will be resized later
			256,
			8,
			512, 512, 5000, 4, nil, // nil w2v = no huge allocation, 4 attention heads
		)
		// numExperts=1 → single Linear output layer (vastly cheaper than 8 MoE experts with 4112 output dim)
		// hiddenSize=256 must match encoder output dim for cross-attention; attentionHeads=4
		intentModel.Decoder, _ = moe.NewRNNDecoder(256, 5000, 256, 4, 1, 0.1, 1)

		// Decoder LSTM hiddenSize for bias initialization
		const decoderHiddenSize = 256

		log.Println("🛠️ Applying Smart Initialization (Orthogonal for LSTM weights, Xavier for rest)...")
		for _, param := range intentModel.Parameters() {
			if param == nil || len(param.Shape) == 0 {
				continue
			}
			switch {
			case isLSTMBias(param, decoderHiddenSize):
				// Forget-gate bias trick: set bias for forget gate to 1.0
				InitializeLSTMBias(param, decoderHiddenSize)
			case isLSTMWeight(param):
				// Orthogonal initialization for LSTM weight matrices (gain 0.8)
				InitializeOrthogonal(param, 0.8)
			default:
				InitializeXavier(param)
			}
		}
	}

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
		layer.CapacityFactor = 1.5          // Increased from 1.25
		layer.LoadBalancingWeight = 0.15   // Increased for more aggressive balance
		layer.RouterTemperature = 1.0     // Normal temperature
		layer.ExpertDropoutRate = 0.1     // Reduced dropout to prevent UNK collapse
		layer.SetMode(true)               // Enable training mode (noise)
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
	vocabPath := filepath.Join(projectRoot, "gob_models/seq2seq_output_vocab.gob")
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

	scanner := bufio.NewScanner(file)
	var chatPairs []struct{ Q, A string }

	var lastLine string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Extract content after "Human X:"
		content := line
		if idx := strings.Index(line, ":"); idx != -1 {
			content = strings.TrimSpace(line[idx+1:])
		}

		if lastLine != "" {
			chatPairs = append(chatPairs, struct{ Q, A string }{lastLine, content})
		}
		lastLine = content
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading chat data: %v", err)
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

	addedCount := 0
	for _, pair := range chatPairs {
		tokens := cleanTokenize(pair.Q)
		for _, t := range tokens {
			if _, ok := w2v.Vocabulary[t]; !ok {
				maxID++
				w2v.Vocabulary[t] = maxID
				// Initialize random vector
				vec := make([]float64, w2v.VectorSize)
				limit := math.Sqrt(6.0 / float64(w2v.VectorSize))
				for i := range vec {
					vec[i] = (rand.Float64() * 2 * limit) - limit
				}
				w2v.WordVectors[maxID] = vec
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
		log.Printf("Resizing Encoder Embedding from %d to %d", intentModel.Embedding.VocabSize, currentVocabSize)
		newEmb := neuralnn.NewEmbedding(currentVocabSize, 256)
		
		// Fill with Word2Vec weights where possible
		for i := 0; i < currentVocabSize; i++ {
			word := intentModel.SentenceVocab.GetWord(i)
			if id, ok := w2v.Vocabulary[word]; ok {
				vec := w2v.WordVectors[id]
				if i*256 < len(newEmb.Weight.Data) {
					copy(newEmb.Weight.Data[i*256:], vec)
				}
			}
		}
		intentModel.Embedding = newEmb
	}

	// Shuffle and Split
	rand.Shuffle(len(chatPairs), func(i, j int) { chatPairs[i], chatPairs[j] = chatPairs[j], chatPairs[i] })
	splitIdx := int(float64(len(chatPairs)) * 0.9)
	trainPairs := chatPairs[:splitIdx]
	valPairs := chatPairs[splitIdx:]

	// Use all available training data
	trainPairs = trainPairs[:min(len(trainPairs), len(trainPairs))]
	log.Printf("DEBUG: Training on %d pairs", len(trainPairs))

	fmt.Printf("Data Split: %d Training, %d Validation\n", len(trainPairs), len(valPairs))

	// Word2Vec coverage check
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

	// Resize Decoder if Vocabulary has grown (OR if architecture changed: LayerNorm needs to be 512)
	currentVocabSize = intentModel.SentenceVocab.Size()
	// Always force a resize if LayerNorm is the old size (256) to handle the architecture upgrade
	needsResize := intentModel.Decoder.Embedding.VocabSize != currentVocabSize
	if intentModel.Decoder.LayerNorm != nil && intentModel.Decoder.LayerNorm.NormalizedShape != (intentModel.Decoder.LSTM.HiddenSize + intentModel.Decoder.Embedding.DimModel) {
		needsResize = true
		log.Printf("Forcing decoder resize due to architecture upgrade (LayerNorm %d -> %d)", 
			intentModel.Decoder.LayerNorm.NormalizedShape, 
			intentModel.Decoder.LSTM.HiddenSize + intentModel.Decoder.Embedding.DimModel)
	}

	if needsResize {
		log.Printf("Resizing Decoder from Vocab %d to %d", intentModel.Decoder.Embedding.VocabSize, currentVocabSize)
		intentModel.Decoder.ResizeOutputLayer(currentVocabSize)
		intentModel.SentenceVocabSize = currentVocabSize
	}

	// Clear any stale state from the loaded model
	DetachModel(intentModel)

	// Curriculum sort
	sort.Slice(chatPairs, func(i, j int) bool {
		return len(cleanTokenize(chatPairs[i].A)) < len(cleanTokenize(chatPairs[j].A))
	})
	log.Println("🎓 Curriculum active: Training starts with shortest sentences.")

	// Split data
	trainCount := int(float64(len(chatPairs)) * 0.95)
	trainPairs = chatPairs[:trainCount]
	valPairs = chatPairs[trainCount:]

	// Use Iterator pattern to save memory and speed up training
	iterator := NewChatDataIterator(trainPairs, w2v, intentModel.SentenceVocab, unkID)

	// Free Word2Vec model from memory — it's no longer needed after the iterator and embeddings are built.
	log.Printf("🗑️  Freeing Word2Vec model from memory (%d vectors)...", w2v.VocabSize)
	w2v.WordVectors = nil
	w2v.Vocabulary = nil
	runtime.GC()
	debug.FreeOSMemory()
	log.Println("✅ Word2Vec memory freed.")

	// Training Loop
	epochs := 60
	const batchSize = 2 // Reduced from 4 to limit memory usage during backward pass
	
	// Optimizer initialization
	const initialLR = 5e-7
	optimizer := neuralnn.NewOptimizer(intentModel.Parameters(), initialLR, 1.0)

	// Learning rate settings
	baseLR := 1e-6 // Used as peak LR for cosine decay
	learningRate := baseLR
	peakLR := 0.0005 // Reduced from 0.05 − logs showed LR climbing to 0.04 which is unstable
	warmupSteps := 1000 // Extended from 300 for a gentler warmup on a sensitive model
	const weightDecay = 0.0001
	const unkPenalty = 5.0

	// Early stopping and metrics state
	patienceLimit := 10
	patienceCounter := 0
	globalStep := 0
	epochUtilization := make(map[string]int)
	epochLBLoss := 0.0
	expertStagnation := make(map[string]int)
	bestPPL := math.MaxFloat64
	lastEpochLoss := math.MaxFloat64
	plateauCount := 0

	// Cross-Entropy Weights setup
	lossWeights := make([]float64, intentModel.SentenceVocab.Size())
	for i := range lossWeights {
		lossWeights[i] = 1.0
	}
	lossWeights[unkID] = 0.01
	lossWeights[intentModel.SentenceVocab.PaddingTokenID] = 0.0

	// Set weight decay on the optimizer
	if opt, ok := optimizer.(*neuralnn.Adam); ok {
		opt.WeightDecay = weightDecay
	}

	fmt.Printf("Training on %d pairs for %d epochs (patience=%d)...\n", len(chatPairs), epochs, patienceLimit)

	// Mute the UNK token in the output layer to prevent it from being a "safe" prediction
	MuteUNKToken(intentModel, unkID)

	for epoch := 0; epoch < epochs; epoch++ {
		epochStartTime := time.Now()
		
		// Curriculum shuffle logic
		if epoch > 2 {
			rand.Shuffle(len(trainPairs), func(i, j int) { 
				trainPairs[i], trainPairs[j] = trainPairs[j], trainPairs[i] 
			})
			log.Println("🔄 Shuffled training data for this epoch")
		}

		// Force diverse routing for the first 2 epochs to break out of mode collapse
		if epoch < 2 {
			for _, layer := range moe.ActiveLayers {
				layer.RouterTemperature = 2.0 // Increased from 1.5
			}
			if epoch == 0 {
				log.Println("🔥 Forcing diverse routing for initial epochs (Temp 2.0)")
			}
		} else if epoch == 2 { // Reset to normal after warmup
			for _, layer := range moe.ActiveLayers {
				layer.RouterTemperature = 0.7
			}
			log.Println("🌡️ Resetting router temperature to normal (0.7).")
		}

		// (LB weight decay removed — LB weight is now small enough that it doesn't need decay)
		iterator.Reset()
		totalLoss := 0.0
		batches := 0
		epochLBLoss = 0.0
		// Reset aggregate utilization for this epoch
		for k := range epochUtilization {
			epochUtilization[k] = 0
		}

		// Prefetch tokenization: start a background goroutine that pre-produces
		// Batch structs into a buffered channel while the main goroutine
		// is busy with forward/backward. Buffer=64 keeps the main loop fed.
		prefetchCh := make(chan *Batch, 2) // Small buffer is enough for pointers
		go func() {
			for iterator.HasNext() {
				prefetchCh <- iterator.NextBatch(batchSize)
			}
			close(prefetchCh)
		}()

		for batch := range prefetchCh {
			if batch == nil || batch.Input == nil {
				continue
			}
			optimizer.ZeroGrad()

			// Memory Management
			if globalStep%100 == 0 {
				runtime.GC()
				debug.FreeOSMemory()
			}

			// 🛡️ ENCODER WEIGHT DAMPENING: Every 10 steps, clip any encoder
			// parameter whose L2 norm exceeds 50 back to norm 10.
			// This is a "soft fix" that runs in addition to the context-vector
			// normalization inside Forward(), providing a second line of defence.
			if globalStep%10 == 0 {
				for _, p := range intentModel.Encoder.Parameters() {
					norm := 0.0
					for _, v := range p.Data {
						norm += v * v
					}
					norm = math.Sqrt(norm)
					if norm > 50.0 {
						scale := 10.0 / norm
						for i := range p.Data {
							p.Data[i] *= scale
						}
					}
				}
			}

			inputTensor := batch.Input
			targetTensor := batch.Target
			if targetTensor.Shape[1] < 2 {
				continue
			}

			var lr float64
			if globalStep < warmupSteps {
				lr = (float64(globalStep) / float64(warmupSteps)) * peakLR
			} else {
				// User's specific cosine schedule
				lr = peakLR * math.Max(0.1, math.Cos(float64(globalStep)/2000.0))
			}
			optimizer.SetLearningRate(lr)
			learningRate = lr 

			// 🛑 CIRCUIT BREAKER: Every 500 Batches
			if globalStep%500 == 0 && globalStep > 0 {
				fmt.Println("\n🛑 Running Circuit Breaker Check...")
				check := GenerateTokens(intentModel, "how are you", 10)
				if isCollapsed(check, intentModel.SentenceVocab) {
					fmt.Println("🚨 Punctuation Loop Detected! Scaling LR down and shaking experts...")
					peakLR *= 0.2
					for _, layer := range moe.ActiveLayers {
						layer.RouterTemperature = 1.8 // Shake up experts
					}
					// Also reset optimizer moments for the next steps
					if opt, ok := optimizer.(*neuralnn.Adam); ok {
						opt.SetLearningRate(peakLR * 0.1)
					}
				} else {
					fmt.Println("✅ Diversity Check Passed.")
				}
			}

			// Forward
			// Teacher Forcing Schedule:
			var samplingProb float64
			if epoch >= 1 {
				// Start sampling even earlier
				samplingProb = math.Min(0.5, float64(epoch)*0.05)
			}
			logits, _, err := intentModel.Forward(samplingProb, inputTensor, targetTensor, batch.InputMask)
			if err != nil {
				log.Printf("Forward error: %v", err)
				continue
			}

			// Label Smoothing Schedule: 0.0 for first 5 epochs, then 0.1
			labelSmoothing := 0.1
			if epoch < 5 {
				labelSmoothing = 0.0
			}

			// Loss
			batchLoss := 0.0
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
				loss, grad := WeightedCrossEntropy(l, targets, lossWeights, labelSmoothing)
				if grad == nil {
					grad = tensor.NewTensor(l.Shape, make([]float64, len(l.Data)), false)
				}
				batchLoss = loss
				grads = []*tensor.Tensor{grad}
			} else {
				// Sequence of logits (Step-by-step path used for scheduled sampling)
				grads = make([]*tensor.Tensor, len(logits))
				currentBatchSize := targetTensor.Shape[0]
				seqLen := targetTensor.Shape[1]
				var stepLossTotal float64
				for t, logit := range logits {
					// Target for this step is AIDs[t+1]
					targets := make([]int, currentBatchSize)
					for b := 0; b < currentBatchSize; b++ {
						targets[b] = int(targetTensor.Data[b*seqLen+t+1])
					}
					l, g := WeightedCrossEntropy(logit, targets, lossWeights, labelSmoothing)
					if g == nil {
						g = tensor.NewTensor(logit.Shape, make([]float64, len(logit.Data)), false)
					}
					stepLossTotal += l
					grads[t] = g
				}
				// Normalize step-by-step path by volume (consistent with vectorized path)
				div := float64(len(logits))
				batchLoss = stepLossTotal / div
				for t := range grads {
					for i := range grads[t].Data {
						grads[t].Data[i] /= div
					}
				}
			}

			// Check for NaN/Inf loss immediately
			if math.IsNaN(batchLoss) || math.IsInf(batchLoss, 0) {
				log.Printf("⚠️ Batch %d loss is NaN/Inf. Skipping batch to prevent model corruption.", batches)
				continue
			}

			// Add MoE Load Balancing Loss and Router Z-Loss
			var currentBatchLB float64
			var batchZLoss float64
			for layerIdx, l := range moe.ActiveLayers {
				currentBatchLB += l.LoadBalancingLoss * l.LoadBalancingWeight
				batchZLoss += l.RouterZLoss // Coefficient 1e-4 is baked into CalculateRouterZLoss
				
				// Track aggregate utilization
				stats := l.UtilizationStats()
				for expIdx, count := range stats {
					key := fmt.Sprintf("%d:%d", layerIdx, expIdx)
					epochUtilization[key] += count
				}
			}
			// Factor in decoder MoE if present
			if intentModel.Decoder.OutputMoE != nil {
				currentBatchLB += intentModel.Decoder.OutputMoE.LoadBalancingLoss * intentModel.Decoder.OutputMoE.LoadBalancingWeight
				batchZLoss += intentModel.Decoder.OutputMoE.RouterZLoss
			}

			// Final total loss used for gradients
			totalGradLoss := batchLoss + currentBatchLB + batchZLoss
			batchLoss = totalGradLoss // Updated for logging
			epochLBLoss += currentBatchLB

			// Backward
			func() {
				defer func() {
					if r := recover(); r != nil {
						log.Printf("⚠️ Recovered from panic in Backward pass (batch skipped): %v", r)
					}
				}()
				if err := intentModel.Backward(grads...); err != nil {
					log.Printf("Backward failed: %v", err)
				} else {
					params := intentModel.Parameters()

					// AdamW handles weight decay now

					// 2. Gradient Norm Calculation & Clipping (The Shield)
					gradNorm := 0.0
					for _, p := range params {
						if p.Grad != nil {
							for _, g := range p.Grad.Data {
								gradNorm += g * g
							}
						}
					}
					gradNorm = math.Sqrt(gradNorm)

					const clipValue = 1.0 // Lowered from 5.0 for stability
					if gradNorm > clipValue {
						scale := clipValue / (gradNorm + 1e-6) // Safety epsilon
						for _, p := range params {
							if p.Grad != nil {
								for i := range p.Grad.Data {
									p.Grad.Data[i] *= scale
								}
							}
						}
						gradNorm = clipValue // Update for logging
					}

					if batches%20 == 0 {
						log.Printf("Batch %d | Grad Norm: %.4f | LR: %.6f", batches, gradNorm, learningRate)
					}

					optimizer.Step()
				}
			}()

			// Update metrics
			totalLoss += batchLoss
			batches++

			// Loss Protection
			if math.IsNaN(totalLoss) || math.IsInf(totalLoss, 0) {
				log.Fatalf("❌ Loss exploded to NaN/Inf at epoch %d, batch %d. Stopping training.", epoch, batches)
			}

			// Console Logging every 50 batches
			if batches%50 == 0 {
				elapsed := time.Since(epochStartTime).Seconds()
				batchesPerSec := float64(batches) / elapsed
				totalBatches := (len(chatPairs) + batchSize - 1) / batchSize
				log.Printf("Epoch %d, Batch %d/%d, Loss: %.4f (LB: %.4f, Step: %d, LR: %.7f) [%.2f b/s]", 
					epoch, batches, totalBatches, batchLoss, epochLBLoss/float64(batches), globalStep, learningRate, batchesPerSec)
			}

			// Memory safety: Clear computation graph every batch
			DetachModel(intentModel)
			// Trigger GC extremely frequently due to GNN adjacency matrices
			if batches%10 == 0 {
				runtime.GC()
			}
			globalStep++
		}
		// End of Epoch: log final batch count, print utilization, clear computation graph.
		if batches > 0 {
			log.Printf("Epoch %d, Batch %d/%d, AvgLoss: %.4f (Avg LB: %.4f, Step: %d)", epoch, batches, len(chatPairs), totalLoss/float64(batches), epochLBLoss/float64(batches), globalStep)
		}
		// Visualize Aggregate Utilization
		fmt.Printf("--- 📊 Aggregate Expert Utilization (Epoch %d) ---\n", epoch+1)
		for layerIdx, layer := range moe.ActiveLayers {
			fmt.Printf("Layer %d Expert Utilization (Capacity Factor: %.2f):\n", layerIdx, layer.CapacityFactor)
			totalTokens := 0
			for i := 0; i < len(layer.Experts); i++ {
				totalTokens += epochUtilization[fmt.Sprintf("%d:%d", layerIdx, i)]
			}
			for i := 0; i < len(layer.Experts); i++ {
				key := fmt.Sprintf("%d:%d", layerIdx, i)
				count := epochUtilization[key]
				percent := 0.0
				if totalTokens > 0 {
					percent = float64(count) / float64(totalTokens) * 100
				}
				bar := strings.Repeat("#", int(percent/2))
				fmt.Printf("  Expert %d: %8d (%5.1f%%) %s\n", i, count, percent, bar)
				// Expert Reset Logic: Reset weights if expert processes NO tokens for > 2 epochs
				if count == 0 {
					expertStagnation[key]++
				} else {
					expertStagnation[key] = 0
				}
				if expertStagnation[key] >= 2 {
					log.Printf("♻️ Expert %d in Layer %d has been stagnant for %d epochs. Resetting weights...", i, layerIdx, expertStagnation[key])
					for _, param := range layer.Experts[i].Parameters() {
						fanIn := param.Shape[0]
						fanOut := 0
						if len(param.Shape) > 1 {
							fanOut = param.Shape[1]
						}
						limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
						for j := range param.Data {
							param.Data[j] = (randv1.Float64() * 2 * limit) - limit
						}
					}
					expertStagnation[key] = 0
				}

				// Dominant Expert Freezing: If one expert does > 40% of the work, freeze it to force others to learn
				usage := percent / 100.0
				if usage > 0.40 {
					fmt.Printf("⚠️ Layer %d Expert %d is dominant (%.1f%%). Freezing for next epoch.\n", layerIdx, i, usage*100)
					layer.SetExpertFreeze(i, true)
				} else {
					layer.SetExpertFreeze(i, false) // Unfreeze if balanced
				}
			}
		}
		DetachModel(intentModel)
		avgLoss := 0.0
		if batches > 0 {
			avgLoss = totalLoss / float64(batches)
		}
		fmt.Printf("Epoch %d: Avg Loss %.4f in %.1fs\n", epoch+1, avgLoss, time.Since(epochStartTime).Seconds())
		
		// End of epoch memory cleanup
		runtime.GC()
		debug.FreeOSMemory()

		// Validation
		valPPL := ValidateChat(intentModel, valPairs, nil)
		log.Printf("📉 Validation Perplexity: %.2f", valPPL)

		// Check for Plateau
		if avgLoss >= lastEpochLoss*0.999 { // More sensitive plateau detection
			plateauCount++
		} else {
			plateauCount = 0
		}

		if plateauCount >= 2 && globalStep > 200 { 
			peakLR *= 0.5
			log.Printf("📉 Learning rate plateau detected. Reducing peakLR to %.8f", peakLR)
			plateauCount = 0
		}
		lastEpochLoss = avgLoss

		// Log History
		logEpochHistory(projectRoot, epoch+1, avgLoss, epochLBLoss/float64(batches), learningRate)

		// Sanity checks: diverse test prompts to monitor generation quality
		testPrompts := []struct{ label, prompt string }{
			{"greeting", "how are you"},
			{"weather", "is it raining today"},
			{"weekend", "any plans for the weekend"},
			{"feeling", "i feel tired today"},
			{"hobby", "what do you like to do"},
		}
		for _, tp := range testPrompts {
			tokens := cleanTokenize(tp.prompt)
			if len(tokens) == 0 {
				log.Printf("⚠️ Skip empty prompt: %s", tp.prompt)
				continue
			}
			runTestSentence(tp.label, tp.prompt, intentModel, nil)
		}

		// Save model at the end of each epoch, overwriting the main file.
		f, err := os.Create(moePath)
		if err != nil {
			log.Printf("⚠️  Failed to create model file for epoch %d: %v", epoch+1, err)
		} else {
			if err := moe.SaveIntentMoEModelToGOB(intentModel, f); err != nil {
				log.Printf("⚠️  Failed to save MoE model for epoch %d: %v", epoch+1, err)
			}
			f.Close()
			fmt.Printf("💾 Overwrote model checkpoint to %s after epoch %d\n", moePath, epoch+1)
		}

		// Save Best Model if loss improved, otherwise track patience
		if valPPL < bestPPL {
			bestPPL = valPPL
			patienceCounter = 0
			bf, err := os.Create(bestMoePath)
			if err != nil {
				log.Printf("⚠️  Failed to create best model file: %v", err)
			} else {
				if err := moe.SaveIntentMoEModelToGOB(intentModel, bf); err != nil {
					log.Printf("⚠️  Failed to save best MoE model: %v", err)
				}
				bf.Close()
				fmt.Printf("🏆 New Best Model! PPL: %.2f (Saved to %s)\n", bestPPL, bestMoePath)
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

	// Analyze expert specialization
	analyzeExpertSpecialization(intentModel, w2v)

	// 5. Save Vocabulary
	if err := intentModel.SentenceVocab.Save(vocabPath); err != nil {
		log.Printf("Failed to save vocabulary: %v", err)
	} else {
		fmt.Printf("💾 Saved vocabulary to %s\n", vocabPath)
	}
}

// StrictGenerate forces the model to generate a response without using UNK or PAD tokens.
func StrictGenerate(model *moe.IntentMoE, input string, w2v *word2vec.SimpleWord2Vec, maxLen int) string {
	// 1. Enter Eval Mode and set Router Temperature for stability
	oldTemps := make(map[*moe.MoELayer]float64)
	for _, layer := range moe.ActiveLayers {
		layer.SetMode(false)
		oldTemps[layer] = layer.RouterTemperature
		layer.RouterTemperature = 0.8 // Force decisive routing for stable generation
	}
	if model.Decoder.OutputMoE != nil {
		oldTemps[model.Decoder.OutputMoE] = model.Decoder.OutputMoE.RouterTemperature
		model.Decoder.OutputMoE.RouterTemperature = 0.1
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
	inputIDs := make([]float64, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float64(lookupVocab(t, model.SentenceVocab))
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
	if ctx.Shape[1] == 0 {
		log.Printf("StrictGenerate Error: encoder produced empty sequence")
		return ""
	}

	// 3. Prepare Decoder States
	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	
	// Initial hidden state from the last token of the context (condensed intent)
	lastIdx := ctx.Shape[1] - 1
	hiddenState, err := ctx.Slice(1, lastIdx, lastIdx+1)
	if err != nil {
		log.Printf("StrictGenerate Error (Initial Hidden): %v", err)
		return ""
	}
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})

	// Projection if needed (copying logic from Decoder.Forward)
	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float64, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)

	// 4. Start Sequence with <s> (BOS)
	resIDs := []int{model.SentenceVocab.BosID}
	currentTokenID := model.SentenceVocab.BosID

	// Track counts for frequency penalty during generation
	counts := make(map[int]int)

	var path []string
	ctxNorm := ctx.L2Norm()
	fmt.Printf("📡 Encoder Context Strength: %.4f\n", ctxNorm)

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float64{float64(currentTokenID)}, false)

		oldHiddenNorm := hiddenState.L2Norm()
		// 1. Step-by-step decoding with expert tracking
		logits, nextHidden, nextCell, expertID, err := model.Decoder.DecodeStepWithExpert(inputT, hiddenState, cellState, ctx)
		if err != nil {
			log.Printf("StrictGenerate Error (DecodeStep): %v", err)
			break
		}
		newHiddenNorm := nextHidden.L2Norm()
		fmt.Printf("🔍 Step %d | Context Influence: %.4f | Expert: E%d\n", i, math.Abs(newHiddenNorm-oldHiddenNorm), expertID)

		hiddenState = nextHidden
		cellState = nextCell

		// 5. Apply Repetition and Frequency Penalty
		// Repetition Penalty (multiplicative)
		moe.ApplyRepetitionPenalty(logits, resIDs, 1.2) // Lowered from 1.8
		
		// Frequency Penalty (additive)
		const frequencyPenalty = 0.5
		for id, count := range counts {
			if id < len(logits.Data) {
				logits.Data[id] -= frequencyPenalty * float64(count)
			}
		}

		// 6. Mute UNK and <pad>
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		unkID := model.SentenceVocab.GetTokenID("UNK")
		if unkID != -1 {
			logits.Data[unkID] = -1e9
		}

		// 7. Pick Best Word using Greedy Search for debugging
		bestID, err := moe.SampleFromLogits(logits, 0.1, 1, 1.0) // Temp 0.1, Top-K 1, Top-P 1.0
		if err != nil {
			log.Printf("StrictGenerate Error (Sampling): %v", err)
			break
		}

		if bestID == model.SentenceVocab.EosID {
			break
		}

		resIDs = append(resIDs, bestID)
		counts[bestID]++
		currentTokenID = bestID

		// Top-K Diagnostic summary
		probs := tensor.Softmax(logits)
		topIndices, topValues := getTopK(probs, 5)
		fmt.Printf("🔍 Step %d | Top Choices:\n", i)
		for k := 0; k < 5; k++ {
			word := model.SentenceVocab.GetWord(topIndices[k])
			fmt.Printf("   [%d] %-12s (%.2f%%)\n", k+1, word, topValues[k]*100)
		}

		// 2. Record the word and the expert that produced it
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

// runTestSentence is a helper to run test questions during training
func runTestSentence(label, input string, model *moe.IntentMoE, w2v *word2vec.SimpleWord2Vec) {
	// 1. Generate the response using Strict Mode
	// This ignores UNK (1) and &lt;pad&gt; (0)
	response := StrictGenerate(model, input, w2v, 20)

	// 2. Clean up the output
	response = strings.TrimSpace(response)
	if response == "" {
		response = "[Still Silent]"
	}

	// 3. Log the result
	log.Printf("🧪 Test '%s' (%s): %s", input, label, response)
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
func logEpochHistory(projectRoot string, epoch int, loss float64, lbLoss float64, lr float64) {
	historyPath := filepath.Join(projectRoot, "training_history.csv")
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
	pairs []struct{ Q, A string }
	w2v   *word2vec.SimpleWord2Vec
	vocab *mainvocab.Vocabulary
	unkID int
	idx   int
}

func NewChatDataIterator(pairs []struct{ Q, A string }, w2v *word2vec.SimpleWord2Vec, vocab *mainvocab.Vocabulary, unkID int) *ChatDataIterator {
	// Shuffle pairs for better training
	rand.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	return &ChatDataIterator{
		pairs: pairs,
		w2v:   w2v,
		vocab: vocab,
		unkID: unkID,
		idx:   0,
	}
}

func (it *ChatDataIterator) HasNext() bool {
	return it.idx < len(it.pairs)
}

func (it *ChatDataIterator) Next() (*tensor.Tensor, *tensor.Tensor) {
	pair := it.pairs[it.idx]
	it.idx++

	// Query Tokenization (now also using SentenceVocab!) — use cascaded lookup to reduce the 22% UNK rate
	qTokens := cleanTokenize(pair.Q)
	qIDs := make([]float64, len(qTokens))
	for i, t := range qTokens {
		qIDs[i] = float64(lookupVocab(t, it.vocab))
	}

	// Safety check: If a query is empty after cleaning,
	// provide a "padding" token so the model doesn't crash on a 0-width tensor.
	if len(qIDs) == 0 {
		qIDs = []float64{0} // Assuming 0 is pad or UNK
	}

	// Response Tokenization (SentenceVocab)
	aTokens := cleanTokenize(pair.A)
	aIDs := make([]float64, len(aTokens)+2) // +2 for BOS and EOS
	aIDs[0] = float64(it.vocab.BosID)
	idx := 1
	for _, t := range aTokens {
		id := it.vocab.GetTokenID(t)
		if id == -1 || (id == 0 && t != "<pad>") {
			aIDs[idx] = float64(it.unkID)
		} else {
			aIDs[idx] = float64(id)
		}
		idx++
	}
	aIDs[idx] = float64(it.vocab.EosID)

	inputTensor := tensor.NewTensor([]int{1, len(qIDs)}, qIDs, false)
	targetTensor := tensor.NewTensor([]int{1, len(aIDs)}, aIDs, false)
	return inputTensor, targetTensor
}

func (it *ChatDataIterator) NextBatch(batchSize int) *Batch {
	var inputs [][]float64
	var targets [][]float64
	maxIn, maxOut := 0, 0

	for i := 0; i < batchSize && it.HasNext(); i++ {
		inp, tgt := it.Next()
		// Sequence length constraint: strictly capped for memory and logical coherence
		if len(inp.Data) > 80 || len(tgt.Data) > 80 {
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

	paddedIn := make([]float64, len(inputs)*maxIn)
	paddedOut := make([]float64, len(targets)*maxOut)
	mask := make([]float64, len(targets)*maxOut)
	inputLogitMask := make([]float64, len(inputs)*maxIn) // For attention: 0 for real, -1e9 for pad
	padID := float64(it.vocab.PaddingTokenID)

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
			param.Grad = nil
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
}

func visualizeExpertUtilization() {
	for i, layer := range moe.ActiveLayers {
		fmt.Printf("Layer %d ", i)
		layer.VisualizeUtilization()
	}
}

func analyzeExpertSpecialization(model *moe.IntentMoE, w2v *word2vec.SimpleWord2Vec) {
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
			ids := make([]float64, len(tokens))
			for i, t := range tokens {
				if id, ok := w2v.Vocabulary[t]; ok {
					ids[i] = float64(id)
				} else if id, ok := w2v.Vocabulary["UNK"]; ok {
					ids[i] = float64(id)
				}
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
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
		fanOut = fanIn // Fallback for 1D bias vectors
	}

	// Xavier Limit: sqrt(6 / (fanIn + fanOut))
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))

	for i := range p.Data {
		// Uniform distribution between -limit and +limit
		p.Data[i] = (rand.Float64() * 2 * limit) - limit
	}
}

// InitializeOrthogonal fills param with an orthonormal row basis using the Gram-Schmidt
// process, then scales by gain. This is the recommended initializer for LSTM weight
// matrices and prevents vanishing/exploding gradients from the start.
func InitializeOrthogonal(param *tensor.Tensor, gain float64) {
	if param == nil || len(param.Shape) < 2 {
		InitializeXavier(param)
		return
	}

	// 1. Fill with random normal distribution
	for i := range param.Data {
		param.Data[i] = rand.NormFloat64()
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
				dot += rowI[k] * rowJ[k]
			}
			// Subtract projection: rowI -= dot * rowJ
			for k := range rowI {
				rowI[k] -= dot * rowJ[k]
			}
		}
		// Normalize the row
		norm := 0.0
		for _, v := range rowI {
			norm += v * v
		}
		norm = math.Sqrt(norm + 1e-8)
		for k := range rowI {
			rowI[k] = (rowI[k] / norm) * gain
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

func ValidateChat(model *moe.IntentMoE, valPairs []struct{ Q, A string }, w2v *word2vec.SimpleWord2Vec) float64 {
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
		qIDs := make([]float64, max(1, len(qTokens)))
		qMask := make([]float64, len(qIDs)) // [1, 1, 1, Seq]
		for i, t := range qTokens {
			id := lookupVocab(t, model.SentenceVocab)
			qIDs[i] = float64(id)
			qMask[i] = 0.0
		}
		if len(qTokens) == 0 {
			qIDs[0] = 0 // Pad
			qMask[0] = -1e9
		}

		aTokens := cleanTokenize(pair.A)
		aIDs := make([]float64, len(aTokens)+2)
		aIDs[0] = float64(model.SentenceVocab.BosID)
		for i, t := range aTokens {
			id := model.SentenceVocab.GetTokenID(t)
			if id == -1 || (id == 0 && t != "<pad>") {
				aIDs[i+1] = float64(unkID)
			} else {
				aIDs[i+1] = float64(id)
			}
		}
		aIDs[len(aIDs)-1] = float64(model.SentenceVocab.EosID)

		inputT := tensor.NewTensor([]int{1, len(qIDs)}, qIDs, false)
		targetT := tensor.NewTensor([]int{1, len(aIDs)}, aIDs, false)
		maskT := tensor.NewTensor([]int{1, 1, 1, len(qIDs)}, qMask, false)

		// Forward pass (samplingProb = 0)
		logits, _, err := model.Forward(0.0, inputT, targetT, maskT)
		if err != nil {
			log.Printf("ValidateChat: Forward failed: %v", err)
			continue
		}

		// Calculate Loss with proper targets for evaluation
		targetSeqLen := targetT.Shape[1] - 1
		targets := make([]int, targetSeqLen)
		for t := 0; t < targetSeqLen; t++ {
			targets[t] = int(targetT.Data[t+1])
		}

		// Recreate loss weights for validation context
		valWeights := make([]float64, model.SentenceVocab.Size())
		for i := range valWeights { valWeights[i] = 1.0 }
		valWeights[unkID] = 0.01
		valWeights[model.SentenceVocab.PaddingTokenID] = 0.0

		loss, _ := WeightedCrossEntropy(logits[0], targets, valWeights, 0.0)
		totalLoss += loss
		tokenCount++

		DetachModel(model)
	}

	if tokenCount == 0 {
		return 0.0
	}

	avgLoss := totalLoss / float64(tokenCount)
	perplexity := math.Exp(avgLoss)
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

func WeightedCrossEntropy(logits *tensor.Tensor, targets []int, weights []float64, labelSmoothing float64) (float64, *tensor.Tensor) {
	// Flatten batch and sequence dimensions to handle 3D tensors [Batch, Seq, Vocab]
	vocabSize := logits.Shape[len(logits.Shape)-1]
	numClasses := vocabSize
	numRows := len(logits.Data) / numClasses
	grad := tensor.NewTensor(logits.Shape, make([]float64, len(logits.Data)), false)

	var totalLoss float64
	var count float64
	softmax := make([]float64, numClasses) // Pre-allocate softmax buffer

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
		sumExp := 0.0
		for j, v := range row {
			softmax[j] = math.Exp(v - maxLogit)
			sumExp += softmax[j]
		}
		invSumExp := 1.0 / sumExp

		// 3. Loss
		prob := softmax[targetID] * invSumExp
		loss := -math.Log(prob + 1e-12)

		currentWeight := weights[targetID]

		totalLoss += loss * currentWeight
		count++

		// 4. Gradient
		for j := 0; j < numClasses; j++ {
			sj := softmax[j] * invSumExp
			targetProb := 0.0
			if j == targetID {
				targetProb = 1.0
			}
			if labelSmoothing > 0 {
				targetProb = targetProb*(1.0-labelSmoothing) + (labelSmoothing / float64(numClasses))
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

func GenerateTokens(model *moe.IntentMoE, input string, maxLen int) []string {
	// Quiet version for circuit breaker
	tokens := cleanTokenize(input)
	if len(tokens) == 0 {
		return nil
	}
	inputIDs := make([]float64, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float64(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	emb, _ := model.Embedding.Forward(inputTensor)
	ctx, _ := model.Encoder.Forward(emb)
	if ctx.Shape[1] == 0 {
		return nil
	}

	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	lastIdx := ctx.Shape[1] - 1
	hiddenState, _ := ctx.Slice(1, lastIdx, lastIdx+1)
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})

	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float64, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)

	resIDs := []int{model.SentenceVocab.BosID}
	currentTokenID := model.SentenceVocab.BosID

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float64{float64(currentTokenID)}, false)
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
	ids := make([]float64, len(tokens))
	for i, t := range tokens {
		ids[i] = float64(lookupVocab(t, model.SentenceVocab))
	}
	if len(ids) == 0 {
		ids = []float64{0}
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
		dot += v1.Data[i] * v2.Data[i]
	}
	norm1 := v1.L2Norm()
	norm2 := v2.L2Norm()
	similarity := dot / (norm1 * norm2)

	fmt.Printf("📊 Similarity ['%s' vs '%s']: %.4f\n", q1, q2, similarity)
	if similarity > 0.98 {
		fmt.Println("⚠️  CRITICAL: Vectors are too similar! The Encoder is collapsing.")
	} else {
		fmt.Println("✅ Encoder is successfully differentiating between these intents.")
	}
}

type Hypothesis struct {
	IDs   []int
	Score float64
}

func convertToFloat(ids []int) []float64 {
	f := make([]float64, len(ids))
	for i, v := range ids {
		f[i] = float64(v)
	}
	return f
}

func getTopK(t *tensor.Tensor, k int) ([]int, []float64) {
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
	topValues := make([]float64, k)
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
			logProbs := tensor.NewTensor(probs.Shape, make([]float64, len(probs.Data)), false)
			for i, p := range probs.Data {
				logProbs.Data[i] = math.Log(p + 1e-9)
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
				logProbs.Data[model.SentenceVocab.EosID] = -math.MaxFloat64
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
			scoreI := candidates[i].Score / math.Pow(float64(len(candidates[i].IDs)), alpha)
			scoreJ := candidates[j].Score / math.Pow(float64(len(candidates[j].IDs)), alpha)
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
			ApplyTemperature(lastLogit.Data, temperature)

			// Get Log-Probabilities
			probs := tensor.Softmax(lastLogit)
			logProbs := tensor.NewTensor(probs.Shape, make([]float64, len(probs.Data)), false)
			for i, p := range probs.Data {
				logProbs.Data[i] = math.Log(p + 1e-9)
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

			// --- NEW: FILTER OUT UNWANTED TOKENS ---
			for id := range filterMap {
				if id < len(logProbs.Data) {
					logProbs.Data[id] = -math.MaxFloat64 // Set to a very low value to avoid being picked
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
			scoreI := candidates[i].Score / math.Pow(float64(len(candidates[i].IDs)), alpha)
			scoreJ := candidates[j].Score / math.Pow(float64(len(candidates[j].IDs)), alpha)
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
	Input    []float64         // The averaged embedding of the user's input
	RawInput string            // The original user input text
	Intent   string            // The resolved intent (e.g., "create_handler")
	Entities map[string]string // Any extracted names/urls
	Response string            // The bot's response text
}

// ChatSession manages the conversation history for sliding window memory and context.
type ChatSession struct {
	History       []ConversationTurn
	MaxHistory    int // Number of exchanges to remember
	ContextVector []float64
	mu            sync.Mutex
}

// NewChatSession creates a new chat session.
func NewChatSession(maxHistory int, vectorSize int) *ChatSession {
	return &ChatSession{
		History:       make([]ConversationTurn, 0),
		MaxHistory:    maxHistory,
		ContextVector: make([]float64, vectorSize),
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

	totalWeight := 0.0
	for i, turn := range s.History {
		weight := float64(i + 1) // Simple linear weight
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
func (s *ChatSession) GetContextVector() []float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	ctxCopy := make([]float64, len(s.ContextVector))
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

func StartChat(model *moe.IntentMoE, w2v *word2vec.SimpleWord2Vec) {
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
		ids := make([]float64, len(tokens))
		avgInputEmbedding := make([]float64, model.Embedding.DimModel)
		tokenCount := 0
		for i, t := range tokens {
			id := lookupW2V(t, w2v)
			ids[i] = id
			if vec, ok := w2v.WordVectors[int(id)]; ok {
				for d := 0; d < model.Embedding.DimModel; d++ {
					avgInputEmbedding[d] += vec[d]
				}
				tokenCount++
			}
		}
		if tokenCount > 0 {
			for d := range avgInputEmbedding {
				avgInputEmbedding[d] /= float64(tokenCount)
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
			RawInput: input,           // Save original input
			Intent:   "chat_response", // Placeholder, would be resolved by classifier
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

func GetSentimentScore(input string) float64 {
	// Basic word-list approach (or use a library like 'go-sentiment')
	posWords := map[string]bool{"happy": true, "great": true, "thanks": true, "love": true}
	negWords := map[string]bool{"angry": true, "bad": true, "hate": true, "error": true, "stop": true}

	score := 0.0
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
	w2v          *word2vec.SimpleWord2Vec
	session      *ChatSession
	systemPrompt string
}

func NewMoEChatBot(model *moe.IntentMoE, w2v *word2vec.SimpleWord2Vec) *MoEChatBot {
	return &MoEChatBot{
		model:        model,
		w2v:          w2v,
		session:      NewChatSession(3, model.Embedding.DimModel),
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
	ids := make([]float64, len(tokens))
	avgInputEmbedding := make([]float64, b.model.Embedding.DimModel)
	tokenCount := 0
	for i, t := range tokens {
		id := lookupW2V(t, b.w2v)
		ids[i] = id
		if vec, ok := b.w2v.WordVectors[int(id)]; ok {
			for d := 0; d < b.model.Embedding.DimModel; d++ {
				avgInputEmbedding[d] += vec[d]
			}
			tokenCount++
		}
	}
	if tokenCount > 0 {
		for d := range avgInputEmbedding {
			avgInputEmbedding[d] /= float64(tokenCount)
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
		ids := make([]float64, len(tokens))
	avgInputEmbedding := make([]float64, b.model.Embedding.DimModel)
	tokenCount := 0
		for i, t := range tokens {
		id := lookupW2V(t, b.w2v)
		ids[i] = id
		if vec, ok := b.w2v.WordVectors[int(id)]; ok {
			for d := 0; d < b.model.Embedding.DimModel; d++ {
				avgInputEmbedding[d] += vec[d]
			}
			tokenCount++
		}
	}
	if tokenCount > 0 {
		for d := range avgInputEmbedding {
			avgInputEmbedding[d] /= float64(tokenCount)
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
		currIDs := []float64{float64(b.model.SentenceVocab.BosID)}
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
			currIDs = append(currIDs, float64(nextID))
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
	maxVal := -1.0
	bestID := 0
	for i, v := range probs.Data {
		if v > maxVal {
			maxVal = v
			bestID = i
		}
	}
	return bestID
}

func StressTestBot(model *moe.IntentMoE, w2v *word2vec.SimpleWord2Vec) {
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
			userBot := NewMoEChatBot(model, w2v)

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
func ApplyTemperature(logits []float64, temperature float64) {
	if temperature == 1.0 {
		return
	}
	for i := range logits {
		logits[i] /= temperature
	}
}
