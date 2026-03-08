package chat

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	randv1 "math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/golangast/gollemer/neural/moe"
	neuralnn "github.com/golangast/gollemer/neural/nn"
	mainvocab "github.com/golangast/gollemer/neural/nnu/vocab"
	"github.com/golangast/gollemer/neural/nnu/word2vec"
	"github.com/golangast/gollemer/neural/tensor"
)

// batchItem holds a pre-tokenized training pair ready for the forward pass.
type batchItem struct {
	input  *tensor.Tensor
	target *tensor.Tensor
}

func TrainChat(projectRoot string) {
	fmt.Println("--- 🗣️  Training Chat Model ---")

	// Pre-declare helper to find MoE layers
	var findMoELayers func(m *moe.IntentMoE) []*moe.MoELayer
	findMoELayers = func(m *moe.IntentMoE) []*moe.MoELayer {
		var layers []*moe.MoELayer
		if m == nil {
			return layers
		}
		// Check Encoder
		if layer, ok := m.Encoder.(*moe.MoELayer); ok {
			layers = append(layers, layer)
		} else if hybrid, ok := m.Encoder.(*moe.HybridLLMGNNEncoder); ok {
			if layer, ok := hybrid.LLMEncoder.(*moe.MoELayer); ok {
				layers = append(layers, layer)
			}
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

	// 3. Load or Initialize MoE Model
	vocabSize := 2000
	embeddingDim := 256 // Increased Capacity
	if w2v != nil {
		vocabSize = w2v.VocabSize
		// embeddingDim = w2v.VectorSize // Keep 128 even if w2v is 64, we will pad/randomly init the rest
	}

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
			// Update embeddingDim to match loaded model to prevent index errors
			embeddingDim = intentModel.Embedding.DimModel
		} else {
			log.Printf("⚠️  Failed to load existing MoE model: %v. Starting fresh.", err)
		}
	}

	if intentModel == nil {
		var err error
		// Increased Embedding to 256, Hidden to 512
		intentModel, err = moe.NewHybridIntentMoE(
			vocabSize,
			256,      // embeddingDim: doubled for better semantic resolution
			8,        // numExperts: doubled to allow more specialized "topic" experts
			512, 512, // hidden layers
			10000,
			8, // maxAttentionHeads: increased for complex query parsing
			w2v,
		)
		if err != nil {
			log.Fatalf("Failed to initialize MoE model: %v", err)
		}

		// Upgrade decoder: 512 Hidden Size, 2 Layers, higher Dropout
		// This gives the RNN enough "memory" to track sentence state over 10+ tokens.
		intentModel.Decoder, _ = moe.NewRNNDecoder(256, 10000, 512, 8, 2, 0.2)

		log.Println("🚀 Initialized High-Capacity MoE (256d Embedding, 512d Hidden, 8 Experts)")

		log.Println("🛠️ Applying Xavier Initialization to all parameters...")
		for _, param := range intentModel.Parameters() {
			InitializeXavier(param)
		}
	}

	// Adjust MoE settings for training to prevent token dropping and encourage diversity
	for _, layer := range moe.ActiveLayers {
		layer.CapacityFactor = 1.0         // Tighter routing: forces experts to specialize
		layer.LoadBalancingWeight = 0.01   // Small regularizer — must NOT dominate loss
		layer.RouterTemperature = 0.7     // Sharper routing encourages expert specialization
		layer.ExpertDropoutRate = 0.25    // Higher dropout forces expert diversification
		layer.SetMode(true)               // Enable training mode (noise)
	}
	log.Println("🔧 Adjusted MoE: Capacity=1.0, LBWeight=0.01, Temp=0.7, Dropout=0.25")

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

	// Shuffle and Split
	rand.Shuffle(len(chatPairs), func(i, j int) { chatPairs[i], chatPairs[j] = chatPairs[j], chatPairs[i] })
	splitIdx := int(float64(len(chatPairs)) * 0.9)
	trainPairs := chatPairs[:splitIdx]
	valPairs := chatPairs[splitIdx:]
	fmt.Printf("Data Split: %d Training, %d Validation\n", len(trainPairs), len(valPairs))

	// Check W2V coverage
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

	// Fix Word2Vec Misses: Re-initialize UNK embedding with noise
	if unkID, ok := w2v.Vocabulary["UNK"]; ok {
		log.Printf("Re-initializing UNK token (ID %d) with random noise...", unkID)
		limit := math.Sqrt(6.0 / float64(vocabSize+embeddingDim))
		start := unkID * embeddingDim
		end := start + embeddingDim
		for j := start; j < end; j++ {
			intentModel.Embedding.Weight.Data[j] = (rand.Float64() * 2 * limit) - limit
		}
	}

	// Ensure SentenceVocab exists and populate it
	if intentModel.SentenceVocab == nil {
		intentModel.SentenceVocab = mainvocab.NewVocabulary()
		intentModel.SentenceVocab.AddToken("<pad>")
		intentModel.SentenceVocab.PaddingTokenID = intentModel.SentenceVocab.GetTokenID("<pad>")
		intentModel.SentenceVocab.AddToken("<s>")
		intentModel.SentenceVocab.AddToken("</s>")
		intentModel.SentenceVocab.BosID = intentModel.SentenceVocab.GetTokenID("<s>")
		intentModel.SentenceVocab.EosID = intentModel.SentenceVocab.GetTokenID("</s>")
	}

	// Debug: Print some vocab info to verify initialization
	log.Printf("SentenceVocab Size: %d", intentModel.SentenceVocab.Size())
	log.Printf("Token 0: %s", intentModel.SentenceVocab.GetWord(0))
	log.Printf("Token 1: %s", intentModel.SentenceVocab.GetWord(1))

	// Add response tokens to vocab with frequency filtering to remove single-occurrence garbage
	tokenCounts := make(map[string]int)
	for _, pair := range chatPairs {
		for _, t := range cleanTokenize(pair.A) {
			tokenCounts[t]++
		}
	}

	prunedCount := 0
	for t, count := range tokenCounts {
		// Only add tokens that appear at least twice OR are in Word2Vec
		_, inW2V := w2v.Vocabulary[t]
		if count >= 3 || inW2V {
			intentModel.SentenceVocab.AddToken(t)
		} else {
			prunedCount++
		}
	}
	log.Printf("✂️ Vocabulary Pruning: Kept %d tokens, Pruned %d single-occurrence noisy tokens.", intentModel.SentenceVocab.Size(), prunedCount)

	// Resize Decoder if Vocabulary has grown
	currentVocabSize := intentModel.SentenceVocab.Size()
	if intentModel.Decoder.Embedding.VocabSize != currentVocabSize {
		log.Printf("Resizing Decoder Embedding/Output from %d to %d", intentModel.Decoder.Embedding.VocabSize, currentVocabSize)

		// 1. Resize Embedding
		oldEmbedding := intentModel.Decoder.Embedding
		newEmbedding := neuralnn.NewEmbedding(currentVocabSize, oldEmbedding.DimModel)

		// Copy existing weights to preserve prior learning
		minVocab := min(oldEmbedding.VocabSize, currentVocabSize)
		dim := oldEmbedding.DimModel
		for i := 0; i < minVocab; i++ {
			copy(newEmbedding.Weight.Data[i*dim:(i+1)*dim], oldEmbedding.Weight.Data[i*dim:(i+1)*dim])
		}
		intentModel.Decoder.Embedding = newEmbedding

		// 2. Resize Output Layer
		newOutput, err := neuralnn.NewLinear(intentModel.Decoder.LSTM.HiddenSize, currentVocabSize)
		if err != nil {
			log.Fatalf("Failed to resize output layer: %v", err)
		}

		// Initialize with small variance to prevent loss explosion
		limit := math.Sqrt(6.0 / float64(intentModel.Decoder.LSTM.HiddenSize+currentVocabSize))
		for i := range newOutput.Weights.Data {
			newOutput.Weights.Data[i] = (rand.Float64() * 2 * limit) - limit
		}

		intentModel.Decoder.OutputLayer = newOutput
		intentModel.Decoder.OutputVocabSize = currentVocabSize
		intentModel.SentenceVocabSize = currentVocabSize
	}

	// Clear any stale state from the loaded model
	DetachModel(intentModel)

	// Use Iterator pattern to save memory and speed up training
	iterator := NewChatDataIterator(trainPairs, w2v, intentModel.SentenceVocab)

	// Training Loop
	epochs := 30 // Increased epochs for better convergence and generalization
	// 1e-4 avoids the gradient explosion zone seen at 4e-4 after epoch 2.
	// Adam's adaptive moments handle the rest; we apply hard decay + clipping on top.
	learningRate := 0.0001
	const minLR = 5e-6    // Floor to prevent LR from collapsing
	optimizer := neuralnn.NewOptimizer(intentModel.Parameters(), learningRate, 1.0) // clipValue=1.0 for tighter gradient control

	// Early stopping state
	patienceLimit := 4    // Stop if no improvement for this many epochs
	patienceCounter := 0  // How many consecutive epochs without improvement

	// Seed v1 rand for MoE noise
	randv1.Seed(time.Now().UnixNano())

	globalStep := 0
	warmupSteps := 500

	// Track expert metrics over the epoch
	epochUtilization := make(map[string]int)
	epochLBLoss := 0.0
	expertStagnation := make(map[string]int)
	bestPPL := math.MaxFloat64

	// State variables before the Epoch loop
	lastEpochLoss := math.MaxFloat64
	plateauCount := 0

	fmt.Printf("Training on %d pairs for %d epochs (patience=%d)...\n", len(chatPairs), epochs, patienceLimit)

	for epoch := 0; epoch < epochs; epoch++ {
		epochStartTime := time.Now()
		// Increase Router Temperature after Epoch 10 to force specialization
		if epoch >= 10 {
			for _, layer := range moe.ActiveLayers {
				layer.RouterTemperature = 1.2
			}
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
		// batchItem structs into a buffered channel while the main goroutine
		// is busy with forward/backward. Buffer=64 keeps the main loop fed.
		prefetchCh := make(chan batchItem, 64)
		go func() {
			for iterator.HasNext() {
				inp, tgt := iterator.Next()
				prefetchCh <- batchItem{inp, tgt}
			}
			close(prefetchCh)
		}()

		for item := range prefetchCh {
			// Learning Rate Warmup
			if globalStep < warmupSteps {
				startLR := 1e-7
				lr := startLR + (learningRate-startLR)*float64(globalStep)/float64(warmupSteps)
				if opt, ok := optimizer.(*neuralnn.Adam); ok {
					opt.SetLearningRate(lr)
				}
			}
			globalStep++

		

			inputTensor := item.input
			targetTensor := item.target

			if targetTensor.Shape[1] < 2 {
				continue
			}

			// Slice targetTensor for decoder input (remove last token, e.g. EOS)
			decoderInput, err := targetTensor.Slice(1, 0, targetTensor.Shape[1]-1)
			if err != nil {
				log.Printf("Slice error: %v", err)
				continue
			}

			optimizer.ZeroGrad()

			// Forward
			// Teacher Forcing Schedule:
			var samplingProb float64
			if epoch >= 2 {
				// Linear climb: 5% at epoch 2, up to 35% max
				samplingProb = math.Min(0.35, float64(epoch-2)*0.05)
			}
			logits, _, err := intentModel.Forward(samplingProb, inputTensor, decoderInput)
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
				targetSeqLen := targetTensor.Shape[1] - 1
				targets := make([]int, targetSeqLen)
				for t := 0; t < targetSeqLen; t++ {
					targets[t] = int(targetTensor.Data[t+1])
				}

				loss, grad := tensor.CrossEntropyLoss(l, targets, intentModel.SentenceVocab.PaddingTokenID, labelSmoothing)
				if grad == nil {
					grad = tensor.NewTensor(l.Shape, make([]float64, len(l.Data)), false)
				}

				// CrossEntropyLoss already normalizes by active (non-pad) tokens,
				// so no additional length factor is needed.

				batchLoss = loss
				grads = []*tensor.Tensor{grad}
			} else {
				// Sequence of logits (Step-by-step path used for scheduled sampling)
				grads = make([]*tensor.Tensor, len(logits))
				var stepLossTotal float64
				for t, logit := range logits {
					// Target for this step is AIDs[t+1]
					targets := []int{int(targetTensor.Data[t+1])}
					l, g := tensor.CrossEntropyLoss(logit, targets, intentModel.SentenceVocab.PaddingTokenID, labelSmoothing)
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

			// Add MoE Load Balancing Loss to the logged batchLoss for monitoring
			var currentBatchLB float64
			for layerIdx, l := range moe.ActiveLayers {
				currentBatchLB += l.LoadBalancingLoss * l.LoadBalancingWeight
				// Track aggregate utilization
				stats := l.UtilizationStats()
				for expIdx, count := range stats {
					key := fmt.Sprintf("%d:%d", layerIdx, expIdx)
					epochUtilization[key] += count
				}
			}
			batchLoss += currentBatchLB
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

					// 1. Weight Decay (Penalty) to prevent weights from growing too large
					const weightDecay = 1e-4
					for _, p := range params {
						if p.Grad != nil {
							for i := range p.Grad.Data {
								p.Grad.Data[i] += weightDecay * p.Data[i]
							}
						}
					}

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

					const clipValue = 1.0
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

					if batches%100 == 0 {
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

			// Console Logging every 100 batches
			if batches%100 == 0 {
				elapsed := time.Since(epochStartTime).Seconds()
				batchesPerSec := float64(batches) / elapsed
				log.Printf("Epoch %d, Batch %d/%d, Loss: %.4f (Avg LB: %.4f, Step: %d) [%.1f b/s]", epoch, batches, len(chatPairs), batchLoss, epochLBLoss/float64(batches), globalStep, batchesPerSec)
			}


			// Memory safety: Clear computation graph every 10 batches
			if batches%10 == 0 {
				DetachModel(intentModel)
			}
			// Trigger GC less frequently (every 500 batches)
			if batches%500 == 0 {
				runtime.GC()
			}
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
			}
		}
		DetachModel(intentModel)
		avgLoss := totalLoss / float64(batches)
		fmt.Printf("Epoch %d: Avg Loss %.4f in %.1fs\n", epoch+1, avgLoss, time.Since(epochStartTime).Seconds())

		// Validation
		valPPL := ValidateChat(intentModel, valPairs, w2v)
		log.Printf("📉 Validation Perplexity: %.2f", valPPL)

		// Check for Plateau
		if avgLoss >= lastEpochLoss*0.99 { // If improvement is less than 1%
			plateauCount++
		} else {
			plateauCount = 0
		}

		if plateauCount >= 2 { // If stuck for 2 epochs, drop LR
			learningRate *= 0.5
			if learningRate < minLR {
				learningRate = minLR
			}
			if opt, ok := optimizer.(*neuralnn.Adam); ok {
				opt.SetLearningRate(learningRate)
			}
			log.Printf("📉 Plateau detected. LR dropped to %f", learningRate)
			plateauCount = 0
		}
		lastEpochLoss = avgLoss

		// Log History
		logEpochHistory(projectRoot, epoch+1, avgLoss, epochLBLoss/float64(batches), learningRate)

		// Sanity checks: diverse test prompts to monitor generation quality
		runTestSentence("greeting", "how are you", intentModel, w2v)
		runTestSentence("weather", "is it raining today", intentModel, w2v)
		runTestSentence("weekend", "any plans for the weekend", intentModel, w2v)
		runTestSentence("feeling", "i feel tired today", intentModel, w2v)
		runTestSentence("hobby", "what do you like to do", intentModel, w2v)

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

// runTestSentence is a helper to run test questions during training
func runTestSentence(label string, sentence string, model *moe.IntentMoE, w2v *word2vec.SimpleWord2Vec) {
	testTokens := cleanTokenize(sentence)
	testIDs := make([]float64, len(testTokens))
	for i, t := range testTokens {
		testIDs[i] = lookupW2V(t, w2v)
	}
	testInput := tensor.NewTensor([]int{1, len(testIDs)}, testIDs, false)
	testEmb, _ := model.Embedding.Forward(testInput)
	testCtx, _ := model.Encoder.Forward(testEmb)
	// Use Beam Search with Repetition and Length Penalties
	testOutIDs := BeamSearchDecode(model, testCtx, 5, 50)
	var testWords []string
	for _, id := range testOutIDs {
		w := model.SentenceVocab.GetWord(id)
		if w != "<s>" && w != "</s>" && w != "<pad>" {
			testWords = append(testWords, w)
		}
	}
	fmt.Printf("   🧪 Test '%s': %s\n", sentence, strings.Join(testWords, " "))
}

// lookupW2V tries to find a word2vec ID for a token using a cascade of fallbacks
// to recover from the ~22% UNK miss rate observed in training:
//  1. Exact match
//  2. Strip trailing punctuation (e.g. "you." → "you")
//  3. Stem prefix: first 5 chars (e.g. "feeling" → "feeli"... then 4 chars)
//  4. Fall back to UNK
func lookupW2V(token string, w2v *word2vec.SimpleWord2Vec) float64 {
	if id, ok := w2v.Vocabulary[token]; ok {
		return float64(id)
	}
	// Lowercase match
	lower := strings.ToLower(token)
	if id, ok := w2v.Vocabulary[lower]; ok {
		return float64(id)
	}
	// Try stripping trailing punctuation
	stripped := strings.TrimRight(token, ".,!?;:'\"")
	if stripped != token {
		if id, ok := w2v.Vocabulary[stripped]; ok {
			return float64(id)
		}
		// Stripped + Lowercase
		lowerStripped := strings.ToLower(stripped)
		if id, ok := w2v.Vocabulary[lowerStripped]; ok {
			return float64(id)
		}
	}
	// Prefix stem: try first 5, then 4 characters
	for _, prefixLen := range []int{5, 4} {
		if len(stripped) > prefixLen {
			if id, ok := w2v.Vocabulary[stripped[:prefixLen]]; ok {
				return float64(id)
			}
		}
	}
	// Final fallback: UNK
	if id, ok := w2v.Vocabulary["UNK"]; ok {
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
	idx   int
}

func NewChatDataIterator(pairs []struct{ Q, A string }, w2v *word2vec.SimpleWord2Vec, vocab *mainvocab.Vocabulary) *ChatDataIterator {
	// Shuffle pairs for better training
	rand.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	return &ChatDataIterator{
		pairs: pairs,
		w2v:   w2v,
		vocab: vocab,
		idx:   0,
	}
}

func (it *ChatDataIterator) HasNext() bool {
	return it.idx < len(it.pairs)
}

func (it *ChatDataIterator) Next() (*tensor.Tensor, *tensor.Tensor) {
	pair := it.pairs[it.idx]
	it.idx++

	// Query Tokenization (W2V) — use cascaded lookup to reduce the 22% UNK rate
	qTokens := cleanTokenize(pair.Q)
	qIDs := make([]float64, len(qTokens))
	for i, t := range qTokens {
		qIDs[i] = lookupW2VIter(t, it.w2v)
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
		aIDs[idx] = float64(it.vocab.GetTokenID(t))
		idx++
	}
	aIDs[idx] = float64(it.vocab.EosID)

	inputTensor := tensor.NewTensor([]int{1, len(qIDs)}, qIDs, false)
	targetTensor := tensor.NewTensor([]int{1, len(aIDs)}, aIDs, false)
	return inputTensor, targetTensor
}

// lookupW2VIter is the same cascade lookup used by the iterator.
// Defined as a method-style helper to avoid duplicating the logic.
func lookupW2VIter(token string, w2v *word2vec.SimpleWord2Vec) float64 {
	if id, ok := w2v.Vocabulary[token]; ok {
		return float64(id)
	}
	// Lowercase match
	lower := strings.ToLower(token)
	if id, ok := w2v.Vocabulary[lower]; ok {
		return float64(id)
	}
	stripped := strings.TrimRight(token, ".,!?;:'\"")
	if stripped != token {
		if id, ok := w2v.Vocabulary[stripped]; ok {
			return float64(id)
		}
		// Stripped + Lowercase
		lowerStripped := strings.ToLower(stripped)
		if id, ok := w2v.Vocabulary[lowerStripped]; ok {
			return float64(id)
		}
	}
	for _, prefixLen := range []int{5, 4} {
		if len(stripped) > prefixLen {
			if id, ok := w2v.Vocabulary[stripped[:prefixLen]]; ok {
				return float64(id)
			}
		}
	}
	if id, ok := w2v.Vocabulary["UNK"]; ok {
		return float64(id)
	}
	return 0
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
		}
	}
	// Clear encoder state
	if model.Encoder != nil {
		if hybrid, ok := model.Encoder.(*moe.HybridLLMGNNEncoder); ok {
			hybrid.ClearState()
		} else if moeEnc, ok := model.Encoder.(*moe.MoELayer); ok {
			moeEnc.ClearState()
		}
	}
	// Clear decoder state
	if model.Decoder != nil {
		model.Decoder.ClearState()
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
	var tokens []string
	var currentToken strings.Builder
	for _, r := range text {
		if unicode.IsSpace(r) {
			if currentToken.Len() > 0 {
				tokens = append(tokens, strings.ToLower(currentToken.String()))
				currentToken.Reset()
			}
		} else if unicode.IsPunct(r) || unicode.IsSymbol(r) {
			if r == '\'' && currentToken.Len() > 0 {
				currentToken.WriteRune(r)
			} else {
				if currentToken.Len() > 0 {
					tokens = append(tokens, strings.ToLower(currentToken.String()))
					currentToken.Reset()
				}
				tokens = append(tokens, string(r))
			}
		} else {
			currentToken.WriteRune(r)
		}
	}
	if currentToken.Len() > 0 {
		tokens = append(tokens, strings.ToLower(currentToken.String()))
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

	for _, pair := range valPairs {
		// Tokenize as usual
		qTokens := cleanTokenize(pair.Q)
		qIDs := make([]float64, len(qTokens))
		for i, t := range qTokens {
			qIDs[i] = lookupW2V(t, w2v)
		}

		aTokens := cleanTokenize(pair.A)
		aIDs := make([]float64, len(aTokens)+2)
		aIDs[0] = float64(model.SentenceVocab.BosID)
		for i, t := range aTokens {
			aIDs[i+1] = float64(model.SentenceVocab.GetTokenID(t))
		}
		aIDs[len(aIDs)-1] = float64(model.SentenceVocab.EosID)

		inputT := tensor.NewTensor([]int{1, len(qIDs)}, qIDs, false)
		targetT := tensor.NewTensor([]int{1, len(aIDs)}, aIDs, false)

		// Forward pass without teacher forcing (samplingProb = 0)
		decIn, _ := targetT.Slice(1, 0, targetT.Shape[1]-1)
		logits, _, _ := model.Forward(0.0, inputT, decIn)

		// Calculate Loss (vectorized)
		targetSeqLen := targetT.Shape[1] - 1
		targets := make([]int, targetSeqLen)
		for t := 0; t < targetSeqLen; t++ {
			targets[t] = int(targetT.Data[t+1])
		}

		loss, _ := tensor.CrossEntropyLoss(logits[0], targets, model.SentenceVocab.PaddingTokenID, 0.0)
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
			logits, _, _ := model.Forward(0.0, nil, inputT)

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

// ChatSession manages the conversation history for sliding window memory.
type ChatSession struct {
	History    []string
	MaxHistory int // Number of exchanges to remember
}

func (s *ChatSession) AddToHistory(user, bot string) {
	s.History = append(s.History, fmt.Sprintf("User: %s", user))
	s.History = append(s.History, fmt.Sprintf("Bot: %s", bot))

	// Keep only the most recent exchanges (2 strings per exchange)
	if len(s.History) > s.MaxHistory*2 {
		s.History = s.History[len(s.History)-(s.MaxHistory*2):]
	}
}

func (s *ChatSession) GetContextualPrompt(systemPrompt string) string {
	// Combine System Prompt + History
	if len(s.History) == 0 {
		return systemPrompt
	}
	fullContext := systemPrompt + "\n" + strings.Join(s.History, "\n")
	return fullContext
}

func StartChat(model *moe.IntentMoE, w2v *word2vec.SimpleWord2Vec) {
	session := &ChatSession{MaxHistory: 3}
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

		// Sentiment Analysis & Emotional Steering
		sentiment := GetSentimentScore(input)
		if sentiment < -0.5 {
			fmt.Println("🤖 [System Note: Bot is in 'Apologetic Mode']")
		}

		// 1. Build the full "story" so far
		// This looks like: [System] + [History Turn 1] + [History Turn 2] + [Current User Input]
		fullContext := session.GetContextualPrompt(systemPrompt) + "\nUser: " + input

		// 2. Standard tokenization and Forward pass
		tokens := cleanTokenize(fullContext)
		ids := make([]float64, len(tokens))
		for i, t := range tokens {
			ids[i] = lookupW2V(t, w2v)
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
		outIDs := BeamSearchDecode(model, ctx, 5, 50)

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
		session.AddToHistory(input, botResponse)

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
		session:      &ChatSession{MaxHistory: 3},
		systemPrompt: "System: You are a friendly, helpful assistant. Tone: Kind.",
	}
}

var modelMutex sync.Mutex

func (b *MoEChatBot) Reply(input string) string {
	modelMutex.Lock()
	defer modelMutex.Unlock()

	// Sentiment Analysis & Emotional Steering
	sentiment := GetSentimentScore(input)
	if sentiment < -0.5 {
		// fmt.Println("🤖 [System Note: Bot is in 'Apologetic Mode']")
	}

	// 1. Build the full "story" so far
	fullContext := b.session.GetContextualPrompt(b.systemPrompt) + "\nUser: " + input

	// 2. Standard tokenization and Forward pass
	tokens := cleanTokenize(fullContext)
	ids := make([]float64, len(tokens))
	for i, t := range tokens {
		ids[i] = lookupW2V(t, b.w2v)
	}

	inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

	// Inference (Eval Mode)
	for _, l := range moe.ActiveLayers {
		l.SetMode(false)
	}

	emb, _ := b.model.Embedding.Forward(inputT)
	ctx, _ := b.model.Encoder.Forward(emb)

	// 4. Beam Search Decoding
	outIDs := BeamSearchDecode(b.model, ctx, 5, 50)

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
	b.session.AddToHistory(input, botResponse)

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
		if sentiment < -0.5 {
		}

		// 1. Build the full "story" so far
		fullContext := b.session.GetContextualPrompt(b.systemPrompt) + "\nUser: " + userInput

		// 2. Tokenize
		tokens := cleanTokenize(fullContext)
		ids := make([]float64, len(tokens))
		for i, t := range tokens {
			ids[i] = lookupW2V(t, b.w2v)
		}
		inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

		// 3. Encode (Eval mode)
		for _, l := range moe.ActiveLayers {
			l.SetMode(false)
		}
		emb, _ := b.model.Embedding.Forward(inputT)
		b.model.Encoder.Forward(emb)

		// 4. Decode Loop
		currIDs := []float64{float64(b.model.SentenceVocab.BosID)}
		var responseTokens []string

		for i := 0; i < 50; i++ {
			decInputT := tensor.NewTensor([]int{1, len(currIDs)}, currIDs, false)
			logits, _, _ := b.model.Forward(0.0, nil, decInputT)

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
		b.session.AddToHistory(userInput, strings.Join(responseTokens, " "))

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
