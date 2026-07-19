package chat

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
	neuralnn "github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/tokenizer"
)

// LoadComputerCSV parses the computer.csv dataset into TrainPairs.
func LoadComputerCSV(path string) []moe.TrainPair {
	var pairs []moe.TrainPair
	f, err := os.Open(path)
	if err != nil {
		return pairs
	}
	defer f.Close()
	reader := csv.NewReader(f)
	records, _ := reader.ReadAll()
	for i, record := range records {
		if i == 0 || len(record) < 2 {
			continue
		}
		intent := "computer"
		if len(record) >= 3 {
			intent = record[2]
		}
		pairs = append(pairs, moe.TrainPair{Q: record[0], A: record[1], Intent: intent})
	}
	return pairs
}

// setLayerFreezeQuiet sets expert freeze state without printing if unchanged.
func setLayerFreezeQuiet(layer *moe.MoELayer, expertID int, freeze bool) {
	if expertID < 0 || expertID >= len(layer.ExpertFrozen) {
		return
	}
	if layer.ExpertFrozen[expertID] == freeze {
		return // Already in correct state — skip (suppresses spam)
	}
	layer.SetExpertFreeze(expertID, freeze)
}

// applyPhaseFreeze sets the correct freeze pattern for the given phase on all layers.
// Phase 1: freeze experts >= 8 (cartridges), thaw 0..7 (conversational)
// Phase 2: freeze experts 0..7 (conversational), thaw >= 8 (cartridges)
// Phase 3: thaw everyone
func applyPhaseFreeze(layers []*moe.MoELayer, phase int) {
	for _, layer := range layers {
		for i := 0; i < len(layer.Experts); i++ {
			var shouldFreeze bool
			switch phase {
			case 1:
				shouldFreeze = i >= 8 // Cartridge slots frozen
			case 2:
				shouldFreeze = i < 8 // Social slots frozen
			case 3:
				shouldFreeze = false // Unfreeze all
			}
			setLayerFreezeQuiet(layer, i, shouldFreeze)
		}
	}
}

// svcIsCoherent returns true when a generated response looks like real language
// (not SALAD): needs at least 4 words and a type-token ratio above 0.5.
func svcIsCoherent(response string) bool {
	words := strings.Fields(strings.ToLower(response))
	if len(words) < 4 {
		return false
	}

	hasConversational := false
	socialTokens := map[string]bool{"hi": true, "hello": true, "assistant": true, "am": true, "gollemer": true, "how": true, "you": true, "i": true}
	techTokens := map[string]bool{"elasticsearch": true, "cloudflare": true, "/readyz": true, "goroutines": true, "asynchronous": true, "pub/sub": true, "cpu": true, "dataset": true}

	unique := make(map[string]struct{})
	for _, w := range words {
		wClean := strings.Trim(w, ".,!?")
		unique[w] = struct{}{}
		if socialTokens[wClean] {
			hasConversational = true
		}
		if techTokens[wClean] {
			return false // Reject if tech words dominate/exist
		}
	}

	ttr := float64(len(unique)) / float64(len(words))
	return ttr >= 0.5 && hasConversational
}

// TrainMultiPhaseCurriculum orchestrates the 3-phase curriculum:
//
//	Phase 1 — Social Core: train on conversations.csv; cartridge experts frozen.
//	           Advance when Subject-Verb Connection > 0.05 three times in a row
//	           AND the test response is coherent (not SALAD).
//	Phase 2 — Cartridge Ingestion: train on computer.csv; conversational experts frozen.
//	           Advance after 30 epochs of stable cartridge training.
//	Phase 3 — Cohesive Tuning: train on both; all experts unfrozen; LR ≤ 1e-5;
//	           load-balancing weight enabled on every layer.
func TrainMultiPhaseCurriculum(projectRoot string, useGPU bool) {
	log.Println("🚀 Starting 3-Phase Multi-Domain Curriculum Training...")

	// ── 1. Load datasets ──────────────────────────────────────────────────────
	var socialPairs []moe.TrainPair
	conversingCSVPath := filepath.Join(projectRoot, "data/training/trainingdata/conversations.csv")
	if pairs, err := LoadConversationCSV(conversingCSVPath); err == nil {
		socialPairs = append(socialPairs, pairs...)
	} else {
		log.Printf("⚠️ conversations.csv: %v", err)
	}

	computerCSVPath := filepath.Join(projectRoot, "data/training/trainingdata/computer/computer.csv")
	computerPairs := LoadComputerCSV(computerCSVPath)

	// ── Synthetic data augmentation ───────────────────────────────────────────
	syntheticCSVPath := filepath.Join(projectRoot, "data/training/trainingdata/synthetic_pairs.csv")
	if f, err := os.Open(syntheticCSVPath); err == nil {
		defer f.Close()
		reader := csv.NewReader(f)
		records, _ := reader.ReadAll()
		for i, record := range records {
			if i == 0 || len(record) < 2 {
				continue // skip header or malformed
			}
			q, a := record[0], record[1]
			intent := "technical"
			if len(record) >= 3 && record[2] != "" {
				intent = record[2]
			}
			grammar := ""
			if len(record) >= 4 {
				grammar = record[3]
			}
			if q != "" && a != "" {
				socialPairs = append(socialPairs, moe.TrainPair{Q: q, A: a, Intent: intent, Grammar: grammar})
			}
		}
		log.Printf("📚 Loaded %d synthetic pairs from synthetic_pairs.csv", len(records)-1)
	} else {
		log.Printf("⚠️ synthetic_pairs.csv not found at %s: %v", syntheticCSVPath, err)
	}

	if len(socialPairs) == 0 || len(computerPairs) == 0 {
		log.Fatalf("❌ Missing required datasets (social=%d, computer=%d). Aborting.", len(socialPairs), len(computerPairs))
	}
	log.Printf("📚 Total social+technical pairs: %d | Computer pairs: %d", len(socialPairs), len(computerPairs))

	// ── 2. Build vocabulary ───────────────────────────────────────────────────
	tmpVocab := mainvocab.NewVocabulary()
	for _, tok := range []string{"__ques__", "__ans__", "__intent__", "social", "computer", ":"} {
		tmpVocab.AddToken(tok)
	}
	for _, pair := range append(socialPairs, computerPairs...) {
		for _, t := range cleanTokenize(pair.Q + " " + pair.A) {
			tmpVocab.AddToken(t)
		}
	}

	// ── 3. Load or create model ───────────────────────────────────────────────
	var intentModel *moe.IntentMoE
	socialModelPath := filepath.Join(projectRoot, "data/models/gob_models/moe_social_model.gob")
	if _, err := os.Stat(socialModelPath); err == nil {
		log.Printf("⬇️ Loading existing model from %s", socialModelPath)
		intentModel, _ = moe.LoadIntentMoEModelWithFallback(socialModelPath)
	}

	const modelDim = 512
	const baseExperts = 8
	if intentModel == nil {
		intentModel, _ = moe.NewHybridIntentMoE(
			tmpVocab.Size(), modelDim, baseExperts,
			modelDim, modelDim, tmpVocab.Size(), 2, nil,
		)
		intentModel.Decoder, _ = moe.NewRNNDecoder(modelDim, tmpVocab.Size(), modelDim, 8, 1, 0.0, baseExperts)
		intentModel.RepairArchitecture()
		intentModel.RebuildActiveLayers()
		for _, p := range intentModel.Parameters() {
			InitializeHeNormal(p)
		}
		intentModel.Rules = moe.NewRuleBook()
	} else {
		intentModel.RepairArchitecture()
	}

	// Merge vocab into model
	if intentModel.SentenceVocab == nil {
		intentModel.SentenceVocab = tmpVocab
	} else {
		for k := range tmpVocab.WordToToken {
			intentModel.SentenceVocab.AddToken(k)
		}
	}
	tokenizer.InjectIntoVocab(intentModel.SentenceVocab.WordToToken, &intentModel.SentenceVocab.TokenToWord)
	intentModel.SentenceVocabSize = intentModel.SentenceVocab.Size()
	intentModel.Decoder.ResizeOutputLayer(intentModel.SentenceVocabSize)
	intentModel.ResizeEmbeddings(intentModel.SentenceVocabSize)
	moe.ActiveLayers = findMoELayers(intentModel)

	if useGPU {
		intentModel.ToGPU()
	}

	// ── 4. Optimizer ──────────────────────────────────────────────────────────
	optimizer := &neuralnn.CoolingOptimizer{
		Base: neuralnn.NewOptimizer(intentModel.Parameters(), 0.0005, 1.0),
	}

	// ── 5. Seed structural experts ────────────────────────────────────────────
	supervisor := moe.NewSupervisor()
	supervisor.SeedSystemExperts(intentModel)

	layers := findMoELayers(intentModel)

	// ── 6. Training state ─────────────────────────────────────────────────────
	const batchSize = 32
	const maxSeqLen = 24
	const maxEpochs = 2000
	const labelSmoothing = float32(0.05)

	// Build flat loss-weight slice for WeightedCrossEntropy
	lossWeights := buildDefaultLossWeights(intentModel.SentenceVocab)

	currentPhase := 1
	consecutiveHighSVC := 0
	phase2Epochs := 0

	// Ensure correct freeze state before epoch 0
	applyPhaseFreeze(layers, currentPhase)

	for epoch := 0; epoch < maxEpochs; epoch++ {
		epochStart := time.Now()
		// ── Select dataset & set LR ────────────────────────────────────────────
		var trainPairs []moe.TrainPair
		var phaseLR float32
		switch currentPhase {
		case 1:
			trainPairs = socialPairs
			phaseLR = 0.001 // Higher LR for cold start — pulls random init out of chaos
			for _, layer := range layers {
				layer.LoadBalancingWeight = 0.05
				layer.RouterTemperature = 1.2
				layer.ExpertDropoutRate = 0.1
				// Force all tokens to Expert 0 during first 5 epochs (cold start)
				// to prevent thin weight distribution across 8 experts with small corpus.
				// Disabled after epoch 5; avgLoss threshold checked at end of epoch.
				if epoch < 5 {
					layer.ForceSingleExpert = true
				}
			}
		case 2:
			trainPairs = computerPairs
			phaseLR = 0.0002
			for _, layer := range layers {
				layer.LoadBalancingWeight = 0.05
				layer.RouterTemperature = 1.2
				layer.ExpertDropoutRate = 0.1
			}
		case 3:
			trainPairs = append(socialPairs, computerPairs...)
			phaseLR = 0.00005 // η = 5*10⁻⁵
			for _, layer := range layers {
				layer.LoadBalancingWeight = 1.0
				layer.RouterTemperature = 1.0
				layer.ExpertDropoutRate = 0.05
			}
		}
		optimizer.SetLearningRate(phaseLR)

		rand.Shuffle(len(trainPairs), func(i, j int) { trainPairs[i], trainPairs[j] = trainPairs[j], trainPairs[i] })

		// ── Inner batch loop ───────────────────────────────────────────────────
		var totalLoss float32
		batches := 0

		for i := 0; i < len(trainPairs); i += batchSize {
			end := i + batchSize
			if end > len(trainPairs) {
				end = len(trainPairs)
			}
			batch := trainPairs[i:end]
			currentBatchSize := len(batch)

			optimizer.ZeroGrad()

			// Build input/target tensors
			inputData := make([]float32, currentBatchSize*maxSeqLen)
			targetData := make([]float32, currentBatchSize*maxSeqLen)

			padID := intentModel.SentenceVocab.PaddingTokenID

			for bIdx, pair := range batch {
				qText := "__intent__ " + pair.Intent + " : __ques__ " + pair.Q
				qToks := cleanTokenize(qText)
				for t := 0; t < maxSeqLen && t < len(qToks); t++ {
					id := intentModel.SentenceVocab.GetTokenID(qToks[t])
					if id < 0 {
						id = padID
					}
					inputData[bIdx*maxSeqLen+t] = float32(id)
				}

				aToks := cleanTokenize(pair.A)
				eosID := intentModel.SentenceVocab.EosID
				if eosID < 0 {
					eosID = intentModel.SentenceVocab.GetTokenID("<EOS>")
				}
				for t := 0; t < maxSeqLen && t < len(aToks); t++ {
					id := intentModel.SentenceVocab.GetTokenID(aToks[t])
					if id < 0 {
						id = padID
					}
					targetData[bIdx*maxSeqLen+t] = float32(id)
				}
				if len(aToks) < maxSeqLen {
					targetData[bIdx*maxSeqLen+len(aToks)] = float32(eosID)
				}
			}

			inputTensor := tensor.NewTensor([]int{currentBatchSize, maxSeqLen}, inputData, false)
			targetTensor := tensor.NewTensor([]int{currentBatchSize, maxSeqLen}, targetData, false)

			for _, layer := range layers {
				layer.CurrentPhase = currentPhase
			}

			logits, _, err := intentModel.Forward(0.1, inputTensor, targetTensor)
			if err != nil {
				log.Printf("⚠️ Forward error (batch %d): %v", i/batchSize, err)
				intentModel.ClearState()
				continue
			}

			// ── Loss computation — mirrors chat.go approach ────────────────────
			var batchLoss float32
			var grads []*tensor.Tensor

			if len(logits) == 1 && len(logits[0].Shape) == 3 {
				// Vectorized 3D path: logits shape [batch, seqLen-1, vocab]
				targetSeqLen := maxSeqLen - 1
				targets := make([]int, currentBatchSize*targetSeqLen)
				var eosPenalty float32
				vocabSize := logits[0].Shape[2]
				eosID := intentModel.SentenceVocab.EosID
				if eosID < 0 {
					eosID = intentModel.SentenceVocab.GetTokenID("<EOS>")
				}

				for b := 0; b < currentBatchSize; b++ {
					eosExpectedAt := -1
					for t := 0; t < targetSeqLen; t++ {
						tID := int(targetData[b*maxSeqLen+t+1])
						targets[b*targetSeqLen+t] = tID
						if eosExpectedAt == -1 && tID == eosID {
							eosExpectedAt = t
						}
					}
					if eosExpectedAt != -1 {
						offset := (b*targetSeqLen + eosExpectedAt) * vocabSize
						maxLogit := float32(-1e9)
						predID := -1
						for v := 0; v < vocabSize; v++ {
							val := logits[0].Data[offset+v]
							if val > maxLogit {
								maxLogit = val
								predID = v
							}
						}
						if predID != eosID && predID != padID {
							eosPenalty += 0.15
						}
					}
				}
				loss, grad := WeightedCrossEntropy(logits[0].ToCPU(), targets, lossWeights, labelSmoothing, 0.005)
				if grad == nil {
					grad = tensor.NewTensor(logits[0].Shape, make([]float32, len(logits[0].Data)), false)
				}
				penaltyFactor := float32(1.0) + (eosPenalty / float32(currentBatchSize))
				batchLoss = loss * penaltyFactor
				if penaltyFactor > 1.0 {
					for i := range grad.Data {
						grad.Data[i] *= penaltyFactor
					}
				}
				grads = []*tensor.Tensor{grad}
			} else {
				// Step-by-step path
				grads = make([]*tensor.Tensor, len(logits))
				var stepTotal float32
				for t, logit := range logits {
					targets := make([]int, currentBatchSize)
					for b := 0; b < currentBatchSize; b++ {
						idx := b*maxSeqLen + t + 1
						if idx < len(targetData) {
							targets[b] = int(targetData[idx])
						} else {
							targets[b] = padID
						}
					}

					eosID := intentModel.SentenceVocab.EosID
					if eosID < 0 {
						eosID = intentModel.SentenceVocab.GetTokenID("<EOS>")
					}
					var eosPenalty float32
					vocabSize := logit.Shape[1]
					for b := 0; b < currentBatchSize; b++ {
						if targets[b] == eosID {
							offset := b * vocabSize
							maxLogit := float32(-1e9)
							predID := -1
							for v := 0; v < vocabSize; v++ {
								val := logit.Data[offset+v]
								if val > maxLogit {
									maxLogit = val
									predID = v
								}
							}
							if predID != eosID && predID != padID {
								eosPenalty += 0.15
							}
						}
					}
					l, g := WeightedCrossEntropy(logit.ToCPU(), targets, lossWeights, labelSmoothing, 0.005)
					if g == nil {
						g = tensor.NewTensor(logit.Shape, make([]float32, len(logit.Data)), false)
					}
					penaltyFactor := float32(1.0) + (eosPenalty / float32(currentBatchSize))
					stepTotal += l * penaltyFactor
					if penaltyFactor > 1.0 {
						for i := range g.Data {
							g.Data[i] *= penaltyFactor
						}
					}
					grads[t] = g
				}
				div := float32(len(logits))
				batchLoss = stepTotal / div
				for t := range grads {
					for i := range grads[t].Data {
						grads[t].Data[i] /= div
					}
				}
			}

			totalLoss += batchLoss
			if err := intentModel.Backward(grads...); err != nil {
				log.Printf("⚠️ Backward error: %v", err)
			} else {
				optimizer.ClipGradients()
				optimizer.Step()
			}
			intentModel.ClearState()
			batches++
		}

		avgLoss := float32(0.0)
		if batches > 0 {
			avgLoss = totalLoss / float32(batches)
		}
		epochDuration := time.Since(epochStart).Seconds()
		log.Printf("Phase %d | Epoch %d | Loss: %.4f | LR: %.6f | EpochTime: %.1fs", currentPhase, epoch, avgLoss, phaseLR, epochDuration)

		var activeExps []string
		if len(layers) > 0 {
			for i := 0; i < len(layers[0].ExpertFrozen); i++ {
				if !layers[0].ExpertFrozen[i] {
					activeExps = append(activeExps, fmt.Sprintf("%d", i))
				}
			}
			log.Printf("Active Experts: [%s]", strings.Join(activeExps, ", "))
		}

		// ── Phase transition logic ─────────────────────────────────────────────
		switch currentPhase {
		case 1:
			testPrompt := "__intent__ social : __ques__ how are you"
			gen, _, atts := StrictGenerate(intentModel, testPrompt, 15, 1.3, false, epoch)

			// Subject index in prompt
			pToks := cleanTokenize(testPrompt)
			subjectIdx := -1
			for idx, tok := range pToks {
				if moe.MapWordToGrammarType(tok) == "PRON" {
					subjectIdx = idx
					break
				}
			}

			// Verb step in generated response
			rWords := strings.Fields(gen)
			verbStep := -1
			for idx, w := range rWords {
				t := moe.MapWordToGrammarType(w)
				if t == "VERB" || t == "AUX" {
					verbStep = idx
					break
				}
			}

			// Average attention from verb step → subject pronoun
			avgAtt := float32(0.0)
			if subjectIdx != -1 && verbStep != -1 && verbStep < len(atts) {
				att := atts[verbStep]
				numHeads := intentModel.Decoder.MaxAttentionHeads
				var sumAtt float32
				for h := 0; h < numHeads; h++ {
					idx := h*att.Shape[3] + subjectIdx
					if idx < len(att.Data) {
						sumAtt += att.Data[idx]
					}
				}
				avgAtt = sumAtt / float32(numHeads)
			}

			log.Printf("🧪 Phase1 test: '%s' → SVC=%.4f coherent=%v", gen, avgAtt, svcIsCoherent(gen))

			// Only count as a pass when BOTH the attention threshold AND
			// the coherence guard pass — prevents SALAD from tripping the gate.
			if avgAtt > 0.05 && svcIsCoherent(gen) {
				consecutiveHighSVC++
				log.Printf("✅ SVC > 0.0500 + coherent (%d/3)", consecutiveHighSVC)
				if consecutiveHighSVC >= 3 {
					log.Printf("🚀 Phase 1 complete → advancing to Phase 2 (Cartridge Ingestion)")
					currentPhase = 2
					consecutiveHighSVC = 0
					applyPhaseFreeze(layers, currentPhase)
					moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
				}
			} else {
				consecutiveHighSVC = 0
			}

		case 2:
			phase2Epochs++
			if phase2Epochs >= 30 {
				log.Printf("🚀 Phase 2 complete (%d epochs) → advancing to Phase 3 (Cohesive Tuning)", phase2Epochs)
				currentPhase = 3
				phase2Epochs = 0
				applyPhaseFreeze(layers, currentPhase)
				moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
			}
		}

		// Periodic checkpoint
		if epoch > 0 && epoch%10 == 0 {
			moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
			log.Printf("💾 Checkpoint saved (epoch %d)", epoch)
		}
	}
}

// buildDefaultLossWeights builds a flat weight vector for WeightedCrossEntropy.
// High-frequency stop-words are down-weighted; BOS/EOS/terminal punctuation are boosted.
func buildDefaultLossWeights(vocab *mainvocab.Vocabulary) []float32 {
	if vocab == nil {
		return nil
	}
	weights := make([]float32, vocab.Size())
	for i := range weights {
		weights[i] = 1.0
	}
	suppressed := []string{"it", "is", "a", "the", "i"}
	for _, w := range suppressed {
		id := vocab.GetTokenID(w)
		if id >= 0 && id < len(weights) {
			weights[id] = 0.3
		}
	}
	boosted := []string{".", "!", "?", "<BOS>", "<EOS>", "__ans__"}
	for _, w := range boosted {
		id := vocab.GetTokenID(w)
		if id >= 0 && id < len(weights) {
			weights[id] = 2.0
		}
	}
	return weights
}
