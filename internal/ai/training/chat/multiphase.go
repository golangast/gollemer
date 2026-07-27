package chat

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
	neuralnn "github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/orchestrator"
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
// Phase 1: freeze experts >= freezeStart (cartridges), thaw 0..freezeStart-1 (conversational)
// Phase 2: freeze experts 0..freezeEnd-1 (conversational), thaw >= freezeEnd (cartridges)
// Phase 3: thaw everyone (freezeStart < 0)
func applyPhaseFreeze(layers []*moe.MoELayer, freezeStart, freezeEnd int) {
	for _, layer := range layers {
		for i := 0; i < len(layer.Experts); i++ {
			var shouldFreeze bool
			if freezeStart >= 0 && freezeEnd > freezeStart {
				// Freeze range [freezeStart, freezeEnd)
				shouldFreeze = i >= freezeStart && i < freezeEnd
			} else if freezeStart >= 0 {
				// Freeze from freezeStart to end
				shouldFreeze = i >= freezeStart
			} else {
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

// TrainMultiPhaseCurriculum orchestrates the 3-phase curriculum.
// All hyperparameters are loaded from data/config/social_train.json.
func TrainMultiPhaseCurriculum(projectRoot string, useGPU bool, dataFile string) {
	log.Println("🚀 Starting 3-Phase Multi-Domain Curriculum Training...")

	// ── 0. Load config ────────────────────────────────────────────────────────
	configPath := filepath.Join(projectRoot, "data/config/social_train.json")
	safeCfg, err := orchestrator.NewSafeConfig(configPath)
	if err != nil {
		log.Fatalf("❌ Failed to load config from %s: %v", configPath, err)
	}
	cfg := safeCfg.Get()

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
	var syntheticPairs []moe.TrainPair
	syntheticCSVPath := filepath.Join(projectRoot, "data/training/trainingdata/synthetic_pairs.csv")
	if f, err := os.Open(syntheticCSVPath); err == nil {
		defer f.Close()
		reader := csv.NewReader(f)
		records, _ := reader.ReadAll()
		for i, record := range records {
			if i == 0 || len(record) < 2 {
				continue
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
				syntheticPairs = append(syntheticPairs, moe.TrainPair{Q: q, A: a, Intent: intent, Grammar: grammar})
			}
		}
		log.Printf("📚 Loaded %d synthetic pairs from %s", len(records)-1, syntheticCSVPath)
	} else {
		log.Printf("⚠️ %s not found: %v", syntheticCSVPath, err)
	}

	if len(syntheticPairs) > 0 {
		socialPairs = append(socialPairs, syntheticPairs...)
		log.Printf("📚 Injected %d synthetic pairs via syntheticCSVPath (total social+technical: %d)", len(syntheticPairs), len(socialPairs))
	}

	// ── FIM dataset augmentation ──────────────────────────────────────────────
	if dataFile != "" {
		log.Printf("📂 Loading FIM dataset from %s...", dataFile)
		if fimData, err := os.ReadFile(dataFile); err == nil {
			var fimDataset struct {
				Train []map[string]interface{} `json:"train"`
			}
			if err := json.Unmarshal(fimData, &fimDataset); err == nil {
				fimCount := 0
				for _, item := range fimDataset.Train {
					instruction, _ := item["instruction"].(string)
					targetPatch, _ := item["target_patch"].(string)
					if instruction == "" || targetPatch == "" {
						continue
					}
					// Create a TrainPair: Q=instruction, A=patch, Intent=computer
					computerPairs = append(computerPairs, moe.TrainPair{
						Q:      instruction,
						A:      targetPatch,
						Intent: "computer",
					})
					fimCount++
				}
				log.Printf("📚 Loaded %d FIM examples from %s into computer dataset", fimCount, dataFile)
			} else {
				log.Printf("⚠️ Failed to parse FIM dataset %s: %v", dataFile, err)
			}
		} else {
			log.Printf("⚠️ Failed to read FIM dataset %s: %v", dataFile, err)
		}
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
	// Add code syntax tokens so the model can generate Go code.
	for _, tok := range []string{"{", "}", "(", ")", "=", ":", ";", ".", ",", "!", "?", "nil", "err", "if", "else", "for", "range", "return", "func", "type", "struct", "int", "string", "bool", "float64", "true", "false", "package", "import", "var", "const", "make", "new", "len", "cap", "append", "fmt", "Println", "Sprintf", "Errorf", "error", "interface{}"} {
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

	modelDim := cfg.ModelDim
	if modelDim <= 0 {
		modelDim = 512
	}
	baseExperts := cfg.NumExperts
	if baseExperts <= 0 {
		baseExperts = 8
	}
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

	// ── 4. Optimizer (base LR from config) ────────────────────────────────────
	baseLR := cfg.LearningRate
	if baseLR <= 0 {
		baseLR = 0.0005
	}
	optimizer := &neuralnn.CoolingOptimizer{
		Base: neuralnn.NewOptimizer(intentModel.Parameters(), baseLR, 1.0),
	}

	// ── 5. Seed structural experts ────────────────────────────────────────────
	supervisor := moe.NewSupervisor()
	supervisor.SeedSystemExperts(intentModel)

	layers := findMoELayers(intentModel)

	// ── 6. Training state (from config) ───────────────────────────────────────
	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = 32
	}
	maxSeqLen := cfg.MaxSeqLen
	if maxSeqLen <= 0 {
		maxSeqLen = 24
	}
	maxEpochs := cfg.Epochs
	if maxEpochs <= 0 {
		maxEpochs = 2000
	}
	labelSmoothing := cfg.LabelSmoothing
	if labelSmoothing <= 0 {
		labelSmoothing = 0.05
	}

	// Build flat loss-weight slice for WeightedCrossEntropy
	lossWeights := buildDefaultLossWeights(intentModel.SentenceVocab, &cfg)

	currentPhase := 1
	phase2Epochs := 0

	// ── Phase 1 learning rate scheduler state ────────────────────────────────
	var phase1BestLoss float32 = 1e9
	var phase1StagnantEpochs int
	var phase1LRFactor float32 = 1.0
	const phase1LRDecayPatience = 10
	const phase1LRImprovementThreshold = 0.05

	// Sliding window for Phase 1 probe pass/fail (relaxed gating)
	const probeWindowSize = 5
	probeWindow := make([]bool, 0, probeWindowSize)

	// Ensure correct freeze state before epoch 0
	phaseCfg := cfg.Phases[strconv.Itoa(currentPhase)]
	if phaseCfg != nil {
		applyPhaseFreeze(layers, phaseCfg.FreezeExpertsStart, phaseCfg.FreezeExpertsEnd)
	}

	for epoch := 0; epoch < maxEpochs; epoch++ {
		epochStart := time.Now()
		// ── Select dataset & set LR from config ────────────────────────────────
		var trainPairs []moe.TrainPair
		var phaseLR float32

		phaseCfg := cfg.Phases[strconv.Itoa(currentPhase)]
		if phaseCfg == nil {
			log.Fatalf("❌ Missing config for phase %d", currentPhase)
		}

		switch phaseCfg.Dataset {
		case "social":
			trainPairs = socialPairs
		case "computer":
			trainPairs = computerPairs
		case "all":
			trainPairs = append(socialPairs, computerPairs...)
		default:
			trainPairs = socialPairs
		}

		phaseLR = phaseCfg.LearningRate
		for _, layer := range layers {
			atomic.StoreInt32(&layer.ResetCount, 0)
			layer.LoadBalancingWeight = phaseCfg.LoadBalancingWeight
			layer.RouterTemperature = phaseCfg.RouterTemperature
			layer.ExpertDropoutRate = phaseCfg.ExpertDropout
			if cfg.CapacityFactor > 0 {
				layer.CapacityFactor = cfg.CapacityFactor
			} else {
				layer.CapacityFactor = 2.0
			}
			// Force single expert during cold start epochs
			if phaseCfg.ForceSingleExpertEpochs > 0 && epoch < phaseCfg.ForceSingleExpertEpochs {
				layer.ForceSingleExpert = true
			} else {
				layer.ForceSingleExpert = false
			}
		}
		optimizer.SetLearningRate(phaseLR)

		rand.Shuffle(len(trainPairs), func(i, j int) { trainPairs[i], trainPairs[j] = trainPairs[j], trainPairs[i] })

		// ── Inner batch loop ───────────────────────────────────────────────────
		var totalLoss float32
		batches := 0

		phaseBatchSize := phaseCfg.BatchSize
		if phaseBatchSize <= 0 {
			phaseBatchSize = batchSize
		}
		phaseMaxSeqLen := phaseCfg.MaxSeqLen
		if phaseMaxSeqLen <= 0 {
			phaseMaxSeqLen = maxSeqLen
		}

		for i := 0; i < len(trainPairs); i += phaseBatchSize {
			end := i + phaseBatchSize
			if end > len(trainPairs) {
				end = len(trainPairs)
			}
			batch := trainPairs[i:end]
			currentBatchSize := len(batch)

			optimizer.ZeroGrad()

			// Build input/target tensors
			inputData := make([]float32, currentBatchSize*phaseMaxSeqLen)
			targetData := make([]float32, currentBatchSize*phaseMaxSeqLen)

			padID := intentModel.SentenceVocab.PaddingTokenID

			for bIdx, pair := range batch {
				qText := "__intent__ " + pair.Intent + " : __ques__ " + pair.Q
				qToks := cleanTokenize(qText)
				for t := 0; t < phaseMaxSeqLen && t < len(qToks); t++ {
					id := intentModel.SentenceVocab.GetTokenID(qToks[t])
					if id < 0 {
						id = padID
					}
					inputData[bIdx*phaseMaxSeqLen+t] = float32(id)
				}

				aToks := cleanTokenize(pair.A)
				eosID := intentModel.SentenceVocab.EosID
				if eosID < 0 {
					eosID = intentModel.SentenceVocab.GetTokenID("<EOS>")
				}
				for t := 0; t < phaseMaxSeqLen && t < len(aToks); t++ {
					id := intentModel.SentenceVocab.GetTokenID(aToks[t])
					if id < 0 {
						id = padID
					}
					targetData[bIdx*phaseMaxSeqLen+t] = float32(id)
				}
				if len(aToks) < phaseMaxSeqLen {
					targetData[bIdx*phaseMaxSeqLen+len(aToks)] = float32(eosID)
				}
			}

			inputTensor := tensor.NewTensor([]int{currentBatchSize, phaseMaxSeqLen}, inputData, false)
			targetTensor := tensor.NewTensor([]int{currentBatchSize, phaseMaxSeqLen}, targetData, false)

			for _, layer := range layers {
				layer.CurrentPhase = currentPhase
			}

			logits, _, err := intentModel.Forward(0.1, inputTensor, targetTensor)
			if err != nil {
				log.Printf("⚠️ Forward error (batch %d): %v", i/phaseBatchSize, err)
				intentModel.ClearState()
				continue
			}

			// ── Loss computation — mirrors chat.go approach ────────────────────
			var batchLoss float32
			var grads []*tensor.Tensor

			if len(logits) == 1 && len(logits[0].Shape) == 3 {
				// Vectorized 3D path: logits shape [batch, seqLen-1, vocab]
				targetSeqLen := phaseMaxSeqLen - 1
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
						tID := int(targetData[b*phaseMaxSeqLen+t+1])
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

				// DYNAMIC NORMALIZATION
				var sumWeights float32
				validTokens := 0
				for _, tID := range targets {
					if tID >= 0 && tID < len(lossWeights) && tID != padID {
						sumWeights += lossWeights[tID]
						validTokens++
					}
				}
				avgWeight := float32(1.0)
				if validTokens > 0 && sumWeights > 0 {
					avgWeight = sumWeights / float32(validTokens)
				}

				penaltyFactor := float32(1.0) + (eosPenalty / float32(currentBatchSize))
				batchLoss = (loss * penaltyFactor) / avgWeight

				scale := penaltyFactor / avgWeight
				if scale != 1.0 {
					for i := range grad.Data {
						grad.Data[i] *= scale
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
						idx := b*phaseMaxSeqLen + t + 1
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

					// DYNAMIC NORMALIZATION
					var sumWeights float32
					validTokens := 0
					for _, tID := range targets {
						if tID >= 0 && tID < len(lossWeights) && tID != padID {
							sumWeights += lossWeights[tID]
							validTokens++
						}
					}
					avgWeight := float32(1.0)
					if validTokens > 0 && sumWeights > 0 {
						avgWeight = sumWeights / float32(validTokens)
					}

					penaltyFactor := float32(1.0) + (eosPenalty / float32(currentBatchSize))
					stepTotal += (l * penaltyFactor) / avgWeight

					scale := penaltyFactor / avgWeight
					if scale != 1.0 {
						for i := range g.Data {
							g.Data[i] *= scale
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

		// ── Phase 1: LR scheduler (reduce on stagnation) ──────────────────────
		if currentPhase == 1 {
			// Track best loss and detect stagnation
			if avgLoss < phase1BestLoss-phase1LRImprovementThreshold {
				phase1BestLoss = avgLoss
				phase1StagnantEpochs = 0
			} else {
				phase1StagnantEpochs++
			}

			// If loss has been stagnant for patience epochs, halve the LR factor
			if phase1StagnantEpochs >= phase1LRDecayPatience && phase1LRFactor > 0.03125 {
				phase1LRFactor *= 0.5
				phase1StagnantEpochs = 0
				log.Printf("🔻 Phase 1 loss stagnant for %d epochs → reducing LR factor to %.4f (effective LR: %.6f)",
					phase1LRDecayPatience, phase1LRFactor, phaseCfg.LearningRate*phase1LRFactor)
			}

			// Apply the LR factor on top of the base phase LR
			effectiveLR := phaseCfg.LearningRate * phase1LRFactor
			optimizer.SetLearningRate(effectiveLR)
		}

		// ── Phase transition logic ─────────────────────────────────────────────
		switch currentPhase {
		case 1:
			// Use low-temperature probe (greedy decoding) for evaluation
			// to test the model's ability to generate Go code symbols.
			// Probes with Go code prompts and checks for structural keywords,
			// balanced braces/parentheses, and semantic symbols.
			testPrompt := "__intent__ computer : __ques__ create a function that adds two numbers"
			gen, _, atts := StrictGenerateLowTemp(intentModel, testPrompt, 25, 1.3, false, epoch)

			// Check Go symbol coverage: extract symbols present in the generated code
			symbolsFound := extractGoSymbols(gen)
			symbolCount := len(symbolsFound)

			// Braces/parens balance check
			openBrace := strings.Count(gen, "{")
			closeBrace := strings.Count(gen, "}")
			openParen := strings.Count(gen, "(")
			closeParen := strings.Count(gen, ")")
			balanced := openBrace == closeBrace && openParen == closeParen

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

			// Pass if: >=2 Go symbols found AND balanced braces/parens AND (symbols include a keyword or identifier)
			hasKeywordOrIdent := false
			for _, sym := range symbolsFound {
				if sym == "func" || sym == "if" || sym == "for" || sym == "return" || sym == "package" || sym == "import" || sym == "type" || sym == "struct" || sym == "var" || len(sym) > 2 {
					hasKeywordOrIdent = true
					break
				}
			}
			probePassed := symbolCount >= 2 && balanced && hasKeywordOrIdent
			log.Printf("🧪 Phase1 test: '%s' → symbols=%d balanced=%v avgAtt=%.4f pass=%v", gen, symbolCount, balanced, avgAtt, probePassed)
			if symbolCount > 0 {
				log.Printf("   Symbols: %v", symbolsFound)
			}

			// Sliding window gating: track last 5 probe results, advance if >= 3 pass
			probeWindow = append(probeWindow, probePassed)
			if len(probeWindow) > probeWindowSize {
				probeWindow = probeWindow[1:]
			}

			passCount := 0
			for _, p := range probeWindow {
				if p {
					passCount++
				}
			}
			log.Printf("📊 Phase1 probe window: %d/%d passes (window: %v)", passCount, len(probeWindow), probeWindow)

			if passCount >= 3 {
				log.Printf("🚀 Phase 1 complete (%d/5 in window) → advancing to Phase 2 (Cartridge Ingestion)", passCount)
				currentPhase = 2
				nextPhaseCfg := cfg.Phases["2"]
				if nextPhaseCfg != nil {
					applyPhaseFreeze(layers, nextPhaseCfg.FreezeExpertsStart, nextPhaseCfg.FreezeExpertsEnd)
				}
				moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
			}

		case 2:
			phase2Epochs++
			if phase2Epochs >= 30 {
				log.Printf("🚀 Phase 2 complete (%d epochs) → advancing to Phase 3 (Cohesive Tuning)", phase2Epochs)
				currentPhase = 3
				phase2Epochs = 0
				nextPhaseCfg := cfg.Phases["3"]
				if nextPhaseCfg != nil {
					applyPhaseFreeze(layers, nextPhaseCfg.FreezeExpertsStart, nextPhaseCfg.FreezeExpertsEnd)
				}
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
// For code-syntax tokens delimiters and structural Go keywords get higher weight.
func buildDefaultLossWeights(vocab *mainvocab.Vocabulary, cfg *orchestrator.TrainingConfig) []float32 {
	if vocab == nil {
		return nil
	}
	weights := make([]float32, vocab.Size())
	for i := range weights {
		weights[i] = 1.0
	}

	if cfg != nil && cfg.TokenWeights != nil {
		for token, weight := range cfg.TokenWeights {
			id := vocab.GetTokenID(token)
			if id >= 0 && id < len(weights) {
				weights[id] = float32(weight)
			}
		}
	} else {
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
	}
	// Always boost Go code syntax tokens for code-fix datasets.
	codeSyntaxBoosted := []string{"{", "}", "(", ")", "=", ":", ";", "if", "else", "for", "range", "func", "return", "package", "type", "struct", "err", "nil", ":= ", "==", "!="}
	for _, w := range codeSyntaxBoosted {
		id := vocab.GetTokenID(w)
		if id >= 0 && id < len(weights) {
			weights[id] = 3.0
		}
	}
	return weights
}

// extractGoSymbols extracts Go code symbols (keywords, identifiers, operators)
// from a generated string. Returns a deduplicated list of symbols found.
func extractGoSymbols(s string) []string {
	if s == "" {
		return nil
	}
	goKeywords := map[string]bool{
		"func": true, "if": true, "for": true, "range": true, "return": true,
		"package": true, "import": true, "type": true, "struct": true,
		"var": true, "const": true, "make": true, "new": true, "len": true,
		"cap": true, "append": true, "defer": true, "go": true, "select": true,
		"switch": true, "case": true, "break": true, "continue": true,
		"fallthrough": true, "else": true, "map": true, "chan": true,
		"interface": true, "error": true, "nil": true, "true": true, "false": true,
	}
	goTypes := map[string]bool{
		"int": true, "string": true, "bool": true, "float64": true,
		"float32": true, "int64": true, "int32": true, "byte": true,
		"rune": true, "uint": true, "uint64": true, "uint32": true,
	}
	operators := map[string]bool{
		"=": true, ":=": true, "==": true, "!=": true, "<": true, ">": true,
		"+": true, "-": true, "*": true, "/": true, "&": true, "|": true,
		"^": true, "<<": true, ">>": true,
	}

	seen := make(map[string]bool)
	var symbols []string

	// Tokenize: split on whitespace and common delimiters
	words := strings.Fields(s)
	for _, w := range words {
		// Trim trailing punctuation/delimiters
		w = strings.TrimRight(w, ".,;:(){}[]\"'`")
		if w == "" {
			continue
		}
		// Check against known sets
		if goKeywords[w] || goTypes[w] || operators[w] {
			if !seen[w] {
				seen[w] = true
				symbols = append(symbols, w)
			}
			continue
		}
		// Identifiers: start with letter or underscore, length > 1
		if len(w) > 1 && ((w[0] >= 'a' && w[0] <= 'z') || (w[0] >= 'A' && w[0] <= 'Z') || w[0] == '_') {
			isAlpha := true
			for _, c := range w[1:] {
				if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
					isAlpha = false
					break
				}
			}
			if isAlpha && !seen[w] {
				seen[w] = true
				symbols = append(symbols, w)
			}
		}
		// Package-qualified names like fmt.Println
		if strings.Contains(w, ".") {
			parts := strings.Split(w, ".")
			for _, p := range parts {
				if p != "" && !seen[p] {
					seen[p] = true
					symbols = append(symbols, p)
				}
			}
		}
	}
	return symbols
}

// codePassCondition evaluates a generated response for code-fix datasets.
// It returns true if:
//  1. The response is a non-empty string.
//  2. Balanced braces { } and parentheses ( ) — same number of opening and closing.
//  3. Contains at least one Go structural keyword (func, if, for, range, return, package, type, struct, var, const, import, make, append, import, defer, go, select, switch, case, break, continue, fallthrough, else).
//
// This replaces the natural-language svcIsCoherent check when training on code datasets.
func codePassCondition(response string) bool {
	response = strings.TrimSpace(response)
	if response == "" {
		return false
	}
	openBrace := strings.Count(response, "{")
	closeBrace := strings.Count(response, "}")
	if openBrace != closeBrace {
		return false
	}
	openParen := strings.Count(response, "(")
	closeParen := strings.Count(response, ")")
	if openParen != closeParen {
		return false
	}
	goKeywords := []string{"func ", "if ", "for ", "range ", "return ", "package ", "type ", "struct ", "var ", "const ", "import ", "make(", "append(", "defer ", "go ", "select {", "switch ", "case ", "break ", "continue ", "fallthrough ", "else ", "err ", "nil", "error", "interface", "map[", "[]", "chan ", "wg.", "http.", "fmt.", "json.", "io.", "os.", "ioutil."}
	lower := strings.ToLower(response)
	for _, kw := range goKeywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}
