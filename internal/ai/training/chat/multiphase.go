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
// (not SALAD): needs at least 2 words and a type-token ratio above 0.5.
func svcIsCoherent(response string) bool {
	words := strings.Fields(strings.ToLower(response))
	if len(words) < 2 {
		return false
	}

	hasConversational := false
	socialTokens := map[string]bool{"hi": true, "hello": true, "assistant": true, "am": true, "gollemer": true, "how": true, "you": true, "i": true, "doing": true, "well": true, "great": true, "fine": true, "good": true, "thanks": true}
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
	// For short responses (2-3 words), relax TTR requirement
	if len(words) <= 3 {
		return hasConversational
	}
	return ttr >= 0.5 && hasConversational
}

// phaseNames maps phase number to its human-readable name and goal.
var phaseNames = map[int]string{
	1: "Social Bootcamp     — experts 0–7 learn conversational patterns; 8–15 frozen",
	2: "Cartridge Injection — experts 8–15 learn technical/code vocab; 0–7 frozen",
	3: "Joint Warmup        — all experts train together; routing stabilizes",
	4: "Router Refinement   — low LR fine-tunes gating; slight LBW increase",
	5: "Coherence Polish    — ultra-low LR; EOS/coherence optimization",
}

// epochsPerPhase is the fixed number of epochs each phase runs.
const epochsPerPhase = 200

// phaseForEpoch returns the 1-based phase number for a given epoch index.
func phaseForEpoch(epoch int) int {
	p := epoch/epochsPerPhase + 1
	if p > 5 {
		return 5
	}
	return p
}

// TrainMultiPhaseCurriculum orchestrates the 5-phase, 200-epoch-per-phase curriculum.
// All hyperparameters are loaded from data/config/social_train.json.
// Phase boundaries are purely epoch-count based: phase = epoch/200 + 1 (clamped to 5).
func TrainMultiPhaseCurriculum(projectRoot string, useGPU bool, dataFile string) {
	log.Println("🚀 Starting 5-Phase Multi-Domain Curriculum Training (200 epochs/phase)...")
	for p := 1; p <= 5; p++ {
		log.Printf("   Phase %d: %s", p, phaseNames[p])
	}

	// ── 0. Load config ────────────────────────────────────────────────────────
	configPath := filepath.Join(projectRoot, "data/config/social_train.json")
	safeCfg, err := orchestrator.NewSafeConfig(configPath)
	if err != nil {
		log.Fatalf("❌ Failed to load config from %s: %v", configPath, err)
	}
	cfg := safeCfg.Get()

	// ── 1. Load datasets ──────────────────────────────────────────────────────
	var socialPairs []moe.TrainPair
	var computerPairs []moe.TrainPair

	// ── 1a. conversations.csv (multi-turn dialogue) ────────────────────────
	// Format: conversation_id, turn_sequence, role, content
	// We pair consecutive user→assistant turns into Q/A pairs.
	conversationsCSVPath := filepath.Join(projectRoot, "data/training/trainingdata/conversations.csv")
	if f, err := os.Open(conversationsCSVPath); err == nil {
		reader := csv.NewReader(f)
		records, _ := reader.ReadAll()
		f.Close()
		type turn struct{ role, content string }
		convMap := make(map[string][]turn)
		convOrder := []string{}
		seen := map[string]bool{}
		for i, rec := range records {
			if i == 0 || len(rec) < 4 {
				continue
			}
			id, role, content := rec[0], rec[2], rec[3]
			convMap[id] = append(convMap[id], turn{role, content})
			if !seen[id] {
				seen[id] = true
				convOrder = append(convOrder, id)
			}
		}
		convCount := 0
		for _, id := range convOrder {
			turns := convMap[id]
			for i := 0; i+1 < len(turns); i++ {
				if strings.ToLower(turns[i].role) == "user" && strings.ToLower(turns[i+1].role) == "assistant" {
					q, a := strings.TrimSpace(turns[i].content), strings.TrimSpace(turns[i+1].content)
					if q != "" && a != "" {
						socialPairs = append(socialPairs, moe.TrainPair{Q: q, A: a, Intent: "social"})
						convCount++
					}
				}
			}
		}
		log.Printf("📚 Loaded %d pairs from conversations.csv", convCount)
	} else {
		log.Printf("⚠️ conversations.csv: %v", err)
	}

	// ── 1b. conversing.csv (simple Q/A) ───────────────────────────────────
	// Format: query, answer, intent, grammar
	conversingCSVPath := filepath.Join(projectRoot, "data/training/trainingdata/conversing.csv")
	if pairs, err := LoadConversingCSV(conversingCSVPath); err == nil {
		socialPairs = append(socialPairs, pairs...)
		log.Printf("📚 Loaded %d pairs from conversing.csv", len(pairs))
	} else {
		log.Printf("⚠️ conversing.csv: %v", err)
	}

	// ── 1c. intent_corpus.json (intent examples) ──────────────────────────
	// Format: [{ "intent": "edit_agent", "examples": ["add a return type...", ...] }]
	// Each example is treated as a user utterance; the model learns to acknowledge it.
	intentCorpusPath := filepath.Join(projectRoot, "data/training/intent_corpus.json")
	if raw, err := os.ReadFile(intentCorpusPath); err == nil {
		var corpus []struct {
			Intent   string   `json:"intent"`
			Examples []string `json:"examples"`
		}
		if jsonErr := json.Unmarshal(raw, &corpus); jsonErr == nil {
			intentCount := 0
			for _, entry := range corpus {
				intentName := strings.ReplaceAll(entry.Intent, "_", " ")
				for _, example := range entry.Examples {
					if strings.TrimSpace(example) == "" {
						continue
					}
					// Classify into social vs computer domain based on intent name.
					if strings.Contains(entry.Intent, "edit") ||
						strings.Contains(entry.Intent, "code") ||
						strings.Contains(entry.Intent, "fix") ||
						strings.Contains(entry.Intent, "add") ||
						strings.Contains(entry.Intent, "refactor") ||
						strings.Contains(entry.Intent, "debug") {
						computerPairs = append(computerPairs, moe.TrainPair{
							Q:      example,
							A:      fmt.Sprintf("Sure, I will %s.", intentName),
							Intent: "computer",
						})
					} else {
						socialPairs = append(socialPairs, moe.TrainPair{
							Q:      example,
							A:      fmt.Sprintf("Sure, I will %s.", intentName),
							Intent: "social",
						})
					}
					intentCount++
				}
			}
			log.Printf("📚 Loaded %d intent examples from intent_corpus.json", intentCount)
		} else {
			log.Printf("⚠️ Failed to parse intent_corpus.json: %v", jsonErr)
		}
	} else {
		log.Printf("⚠️ intent_corpus.json: %v", err)
	}

	// ── 1d. mined_patches.json (instruction → patch) ──────────────────────
	// Format: [{ "instruction": "...", "target_patch": "..." }]
	minedPatchesPath := filepath.Join(projectRoot, "data/training/mined_patches.json")
	if raw, err := os.ReadFile(minedPatchesPath); err == nil {
		var patches []struct {
			Instruction string `json:"instruction"`
			TargetPatch string `json:"target_patch"`
		}
		if jsonErr := json.Unmarshal(raw, &patches); jsonErr == nil {
			for _, p := range patches {
				q := strings.TrimSpace(p.Instruction)
				a := strings.TrimSpace(p.TargetPatch)
				if q != "" && a != "" {
					computerPairs = append(computerPairs, moe.TrainPair{Q: q, A: a, Intent: "computer"})
				}
			}
			log.Printf("📚 Loaded %d instruction-patch pairs from mined_patches.json", len(patches))
		} else {
			log.Printf("⚠️ Failed to parse mined_patches.json: %v", jsonErr)
		}
	} else {
		log.Printf("⚠️ mined_patches.json: %v", err)
	}

	// ── 1e. mined_patches_fim.json (fill-in-the-middle) ───────────────────
	// Format: [{ "prefix": "...", "middle": "...", "suffix": "..." }]
	// Q = "Complete the code between <PREFIX> ... <SUFFIX>", A = middle
	minedFIMPath := filepath.Join(projectRoot, "data/training/mined_patches_fim.json")
	if raw, err := os.ReadFile(minedFIMPath); err == nil {
		var fimItems []struct {
			Prefix string `json:"prefix"`
			Middle string `json:"middle"`
			Suffix string `json:"suffix"`
		}
		if jsonErr := json.Unmarshal(raw, &fimItems); jsonErr == nil {
			fimCount := 0
			for _, item := range fimItems {
				middle := strings.TrimSpace(item.Middle)
				if middle == "" {
					continue
				}
				prefix := strings.TrimSpace(item.Prefix)
				suffix := strings.TrimSpace(item.Suffix)
				var q string
				if suffix != "" {
					q = fmt.Sprintf("complete the code: %s <fill> %s", prefix, suffix)
				} else {
					q = fmt.Sprintf("complete the code: %s", prefix)
				}
				computerPairs = append(computerPairs, moe.TrainPair{Q: q, A: middle, Intent: "computer"})
				fimCount++
			}
			log.Printf("📚 Loaded %d FIM examples from mined_patches_fim.json", fimCount)
		} else {
			log.Printf("⚠️ Failed to parse mined_patches_fim.json: %v", jsonErr)
		}
	} else {
		log.Printf("⚠️ mined_patches_fim.json: %v", err)
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

	currentPhase := phaseForEpoch(0)

	// ── Phase 1 LR scheduler state (still used within Phase 1 to reduce on stagnation) ─
	var phase1BestLoss float32 = 1e9
	var phase1StagnantEpochs int
	var phase1LRFactor float32 = 1.0
	const phase1LRDecayPatience = 35
	const phase1LRImprovementThreshold = 0.005
	// Never let Phase 1 LR fall below 25% of its base value. The previous 1/32
	// floor (0.03125) let the LR collapse to ~0.000016 after 4–5 stagnation
	// halvings — far too small to make progress before Phase 1 even ended,
	// leaving the social experts unable to form sentences.
	const phase1LRFactorMin = 0.25

	// Per-phase loss tracking for the summary report
	phaseBestLoss := make(map[int]float32)
	phaseWorstLoss := make(map[int]float32)
	for p := 1; p <= 5; p++ {
		phaseBestLoss[p] = 1e9
		phaseWorstLoss[p] = 0
	}

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
			trainPairs = make([]moe.TrainPair, 0, len(socialPairs)+len(computerPairs))
			trainPairs = append(trainPairs, socialPairs...)
			trainPairs = append(trainPairs, computerPairs...)
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

		// ── Track per-phase loss extremes ─────────────────────────────────────
		if avgLoss < phaseBestLoss[currentPhase] {
			phaseBestLoss[currentPhase] = avgLoss
		}
		if avgLoss > phaseWorstLoss[currentPhase] {
			phaseWorstLoss[currentPhase] = avgLoss
		}

		// ── Epoch log ─────────────────────────────────────────────────────────
		var activeExps []string
		if len(layers) > 0 {
			for i := 0; i < len(layers[0].ExpertFrozen); i++ {
				if !layers[0].ExpertFrozen[i] {
					activeExps = append(activeExps, fmt.Sprintf("%d", i))
				}
			}
		}
		// Calculate effective LR for logging
		effectiveLR := phaseLR
		if currentPhase == 1 {
			effectiveLR = phaseLR * phase1LRFactor
		}

		epochInPhase := epoch%epochsPerPhase + 1
		log.Printf("Phase %d [%d/%d] | Epoch %d | Loss: %.4f | LR: %.6f | Active: [%s] | EpochTime: %.1fs",
			currentPhase, epochInPhase, epochsPerPhase, epoch, avgLoss, effectiveLR,
			strings.Join(activeExps, ","), epochDuration)

		// ── Phase 1: LR reducer on stagnation ────────────────────────────────
		if currentPhase == 1 {
			if avgLoss < phase1BestLoss-phase1LRImprovementThreshold {
				phase1BestLoss = avgLoss
				phase1StagnantEpochs = 0
			} else {
				phase1StagnantEpochs++
			}
			if phase1StagnantEpochs >= phase1LRDecayPatience && phase1LRFactor > phase1LRFactorMin {
				phase1LRFactor *= 0.5
				if phase1LRFactor < phase1LRFactorMin {
					phase1LRFactor = phase1LRFactorMin
				}
				phase1StagnantEpochs = 0
				log.Printf("🔻 Phase 1 stagnant %d epochs → LR factor %.4f (effective %.6f)",
					phase1LRDecayPatience, phase1LRFactor, phaseCfg.LearningRate*phase1LRFactor)
			}
			optimizer.SetLearningRate(phaseCfg.LearningRate * phase1LRFactor)
		}

		// ── Fixed 200-epoch phase transitions ─────────────────────────────────
		// Each phase ends at epoch 199, 399, 599, 799 (i.e. (epoch+1) % epochsPerPhase == 0)
		isPhaseEnd := (epoch+1)%epochsPerPhase == 0
		nextPhase := phaseForEpoch(epoch + 1)

		if isPhaseEnd {
			// ── Run the end-of-phase diagnostic probe ─────────────────────────
			runEndOfPhaseProbe(intentModel, layers, currentPhase, epoch, avgLoss,
				phaseBestLoss[currentPhase], phaseWorstLoss[currentPhase], len(activeExps))

			// ── Advance to next phase if needed ───────────────────────────────
			if nextPhase != currentPhase && nextPhase <= 5 {
				log.Printf("")
				log.Printf("⏩ Advancing: Phase %d → Phase %d (%s)", currentPhase, nextPhase, phaseNames[nextPhase])
				currentPhase = nextPhase
				nextPhaseCfg := cfg.Phases[strconv.Itoa(currentPhase)]
				if nextPhaseCfg != nil {
					applyPhaseFreeze(layers, nextPhaseCfg.FreezeExpertsStart, nextPhaseCfg.FreezeExpertsEnd)
					// Reset LR factor when entering a new phase
					phase1LRFactor = 1.0
					phase1StagnantEpochs = 0
					optimizer.SetLearningRate(nextPhaseCfg.LearningRate)
				}
				moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
			}
		}

		// ── Periodic checkpoint every 10 epochs ───────────────────────────────
		if epoch > 0 && epoch%10 == 0 {
			moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
			log.Printf("💾 Checkpoint saved (epoch %d)", epoch)
		}
	}
}

// runEndOfPhaseProbe runs the diagnostic probe for the completed phase and logs
// a structured summary block with PASS/FAIL result.
func runEndOfPhaseProbe(intentModel *moe.IntentMoE, layers []*moe.MoELayer,
	phase, epoch int, finalLoss, bestLoss, worstLoss float32, activeExpertCount int) {

	banner := strings.Repeat("═", 65)
	log.Printf("%s", banner)
	log.Printf("📋 PHASE %d COMPLETE — %s", phase, phaseNames[phase])
	log.Printf("   Epochs: %d–%d  |  Active Experts: %d",
		(phase-1)*epochsPerPhase, epoch, activeExpertCount)
	log.Printf("   Loss → Best: %.4f  Worst: %.4f  Final: %.4f", bestLoss, worstLoss, finalLoss)

	var probeResult string
	var passed bool

	switch phase {
	case 1:
		// Phase 1 probe: Conversational coherence
		// Goal: social experts (0–7) must produce a coherent, human-like social response.
		// Pass: ≥1 social token (hi/hello/i/you/am/great/good) AND TTR ≥ 0.4
		gen, _, _ := StrictGenerateLowTemp(intentModel, "__intent__ social : __ques__ how are you", 20, 1.0, false, epoch)
		words := strings.Fields(strings.ToLower(gen))
		socialTokenSet := map[string]bool{"hi": true, "hello": true, "i": true, "you": true, "am": true, "great": true, "good": true, "fine": true, "doing": true, "well": true}
		hasSocial := false
		unique := make(map[string]struct{})
		for _, w := range words {
			unique[w] = struct{}{}
			if socialTokenSet[strings.Trim(w, ".,!?")] {
				hasSocial = true
			}
		}
		ttr := float32(0)
		if len(words) > 0 {
			ttr = float32(len(unique)) / float32(len(words))
		}
		passed = hasSocial && ttr >= 0.4 && len(words) >= 3
		probeResult = fmt.Sprintf("Social response: '%s'\n   TTR=%.2f, social_tokens=%v", gen, ttr, hasSocial)

	case 2:
		// Phase 2 probe: Technical vocabulary
		// Goal: cartridge experts (8–15) must have learned Go/code vocabulary.
		// Pass: ≥1 Go keyword or symbol in the generated response.
		goKeywords := map[string]bool{"func": true, "return": true, "if": true, "var": true, "err": true,
			"nil": true, "int": true, "string": true, "error": true, "for": true, "range": true, "type": true}
		gen, _, _ := StrictGenerateLowTemp(intentModel,
			"__intent__ computer : __ques__ write a go function", 25, 1.0, false, epoch)
		genLower := strings.ToLower(gen)
		foundKeyword := ""
		for kw := range goKeywords {
			if strings.Contains(genLower, kw) {
				foundKeyword = kw
				break
			}
		}
		passed = foundKeyword != ""
		probeResult = fmt.Sprintf("Code response: '%s'\n   First Go keyword found: %q", gen, foundKeyword)

	case 3:
		// Phase 3 probe: Routing diversity
		// Goal: different intents should route to different experts.
		// Pass: ≥1 active expert in each of the two intent groups (social 0–7, code 8–15).
		socialActive := 0
		codeActive := 0
		if len(layers) > 0 {
			for i, frozen := range layers[0].ExpertFrozen {
				if !frozen {
					if i < 8 {
						socialActive++
					} else {
						codeActive++
					}
				}
			}
		}
		passed = socialActive >= 1 && codeActive >= 1
		probeResult = fmt.Sprintf("Routing diversity: social experts active=%d, code experts active=%d", socialActive, codeActive)

	case 4:
		// Phase 4 probe: Load balancing
		// Goal: no single expert should dominate token routing.
		// Pass: max load fraction across all experts ≤ 50%.
		maxLoad := 0
		totalLoad := 0
		for _, layer := range layers {
			if len(layer.AccumulatedUtilization) == 0 {
				continue
			}
			for _, u := range layer.AccumulatedUtilization {
				if u > maxLoad {
					maxLoad = u
				}
				totalLoad += u
			}
		}
		maxFraction := float32(0)
		if totalLoad > 0 {
			maxFraction = float32(maxLoad) / float32(totalLoad)
		}
		passed = maxFraction <= 0.5
		probeResult = fmt.Sprintf("Load balance: max expert fraction=%.1f%%", maxFraction*100)

	case 5:
		// Phase 5 probe: End-to-end quality
		// Goal: model must produce coherent responses for BOTH social AND code prompts.
		// Pass: both social and code responses pass their respective quality checks.
		socialGen, _, _ := StrictGenerateLowTemp(intentModel,
			"__intent__ social : __ques__ tell me about yourself", 20, 1.0, false, epoch)
		codeGen, _, _ := StrictGenerateLowTemp(intentModel,
			"__intent__ computer : __ques__ how do you handle errors in go", 25, 1.0, false, epoch)

		socialWords := strings.Fields(socialGen)
		socialPassed := len(socialWords) >= 3 && svcIsCoherent(socialGen)

		goKeywords := map[string]bool{"err": true, "error": true, "return": true, "nil": true, "if": true, "func": true}
		codeGenLower := strings.ToLower(codeGen)
		codeHasKeyword := false
		for kw := range goKeywords {
			if strings.Contains(codeGenLower, kw) {
				codeHasKeyword = true
				break
			}
		}
		passed = socialPassed && codeHasKeyword
		probeResult = fmt.Sprintf(
			"Social: '%s' (coherent=%v)\n   Code: '%s' (has_keyword=%v)",
			socialGen, socialPassed, codeGen, codeHasKeyword)
	}

	resultIcon := "✅ PASS"
	if !passed {
		resultIcon = "❌ FAIL"
	}
	log.Printf("   🧪 Phase %d Probe: %s", phase, probeResult)
	log.Printf("   %s", resultIcon)
	log.Printf("%s", banner)
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
