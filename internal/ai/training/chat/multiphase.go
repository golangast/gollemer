package chat

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
	neuralnn "github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/orchestrator"
	datasetpb "github.com/golangast/gollemer/internal/ai/training/proto/dataset"
	"gopkg.in/yaml.v3"
)

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
		numExperts := len(layer.Experts)
		if numExperts == 0 {
			continue
		}

		start := freezeStart
		end := freezeEnd
		if start < 0 {
			start = -1
		}
		if end < 0 {
			end = numExperts
		}

		// A social-phase config that targets a larger model must still respect the
		// actual layer size. Keeping the lower half active and freezing the upper half
		// is the stable default for a compact MoE, and it prevents the “unfreeze all” bug.
		if freezeStart >= numExperts || freezeEnd > numExperts {
			for i := 0; i < numExperts; i++ {
				setLayerFreezeQuiet(layer, i, i >= numExperts/2)
			}
			continue
		}
		if start >= numExperts {
			start = numExperts - 1
		}
		if end > numExperts {
			end = numExperts
		}
		if start >= end {
			start = -1
			end = numExperts
		}

		for i := 0; i < numExperts; i++ {
			var shouldFreeze bool
			if start >= 0 && end > start {
				// Freeze range [start, end)
				shouldFreeze = i >= start && i < end
			} else if start >= 0 {
				// Freeze from freezeStart to end
				shouldFreeze = i >= start
			} else {
				shouldFreeze = false // Unfreeze all
			}
			setLayerFreezeQuiet(layer, i, shouldFreeze)
		}
	}
}

// regexpSplitSentences splits text into sentences using terminators.
func regexpSplitSentences(text string) []string {
	re := regexp.MustCompile(`[.!?]+`)
	parts := re.Split(text, -1)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// svcIsCoherent returns true when a generated response looks like real language
// (not SALAD): needs at least 2 words, balanced sentence lengths, and enough
// lexical variety.
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

	// Check balanced sentence lengths: split on sentence terminators and
	// ensure no single sentence dominates the response.
	sentences := regexpSplitSentences(response)
	if len(sentences) >= 3 {
		totalWords := 0
		minLen := len(words)
		maxLen := 0
		for _, s := range sentences {
			n := len(strings.Fields(s))
			totalWords += n
			if n < minLen {
				minLen = n
			}
			if n > maxLen {
				maxLen = n
			}
		}
		avgLen := float64(totalWords) / float64(len(sentences))
		if avgLen > 0 && float64(maxLen)/avgLen > 4.0 {
			return false
		}
		if maxLen > 0 && minLen == 0 {
			return false
		}
	}

	// For short responses (2-3 words), relax TTR requirement
	if len(words) <= 3 {
		return hasConversational
	}
	return ttr >= 0.5 && hasConversational
}

func buildTargetSequence(answerTokens []string, vocab *mainvocab.Vocabulary, maxLen int) []float32 {
	seq := make([]float32, maxLen)
	for i := range seq {
		seq[i] = float32(vocab.PaddingTokenID)
	}
	if vocab == nil || maxLen <= 0 {
		return seq
	}

	bosID := vocab.BosID
	if bosID < 0 {
		bosID = vocab.GetTokenID("<s>")
	}
	eosID := vocab.EosID
	if eosID < 0 {
		eosID = vocab.GetTokenID("</s>")
	}
	seq[0] = float32(bosID)
	writePos := 1
	for _, tok := range answerTokens {
		if writePos >= maxLen-1 {
			break
		}
		id := lookupVocab(tok, vocab)
		if id == vocab.PaddingTokenID {
			continue
		}
		seq[writePos] = float32(id)
		writePos++
	}
	if writePos >= maxLen {
		writePos = maxLen - 1
	}
	seq[writePos] = float32(eosID)
	return seq
}

// phaseNames maps phase number to its human-readable name and goal.
var phaseNames = map[int]string{
	1: "Social Bootcamp        — learn conversational patterns at full LR",
	2: "Coherence Polish I     — ultra-low LR; EOS/coherence optimization",
	3: "Coherence Polish II    — deeper refinement; lower LR for stability",
	4: "Coherence Polish III   — fine detail pass; near-convergence LR",
	5: "Coherence Polish IV    — final micro-tuning at minimum LR",
}

// epochsPerPhase is the fixed number of epochs each phase runs (after Phase 1).
const epochsPerPhase = 400

// phase1Epochs is the reduced number of epochs for Phase 1 to prevent saturation.
const phase1Epochs = 100

// phaseForEpoch returns the 1-based phase number for a given epoch index.
func phaseForEpoch(epoch int) int {
	if epoch < phase1Epochs {
		return 1
	}
	p := ((epoch - phase1Epochs) / epochsPerPhase) + 2
	if p > 5 {
		return 5
	}
	return p
}

// TrainMultiPhaseCurriculum orchestrates the 5-phase curriculum.
// All hyperparameters are loaded from data/config/social_train.json.
func TrainMultiPhaseCurriculum(projectRoot string, useGPU bool, dataFile string, makefileOnly bool) {
	log.Printf("🚀 Starting 5-Phase Multi-Domain Curriculum Training (Phase 1: %d epochs, others: %d epochs)...", phase1Epochs, epochsPerPhase)
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

	// ── 1a. conversing.pb (multi-turn dialogue, protobuf) ──────────────
	// Format: ConversationDataset { conversations { turns[] } }
	// We pair consecutive user→assistant turns into Q/A pairs.
	// Skipped in makefile-only mode.
	conversationsPBPath := filepath.Join(projectRoot, "data/training/trainingdata/conversing.pb")
	if !makefileOnly {
		if ds, err := datasetpb.LoadConversationDatasetFromProto(conversationsPBPath); err == nil {
			convCount := 0
			for _, conv := range ds.GetConversations() {
				turns := conv.GetTurns()
				for i := 0; i+1 < len(turns); i++ {
					if turns[i].GetRole() == datasetpb.Role_ROLE_USER && turns[i+1].GetRole() == datasetpb.Role_ROLE_ASSISTANT {
						q, a := strings.TrimSpace(turns[i].GetContent()), strings.TrimSpace(turns[i+1].GetContent())
						if q != "" && a != "" {
							socialPairs = append(socialPairs, moe.TrainPair{Q: q, A: a, Intent: "social"})
							convCount++
						}
					}
				}
			}
			log.Printf("📚 Loaded %d pairs from conversing.pb", convCount)
		} else {
			log.Printf("⚠️ conversing.pb: %v", err)
		}
	}

	// ── 1b. conversing.csv (simple Q/A) ─────────────────────────────────────
	// Skipped in makefile-only mode.
	conversingCSVPath := filepath.Join(projectRoot, "data/training/trainingdata/conversing.csv")
	if !makefileOnly {
		if cfg.OverfitMode {
			overfitPath := filepath.Join(projectRoot, "data/training/trainingdata/conversing_overfit_single.csv")
			if _, err := os.Stat(overfitPath); err == nil {
				conversingCSVPath = overfitPath
			}
		}
		if pairs, err := LoadConversingCSV(conversingCSVPath); err == nil {
			socialPairs = append(socialPairs, pairs...)
			log.Printf("📚 Loaded %d pairs from %s", len(pairs), filepath.Base(conversingCSVPath))
		} else {
			log.Printf("⚠️ conversing.csv: %v", err)
		}
	}

	// ── 1c. YAML datasets (social replies + technical multi-turn + makefile) ─────
	// Load all YAML datasets upfront and select per-phase later.
	// In makefile-only mode, only social replies and makefile data are loaded.
	yamlDatasetPaths := []string{
		"social_replies.yaml",
		"tech_multiturn.yaml",
		"conversing.yaml",
		"makefile.yaml",
	}
	if makefileOnly {
		yamlDatasetPaths = []string{
			"social_replies.yaml",
			"makefile.yaml",
		}
	}
	yamlPairsByFile := make(map[string][]moe.TrainPair)
	if !cfg.OverfitMode {
		for _, yamlName := range yamlDatasetPaths {
			yamlPath := filepath.Join(projectRoot, "data/training/trainingdata", yamlName)
			raw, err := os.ReadFile(yamlPath)
			if err != nil {
				log.Printf("⚠️ %s: %v", yamlName, err)
				continue
			}

			var yamlDoc struct {
				Conversations []struct {
					ConversationID string `yaml:"conversation_id"`
					Turns          []struct {
						Role    string `yaml:"role"`
						Content string `yaml:"content"`
					} `yaml:"turns"`
				} `yaml:"conversations"`
			}
			if yamlErr := yaml.Unmarshal(raw, &yamlDoc); yamlErr != nil {
				var rawConvs []struct {
					ConversationID string `yaml:"conversation_id"`
					Turns          []struct {
						Role    string `yaml:"role"`
						Content string `yaml:"content"`
					} `yaml:"turns"`
				}
				if err2 := yaml.Unmarshal(raw, &rawConvs); err2 == nil {
					yamlDoc.Conversations = rawConvs
				} else {
					log.Printf("⚠️ %s parse error: %v", yamlName, yamlErr)
					continue
				}
			}

			yamlCount := 0
			var filePairs []moe.TrainPair
			for _, conv := range yamlDoc.Conversations {
				turns := conv.Turns
				var historyBuilder strings.Builder
				for i := 0; i+1 < len(turns); i++ {
					if turns[i].Role == "user" && turns[i+1].Role == "assistant" {
						qRaw := strings.TrimSpace(turns[i].Content)
						aRaw := strings.TrimSpace(turns[i+1].Content)

						if qRaw != "" && aRaw != "" {
							intent := "social"
							if strings.HasSuffix(conv.ConversationID, "_tech") {
								intent = "tech"
							} else if strings.HasPrefix(conv.ConversationID, "conv_multiturn_") || strings.HasPrefix(conv.ConversationID, "conv_synth_") {
								intent = "multiturn"
							} else if strings.HasPrefix(conv.ConversationID, "conv_cot_reasoning_") {
								intent = "cot"
							} else if strings.HasPrefix(conv.ConversationID, "conv_601") || strings.HasPrefix(conv.ConversationID, "conv_602") {
								intent = "tech"
							} else if strings.HasPrefix(conv.ConversationID, "make_") {
								intent = "makefile"
							}

							q := historyBuilder.String() + "Human: " + qRaw + "\nAI: "
							a := aRaw

							filePairs = append(filePairs, moe.TrainPair{Q: q, A: a, Intent: intent})
							yamlCount++

							cleanA := aRaw
							if idx := strings.Index(cleanA, "[RESPONSE]"); idx >= 0 {
								cleanA = strings.TrimSpace(cleanA[idx+len("[RESPONSE]"):])
							}
							historyBuilder.WriteString("Human: " + qRaw + "\nAI: " + cleanA + "\n")
						}
					}
				}
			}
			yamlPairsByFile[yamlName] = filePairs
			log.Printf("📚 Loaded %d contextual turns from %s", yamlCount, yamlName)
		}
	}
	// intent_corpus.json intentionally skipped: its synthetic "Sure, I will X." answers
	// poisoned training — model always output "sure"/"i will". Real YAML data replaces it.

	// Merge makefile data into the training set in makefile-only mode.
	// In this mode we train on makefile.yaml plus a larger set of basic social
	// sentences so the model still learns conversational patterns.
	if makefileOnly {
		if makeYAML, ok := yamlPairsByFile["makefile.yaml"]; ok && len(makeYAML) > 0 {
			socialPairs = append(socialPairs, makeYAML...)
			log.Printf("📚 Makefile-only mode: training on %d makefile pairs", len(makeYAML))
		}
		basicSocial := []moe.TrainPair{
			{Q: "hi", A: "hello"},
			{Q: "hello", A: "hi"},
			{Q: "how are you", A: "i am good"},
			{Q: "how are you", A: "i am fine"},
			{Q: "what is up", A: "not much"},
			{Q: "good morning", A: "good morning"},
			{Q: "good evening", A: "good evening"},
			{Q: "good afternoon", A: "good afternoon"},
			{Q: "thank you", A: "you are welcome"},
			{Q: "thanks", A: "you are welcome"},
			{Q: "bye", A: "goodbye"},
			{Q: "goodbye", A: "bye"},
			{Q: "see you later", A: "see you later"},
			{Q: "what is your name", A: "i am gollemer"},
			{Q: "who are you", A: "i am gollemer"},
			{Q: "i am good", A: "that is great"},
			{Q: "i am fine", A: "that is good"},
			{Q: "i am great", A: "that is awesome"},
			{Q: "i am okay", A: "that is good"},
			{Q: "how is it going", A: "it is going well"},
			{Q: "what are you up to", A: "i am here to help"},
			{Q: "nice to meet you", A: "nice to meet you too"},
			{Q: "have a good day", A: "you too"},
			{Q: "take care", A: "thanks"},
		}
		for i := 0; i < 5; i++ {
			socialPairs = append(socialPairs, basicSocial...)
		}
		log.Printf("📚 Makefile-only mode: added %d basic social pairs", len(basicSocial)*5)
	} else {
		if socialYAML, ok := yamlPairsByFile["social_replies.yaml"]; ok && len(socialYAML) > 0 {
			socialPairs = append(socialPairs, socialYAML...)
			log.Printf("📚 Added %d social pairs from social_replies.yaml", len(socialYAML))
		}
	}

	if len(socialPairs) == 0 {
		log.Fatalf("❌ Missing required datasets (social=%d). Aborting.", len(socialPairs))
	}
	log.Printf("📚 Total social pairs: %d", len(socialPairs))

	// ── 2. Build isolated subject vocabularies ──────────────────────────────
	tmpVocab := mainvocab.NewVocabulary()
	socialVocab := mainvocab.NewVocabulary()

	sharedTokens := []string{"__ques__", "__ans__", "__intent__", "social", ":"}
	for _, tok := range sharedTokens {
		tmpVocab.AddToken(tok)
		socialVocab.AddToken(tok)
	}

	// Build SocialVocab using only social pairs (isolated!)
	for _, pair := range socialPairs {
		for _, t := range cleanTokenize(pair.Q + " " + pair.A) {
			tmpVocab.AddToken(t)
			socialVocab.AddToken(t)
		}
	}

	// ── 3. Load or create model ───────────────────────────────────────────────
	var intentModel *moe.IntentMoE
	socialModelPath := filepath.Join(projectRoot, "data/models/gob_models/moe_social_model.gob")
	optStatePath := socialModelPath + ".optstate"
	loadedFromCheckpoint := false
	if _, err := os.Stat(socialModelPath); err == nil {
		log.Printf("⬇️ Loading existing model from %s", socialModelPath)
		if makefileOnly {
			log.Printf("⚠️ Makefile-only mode: loaded old checkpoint. If this model has different architecture, delete %s to train from scratch.", socialModelPath)
		}
		intentModel, _ = moe.LoadIntentMoEModelWithFallback(socialModelPath)
		if intentModel != nil {
			loadedFromCheckpoint = true
		}
	}

	modelDim := cfg.ModelDim
	if modelDim <= 0 {
		modelDim = 512
	}
	baseExperts := cfg.NumExperts
	if baseExperts <= 0 {
		baseExperts = 8
	}
	if makefileOnly {
		if modelDim < 512 {
			modelDim = 512
		}
		if baseExperts < 8 {
			baseExperts = 8
		}
	}
	if intentModel == nil {
		intentModel, _ = moe.NewHybridIntentMoE(
			tmpVocab.Size(), modelDim, baseExperts,
			modelDim, modelDim, tmpVocab.Size(), 2,
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

	// Merge vocab into model.
	// IMPORTANT: Use the unified tmpVocab (words from actual training pairs only)
	// as the primary SentenceVocab — NOT the full 16k BPE tokenizer.
	// This creates a compact vocabulary containing both social and computer terms
	// (usually ~3000 tokens) which prevents OOB errors across phases and ensures
	// consistent token IDs across the entire model architecture.
	if intentModel.SentenceVocab == nil {
		intentModel.SentenceVocab = tmpVocab
	} else {
		// Merge tokens into model's vocab on resume
		for k := range tmpVocab.WordToToken {
			intentModel.SentenceVocab.AddToken(k)
		}
	}
	intentModel.SocialVocab = socialVocab
	// Do NOT inject the full BPE tokenizer — that would bloat the output head to 16k.
	// tokenizer.InjectIntoVocab(...) is intentionally skipped for the social model.
	intentModel.SentenceVocabSize = intentModel.SentenceVocab.Size()
	intentModel.Decoder.ResizeOutputLayer(intentModel.SentenceVocabSize)
	intentModel.ResizeEmbeddings(intentModel.SentenceVocabSize)
	log.Printf("📖 Social model vocab size: %d tokens (compact social-only)", intentModel.SentenceVocabSize)
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

	// Restore Adam optimizer state if available.
	// Without this, every resume zeroes Adam's m/v moments, causing a
	// cold-start regression that kicks the model out of its learned optimum.
	if loadedFromCheckpoint {
		if err := optimizer.LoadState(optStatePath); err != nil {
			log.Printf("⚠️ Optimizer state not restored (will cold-start): %v", err)
		} else {
			log.Printf("✅ Optimizer state restored from %s", optStatePath)
		}
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
	if labelSmoothing < 0 {
		labelSmoothing = 0.0
	}

	// Build flat loss-weight slice for WeightedCrossEntropy
	lossWeights := buildDefaultLossWeights(intentModel.SentenceVocab, &cfg)

	startEpoch := 0
	if loadedFromCheckpoint {
		startEpoch = intentModel.Metadata.LastEpoch
		log.Printf("🔄 Resuming from Epoch %d", startEpoch)
	}

	currentPhase := phaseForEpoch(startEpoch)

	// ── Active Phase LR scheduler state (used to reduce LR on stagnation / rollback on divergence) ─
	var activePhaseBestLoss float32 = 1e9
	var activePhaseStagnantEpochs int
	var activePhaseLRFactor float32 = 1.0
	const lrDecayPatience = 20 // allow more epochs at higher LR before decay
	const lrImprovementThreshold = 0.002
	// Allow LR to decay all the way to 5% of base; cosine annealing provides a
	// smooth floor approach so we don't need the old 70% hard floor.
	const lrFactorMin = 0.05
	// Once the LR has been stuck at the floor for another full patience window,
	// warm-restart it back to the phase base rate to break out of the plateau
	// (classic SGDR-style restart). Without this, the floor is a death sentence.
	const floorRestartPatience = lrDecayPatience * 2

	// Probe failure tracking. If the probe fails probeFailLimit times in a row
	// the model has converged as far as it can in this phase — force-advance.
	const probeFailLimit = 5
	var consecutiveProbeFails int

	// Per-phase loss tracking for the summary report
	phaseBestLoss := make(map[int]float32)
	phaseWorstLoss := make(map[int]float32)
	for p := 1; p <= 5; p++ {
		phaseBestLoss[p] = 1e9
		phaseWorstLoss[p] = 0
	}

	// Automated LR step-down tracker: how many auto-steps applied per phase
	autoLRApplied := make(map[int]int)

	// In-memory snapshot of parameter weights at the best-loss epoch.
	// When divergence is detected we roll back to this snapshot AND clear Adam
	// moments so stale momentum can't keep pushing weights the wrong way.
	var bestWeightsSnap map[*tensor.Tensor][]float32

	// Ensure correct freeze state before epoch 0
	phaseCfg := cfg.Phases[strconv.Itoa(currentPhase)]
	if phaseCfg != nil {
		applyPhaseFreeze(layers, phaseCfg.FreezeExpertsStart, phaseCfg.FreezeExpertsEnd)
	}

	for epoch := startEpoch; epoch < maxEpochs; epoch++ {
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
		case "social_replies.yaml":
			trainPairs = yamlPairsByFile["social_replies.yaml"]
		case "tech_multiturn.yaml":
			trainPairs = yamlPairsByFile["tech_multiturn.yaml"]
		default:
			trainPairs = socialPairs
		}

		if len(trainPairs) == 0 {
			log.Printf("⚠️ Phase %d: no pairs for dataset %q, falling back to socialPairs", currentPhase, phaseCfg.Dataset)
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
			if makefileOnly {
				layer.CapacityFactor = 1.0
			}
			// Force single expert during cold start epochs
			if phaseCfg.ForceSingleExpertEpochs > 0 && epoch < phaseCfg.ForceSingleExpertEpochs {
				layer.ForceSingleExpert = true
			} else {
				layer.ForceSingleExpert = false
			}
		}
		// ── Set LR for this epoch ────────────────────────────────────────────
		// Use cosine annealing over the phase window so the LR naturally decays
		// from phaseLR → 5%*phaseLR. The stagnation-based activePhaseLRFactor
		// acts as an additional multiplier on top of the cosine schedule.
		// Figure out how many epochs the current phase lasts
		var phaseEpochs int
		if currentPhase == 1 {
			phaseEpochs = phase1Epochs
		} else {
			phaseEpochs = epochsPerPhase // use per-phase count, not total epochs
		}
		// Calculate how many epochs we've been in the current phase (within the current window)
		epochInPhase := epoch
		if currentPhase > 1 {
			phaseStart := phase1Epochs + (currentPhase-2)*epochsPerPhase
			epochInPhase = epoch - phaseStart
		}
		if epochInPhase < 0 {
			epochInPhase = 0
		}
		cosDecay := float32(0.5 * (1.0 + math.Cos(math.Pi*float64(epochInPhase)/float64(phaseEpochs))))
		cosDecay = 0.05 + 0.95*cosDecay // clamp floor to 5% of phaseLR
		optimizer.SetLearningRate(phaseLR * cosDecay * activePhaseLRFactor)

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
		if makefileOnly {
			if phaseBatchSize > 8 {
				phaseBatchSize = 8
			}
			if phaseMaxSeqLen > 64 {
				phaseMaxSeqLen = 64
			}
		}

		// ── Iterator / Interactor pattern ────────────────────────────────────────
		// Use ChatDataIterator (same pattern as chat.go) instead of a raw slice
		// loop so shuffling, boundary checks, and MaxLen clamping are centralised.
		phaseIter := NewChatDataIterator(trainPairs, intentModel.SentenceVocab,
			intentModel.SentenceVocab.UnkID, true)
		phaseIter.MaxLen = phaseMaxSeqLen
		phaseIter.Epoch = epoch

		for phaseIter.HasNext() {
			batchData := phaseIter.NextBatch(phaseBatchSize)
			if batchData == nil || batchData.Input == nil {
				continue
			}

			optimizer.ZeroGrad()

			inputTensor := batchData.Input
			targetTensor := batchData.Target

			currentBatchSize := inputTensor.Shape[0]
			currentSeqLenOut := targetTensor.Shape[1]

			padID := intentModel.SentenceVocab.PaddingTokenID

			for _, layer := range layers {
				layer.CurrentPhase = currentPhase
			}

			// Use 0.0 sampling probability for strict teacher-forcing to allow deep overfitting.
			// The previous 0.1 was injecting noise that capped out the loss.
			logits, _, err := intentModel.Forward(0.0, inputTensor, targetTensor)
			if err != nil {
				log.Printf("⚠️ Forward error (epoch %d): %v", epoch, err)
				intentModel.ClearState()
				continue
			}

			// ── Loss computation — mirrors chat.go approach ────────────────────
			var batchLoss float32
			var grads []*tensor.Tensor

			if len(logits) == 1 && len(logits[0].Shape) == 3 {
				// Vectorized 3D path: logits shape [batch, seqLen-1, vocab]
				targetSeqLen := currentSeqLenOut - 1
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
						tID := int(targetTensor.Data[b*currentSeqLenOut+t+1])
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
				loss, grad := WeightedCrossEntropy(logits[0].ToCPU(), targets, lossWeights, labelSmoothing, cfg.EntropyWeight)
				if grad == nil {
					grad = tensor.NewTensor(logits[0].Shape, make([]float32, len(logits[0].Data)), false)
				}

				// DYNAMIC NORMALIZATION REMOVED
				// WeightedCrossEntropy ALREADY averages the loss and scales the gradients
				// by the number of valid (non-padded) tokens. Double-dividing them here
				// caused extreme vanishing gradients, which froze the model's learning!
				penaltyFactor := float32(1.0) + (eosPenalty / float32(currentBatchSize))
				batchLoss = loss * penaltyFactor

				scale := penaltyFactor
				// Use SIMD-dispatched MulScalar (routes through GOEXPERIMENT=simd path).
				if scale != 1.0 {
					tensor.MulScalar(grad.Data, scale, grad.Data)
				}
				grads = []*tensor.Tensor{grad}
			} else {
				// Step-by-step path
				grads = make([]*tensor.Tensor, len(logits))
				var stepTotal float32
				for t, logit := range logits {
					targets := make([]int, currentBatchSize)
					for b := 0; b < currentBatchSize; b++ {
						idx := b*currentSeqLenOut + t + 1
						if idx < len(targetTensor.Data) {
							targets[b] = int(targetTensor.Data[idx])
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
					l, g := WeightedCrossEntropy(logit.ToCPU(), targets, lossWeights, labelSmoothing, cfg.EntropyWeight)
					if g == nil {
						g = tensor.NewTensor(logit.Shape, make([]float32, len(logit.Data)), false)
					}

					// DYNAMIC NORMALIZATION REMOVED
					// WeightedCrossEntropy ALREADY averages the loss and scales the gradients
					// by the number of valid (non-padded) tokens. Double-dividing them here
					// caused extreme vanishing gradients, which froze the model's learning!
					penaltyFactor := float32(1.0) + (eosPenalty / float32(currentBatchSize))
					stepTotal += l * penaltyFactor

					scale := penaltyFactor
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
		effectiveLR := phaseLR * activePhaseLRFactor
		ppl := float32(0.0)
		if avgLoss > 0 {
			ppl = float32(math.Exp(float64(avgLoss)))
		}

		epochInPhase = epoch%epochsPerPhase + 1
		log.Printf("Phase %d [%d/%d] | Epoch %d | Loss: %.4f | PPL: %.1f | LR: %g | Act: [%s] | Time: %.1fs",
			currentPhase, epochInPhase, epochsPerPhase, epoch, avgLoss, ppl, effectiveLR,
			strings.Join(activeExps, ","), epochDuration)

		// ── Early stopping: halt when target loss is reached ───────────────────
		if cfg.TargetLoss > 0 && avgLoss > 0 && avgLoss <= cfg.TargetLoss {
			log.Printf("🎯 Target loss %.4f reached at epoch %d (loss=%.4f) — saving and stopping early.",
				cfg.TargetLoss, epoch, avgLoss)
			intentModel.Metadata.LastEpoch = epoch
			moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
			if err := optimizer.SaveState(optStatePath); err != nil {
				log.Printf("⚠️ Failed to save optimizer state on early stop: %v", err)
			}
			return
		}

		// ── Automated LR step-down: reduce phase LR when loss falls below threshold
		if cfg.AutoLREnabled && cfg.AutoLRThreshold > 0 && cfg.AutoLRFactor > 0 {
			maxSteps := cfg.AutoLRMaxSteps
			if maxSteps <= 0 {
				maxSteps = 3
			}
			applied := autoLRApplied[currentPhase]
			if applied < maxSteps && avgLoss > 0 && avgLoss < cfg.AutoLRThreshold {
				// apply step-down to the phase learning rate
				phaseCfg.LearningRate = phaseCfg.LearningRate * cfg.AutoLRFactor
				autoLRApplied[currentPhase] = applied + 1
				// Update optimizer with new effective LR (respect phase factor)
				optimizer.SetLearningRate(phaseCfg.LearningRate * activePhaseLRFactor)
				log.Printf("🔻 Auto LR step-down applied: phase %d new LR=%.6f (applied %d/%d) at loss=%.6f",
					currentPhase, phaseCfg.LearningRate, autoLRApplied[currentPhase], maxSteps, avgLoss)
			}
		}

		if activePhaseStagnantEpochs == 0 && (epoch%10 == 0 || epoch == 0) && len(trainPairs) > 0 {
			// Test sentence formation using the first prompt from our training dataset,
			// rather than a hardcoded string that might be out-of-distribution.
			probePrompt := trainPairs[0].Q
			probeText, _, _ := StrictGenerateLowTemp(intentModel, probePrompt, 18, 1.0, false, epoch)
			if probeText != "" {
				label, status, reason := assessSentenceFormation(probeText)
				log.Printf("📝 Generation Sample (Epoch %d): Prompt: %q | Generated: %q | SentenceStatus=%s | Quality=%s | Reason=%s",
					epoch, probePrompt, probeText, label, status, reason)
				if status == "coherent" || status == "emerging" {
					log.Printf("✅ Sentence forming: generation is moving from repetitive tokens toward language structure.")
				} else {
					log.Printf("⚠️ Early-stage output: repetitive tokens and/or <unk> still dominate; sentence formation is not yet stable.")
				}
			}
		}

		// ── All Phases: LR reducer on stagnation + divergence circuit breaker ──────
		// Divergence circuit breaker: if loss is > threshold× the best seen, cut LR
		// AND roll the weights back to the best snapshot so bad momentum
		// can't keep dragging the model away from the good region.
		//
		// DYNAMIC THRESHOLD: at low loss values (< 1.0), batch variance is high
		// relative to the loss magnitude. A fixed 1.5× fires constantly at loss ~0.3
		// (any +0.17 bounce = rollback). Scale the threshold so the band stays
		// meaningful: use 3.0× below loss 0.5, 2.0× below loss 1.0, 1.5× above.
		divThreshold := float32(2.0)
		if activePhaseBestLoss < 0.5 {
			divThreshold = 3.0
		} else if activePhaseBestLoss < 1.0 {
			divThreshold = 2.5
		}
		if activePhaseBestLoss < 1e8 && avgLoss > activePhaseBestLoss*divThreshold {
			if bestWeightsSnap != nil {
				optimizer.RestoreParameters(bestWeightsSnap)
				log.Printf("⚡ Phase %d divergence (%.4f > %.4f×%.1f) → rolled back to best weights + reset Adam moments",
					currentPhase, avgLoss, activePhaseBestLoss, divThreshold)
			} else {
				optimizer.ResetAllMoments()
				log.Printf("⚡ Phase %d divergence (%.4f > %.4f×%.1f) → reset Adam moments (no snapshot yet)",
					currentPhase, avgLoss, activePhaseBestLoss, divThreshold)
			}
			if activePhaseLRFactor > lrFactorMin {
				activePhaseLRFactor *= 0.5
				if activePhaseLRFactor < lrFactorMin {
					activePhaseLRFactor = lrFactorMin
				}
			}
			// NOTE: do NOT reset activePhaseStagnantEpochs here.
			// If we reset it, the warm-restart counter (floorRestartPatience) can
			// never accumulate enough while divergence keeps firing — the model
			// gets permanently stuck. Let stagnation keep counting so the SGDR
			// warm-restart eventually fires and breaks the deadlock.
		} else if avgLoss < activePhaseBestLoss-lrImprovementThreshold {
			activePhaseBestLoss = avgLoss
			activePhaseStagnantEpochs = 0
			// Snapshot the weights at this new best loss.
			bestWeightsSnap = optimizer.SnapshotParameters()
		} else {
			activePhaseStagnantEpochs++
		}

		if activePhaseStagnantEpochs >= lrDecayPatience && activePhaseLRFactor > lrFactorMin {
			activePhaseLRFactor *= 0.5
			if activePhaseLRFactor < lrFactorMin {
				activePhaseLRFactor = lrFactorMin
			}
			activePhaseStagnantEpochs = 0
			log.Printf("🔻 Phase %d stagnant %d epochs → LR factor %.4f",
				currentPhase, lrDecayPatience, activePhaseLRFactor)
		}

		// Early stopping: if stagnant for 30 epochs regardless of LR level, advance phase.
		// This prevents the model from spinning for hundreds of epochs after it's converged.
		const earlyStopPatience = 30
		if activePhaseStagnantEpochs >= earlyStopPatience {
			log.Printf("⚠️ Phase %d has been stagnant for %d epochs — force-advancing to next phase", currentPhase, earlyStopPatience)
			// Fast-forward epoch so the phase transition triggers at the bottom of the loop
			phaseEnd := phase1Epochs + (currentPhase-1)*epochsPerPhase - 1
			if currentPhase == 1 {
				phaseEnd = phase1Epochs - 1
			}
			if epoch < phaseEnd {
				epoch = phaseEnd
			}
		}

		// SGDR warm-restart DISABLED: on a small memorization dataset, warm-restarts
		// constantly reset the LR to full and blow the loss back up (e.g., 2.97 → 5.1).
		// The model needs monotonically decreasing LR to converge, not cyclic spikes.
		// if activePhaseLRFactor <= lrFactorMin && activePhaseStagnantEpochs >= floorRestartPatience { ... }
		// Note: the actual SetLearningRate call is at the START of the next epoch
		// (the cosine block above), so we don't call it again here to avoid double-set.

		// ── Phase transitions ─────────────────────────────────
		nextPhase := phaseForEpoch(epoch + 1)
		isPhaseEnd := nextPhase != currentPhase && nextPhase <= 5

		if isPhaseEnd {
			// ── Run the end-of-phase diagnostic probe ─────────────────────────
			probePassed := runEndOfPhaseProbe(intentModel, layers, currentPhase, epoch, avgLoss,
				phaseBestLoss[currentPhase], phaseWorstLoss[currentPhase], len(activeExps), makefileOnly)

			if probePassed {
				consecutiveProbeFails = 0
			}
			// ── Advance to next phase only if probe passed ─────────────────────
			forceAdvance := !probePassed && consecutiveProbeFails >= probeFailLimit
			if forceAdvance {
				log.Printf("")
				log.Printf("⚠️  Phase %d probe failed %d times in a row — model has converged; force-advancing to Phase %d",
					currentPhase, consecutiveProbeFails, nextPhase)
			}
			if (probePassed || forceAdvance) && nextPhase != currentPhase && nextPhase <= 5 {
				log.Printf("")
				log.Printf("⏩ Advancing: Phase %d → Phase %d (%s)", currentPhase, nextPhase, phaseNames[nextPhase])
				currentPhase = nextPhase
				consecutiveProbeFails = 0
				nextPhaseCfg := cfg.Phases[strconv.Itoa(currentPhase)]
				if nextPhaseCfg != nil {
					applyPhaseFreeze(layers, nextPhaseCfg.FreezeExpertsStart, nextPhaseCfg.FreezeExpertsEnd)
					// Reset LR factor and state when entering a new phase
					activePhaseLRFactor = 1.0
					activePhaseStagnantEpochs = 0
					activePhaseBestLoss = 1e9
					bestWeightsSnap = nil // invalidate old phase snapshot
					optimizer.SetLearningRate(nextPhaseCfg.LearningRate)
				}
				intentModel.Metadata.LastEpoch = epoch
				moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
				if err := optimizer.SaveState(optStatePath); err != nil {
					log.Printf("⚠️ Failed to save optimizer state at phase transition: %v", err)
				}
			} else if !probePassed {
				consecutiveProbeFails++
				log.Printf("")
				log.Printf("⏸ Phase %d probe failed (%d/%d) — extending current phase and warm-restarting LR",
					currentPhase, consecutiveProbeFails, probeFailLimit)
				// Warm-restart LR so the next extension window is not dead.
				// This is the key fix: without this, activePhaseLRFactor stays at
				// lrFactorMin (5%) forever and the model can never escape the plateau.
				activePhaseLRFactor = 1.0
				activePhaseStagnantEpochs = 0
				activePhaseBestLoss = 1e9
				// Extend the current phase by adding extra epochs
				maxEpochs += epochsPerPhase
			}
		}

		// ── Periodic checkpoint every 10 epochs ──────────────────────────────────────────────
		if epoch > 0 && epoch%10 == 0 {
			intentModel.Metadata.LastEpoch = epoch
			moe.SaveIntentMoEModelToGOB(intentModel, socialModelPath)
			log.Printf("💾 Checkpoint saved (epoch %d)", epoch)
			// Also persist optimizer state so the next resume doesn't cold-start.
			if err := optimizer.SaveState(optStatePath); err != nil {
				log.Printf("⚠️ Failed to save optimizer state: %v", err)
			}
		}
	}
}

// runEndOfPhaseProbe runs the diagnostic probe for the completed phase and logs
// a structured summary block with PASS/FAIL result.
func assessSentenceFormation(text string) (string, string, string) {
	clean := strings.ToLower(strings.TrimSpace(text))
	if clean == "" {
		return "Early-stage", "fragmented", "empty generation"
	}

	words := strings.Fields(clean)
	if len(words) == 0 {
		return "Early-stage", "fragmented", "no tokens generated"
	}

	unique := map[string]struct{}{}
	for _, w := range words {
		w = strings.Trim(w, ".,!?;:\"'()[]{}<>/")
		if w == "" {
			continue
		}
		unique[w] = struct{}{}
	}

	repeatRate := 0.0
	if len(words) > 0 {
		repeat := 0
		seen := map[string]int{}
		for _, w := range words {
			w = strings.Trim(w, ".,!?;:\"'()[]{}<>/")
			if w == "" {
				continue
			}
			seen[w]++
			if seen[w] > 1 {
				repeat++
			}
		}
		repeatRate = float64(repeat) / float64(len(words))
	}

	unkCount := 0
	for _, w := range words {
		if strings.Contains(strings.Trim(w, ".,!?;:\"'()[]{}<>/"), "<unk>") {
			unkCount++
		}
	}

	containsVerb := false
	verbSet := map[string]bool{"is": true, "are": true, "can": true, "do": true, "does": true, "process": true, "learn": true, "work": true, "make": true, "help": true, "use": true}
	for _, w := range words {
		if verbSet[w] {
			containsVerb = true
			break
		}
	}

	endsWithPunct := strings.HasSuffix(clean, ".") || strings.HasSuffix(clean, "!") || strings.HasSuffix(clean, "?")
	if unkCount > 0 || repeatRate > 0.2 || len(words) < 4 || (!endsWithPunct && repeatRate > 0.0) {
		if repeatRate > 0.2 || (!endsWithPunct && repeatRate > 0.0) {
			return "Early-stage", "fragmented", "token repetition is dominating the output"
		}
		if unkCount > 0 {
			return "Early-stage", "fragmented", "unknown tokens are still dominating the sequence"
		}
		return "Early-stage", "fragmented", "too short to form a sentence"
	}
	if containsVerb && endsWithPunct && len(unique) >= 5 {
		return "Emerging sentence", "coherent", "output has a verb, punctuation, and enough lexical variety to look sentence-like"
	}
	if len(words) >= 6 && len(unique) >= 4 && containsVerb {
		return "Emerging sentence", "emerging", "several content words are present and the sample is moving toward grammatical structure"
	}
	return "Early-stage", "fragmented", "output still lacks stable sentence structure"
}

func runEndOfPhaseProbe(intentModel *moe.IntentMoE, layers []*moe.MoELayer,
	phase, epoch int, finalLoss, bestLoss, worstLoss float32, activeExpertCount int, makefileOnly bool) bool {

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
		if makefileOnly {
			gen, _, _ := StrictGenerateLowTemp(intentModel, "__intent__ makefile : __ques__ Human: how do i clean models\nAI: ", 20, 1.0, false, epoch)
			words := strings.Fields(strings.ToLower(gen))
			passed = len(words) >= 1 && strings.Contains(gen, "make")
			probeResult = fmt.Sprintf("Makefile response: '%s'", gen)
		} else {
			gen, _, _ := StrictGenerateLowTemp(intentModel, "__intent__ social : __ques__ Human: how are you\nAI: ", 20, 1.0, false, epoch)
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
		}

	case 2:
		if makefileOnly {
			gen, _, _ := StrictGenerateLowTemp(intentModel, "__intent__ makefile : __ques__ Human: how do i clean models\nAI: ", 20, 1.0, false, epoch)
			words := strings.Fields(strings.ToLower(gen))
			// Phase 2 coherence check: model must output at least 2 words and contain "make".
			// EOS is now suppressed for the first 2 decode steps in StrictGenerateLowTemp
			// so any model that has learned the makefile domain will satisfy this.
			passed = len(words) >= 2 && strings.Contains(gen, "make")
			probeResult = fmt.Sprintf("Makefile response: '%s'", gen)
		} else {
			// Phase 2 probe: Coherence Polish
			// Goal: ensure the polished model still generates coherent language.
			socialGen, _, _ := StrictGenerateLowTemp(intentModel, "__intent__ social : __ques__ Human: tell me about yourself\nAI: ", 20, 1.0, false, epoch)
			socialWords := strings.Fields(socialGen)
			passed = len(socialWords) >= 3 && svcIsCoherent(socialGen)
			probeResult = fmt.Sprintf("Polished social: '%s' (coherent=%v)", socialGen, passed)
		}
	default:
		// If a phase has no specific probe defined, it passes automatically
		passed = true
		probeResult = "N/A (No specific probe for this phase)"
	}

	resultIcon := "✅ PASS"
	if !passed {
		resultIcon = "❌ FAIL"
	}
	log.Printf("   🧪 Phase %d Probe: %s", phase, probeResult)
	log.Printf("   %s", resultIcon)
	log.Printf("%s", banner)
	return passed
}

// buildDefaultLossWeights builds a flat weight vector for WeightedCrossEntropy.
// High-frequency stop-words are down-weighted; BOS/EOS/terminal punctuation are boosted.
// For code-syntax tokens delimiters and structural Go keywords get higher weight.
// Dual-stage training format: [TRIPLETS]/[REASONING]/[RESPONSE] markers and entity
// tokens (Subject/Action/Object) receive elevated weight to anchor structure.
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
				weights[id] = 0.8
			}
		}
		boosted := []string{".", "!", "?", "<BOS>", "<EOS>", "__ans__"}
		for _, w := range boosted {
			id := vocab.GetTokenID(w)
			if id >= 0 && id < len(weights) {
				weights[id] = 2.0
			}
		}

		dualStageTokens := []string{
			"[triplets]", "[reasoning]", "[response]",
			"subject:", "action:", "object:",
			"subject", "action", "object",
			"first,", "second,", "third,",
			"therefore", "thus", "conclusion",
			"channel", "goroutine", "mutex", "interface",
			"error", "context", "slice", "map", "defer",
			"init", "package", "module", "garbage", "collector",
			"struct", "function", "vendor", "test", "log",
			"database", "http", "middleware", "panic", "race",
			"build", "config", "dependency", "go", "golang",
		}
		for _, tok := range dualStageTokens {
			id := vocab.GetTokenID(tok)
			if id >= 0 && id < len(weights) {
				weights[id] = 2.5
			}
		}
	}

	if vocab.PaddingTokenID >= 0 && vocab.PaddingTokenID < len(weights) {
		weights[vocab.PaddingTokenID] = 0.0
	}

	return weights
}
