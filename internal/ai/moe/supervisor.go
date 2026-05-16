package moe

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// Supervisor monitors the training state and performs autonomous repairs on MoE layers.
type Supervisor struct {
	BestPerplexity       float32
	PlateauCount         int
	MumbleCount          int
	LastHealStep         int
	JustPerformedSurgery bool
	TemporalAttention    *nn.MultiHeadAttention
	GrammarJudge         *nn.Linear

	// Per-expert variable overrides (layerIdx -> expertIdx -> ExpertConfig)
	expertOverrides map[int]map[int]*ExpertOverride
	FailureLogs     map[string]int // Tracks failures per expert ID (e.g., "E1")
	mu              sync.Mutex
}

// ExpertOverride holds per-expert variable overrides set by the supervisor.
type ExpertOverride struct {
	DropoutRate  float32
	LRMultiplier float32 // multiplier on global LR for this expert's params
	OutputScale  float32
}

// NewSupervisor initializes a new training supervisor.
func NewSupervisor() *Supervisor {
	// Initialize with a small temporal attention head for sequence judging
	att, _ := nn.NewMultiHeadAttention(64, 1, 1)
	judge, _ := nn.NewLinear(64, 1)

	return &Supervisor{
		BestPerplexity:  1e9,
		TemporalAttention: att,
		GrammarJudge:      judge,
		expertOverrides:   make(map[int]map[int]*ExpertOverride),
	}
}

// Reflect nudges variables (LR, Noise, Temperature) based on training stats.
func (s *Supervisor) Reflect(stats TrainingStats, opt *nn.Adam, model *IntentMoE) {
	// 0. Jump Start Recovery (The "Heat" nudge)
	if s.JustPerformedSurgery {
		log.Println("🔥 Surgery detected: Increasing Heat (LR) to bake in new weights.")
		opt.SetLearningRate(0.0005)
		RouterNoiseFactor += 0.05
		s.JustPerformedSurgery = false
	}

	// 1. Dominance Check (The "Monopoly" Nudge)
	// If one expert is handling most of the traffic, we increase noise to force exploration.
	if stats.MaxDominance > 0.85 {
		log.Printf("🤖 Supervisor Reflect: Expert Dominance too high (%.2f%%). Nudging Router Noise and Temperature...\n", stats.MaxDominance*100)
		for _, layer := range ActiveLayers {
			layer.RouterTemperature += 0.15
		}
		RouterNoiseFactor += 0.05
	} else if stats.MaxDominance < 0.25 {
		// If dominance is too low (uniform distribution), we may be too noisy to specialize.
		for _, layer := range ActiveLayers {
			if layer.RouterTemperature > 0.8 {
				layer.RouterTemperature -= 0.05
			}
		}
	}

	// 2. Plateau Detection (The "Learning Rate" Nudge)
	// If perplexity isn't improving, we lower the LR to settle into a better minimum.
	if stats.Perplexity < s.BestPerplexity && stats.Perplexity > 0 {
		s.BestPerplexity = stats.Perplexity
		s.PlateauCount = 0
	} else {
		s.PlateauCount++
	}

	if s.PlateauCount > 500 {
		newLR := opt.GetLearningRate() * 0.75
		log.Printf("📉 Supervisor Reflect: Training plateaued for 500 steps. Reducing LR to %e\n", newLR)
		opt.SetLearningRate(newLR)
		s.PlateauCount = 0
	}

	// 3. Confidence Check (The "Entropy" nudge)
	// Low confidence often precedes "word salad" output.
	if stats.StepConfidence < 0.18 && stats.Epoch > 5 {
		log.Printf("⚠️ Supervisor Reflect: Step Confidence low (%.2f%%). Increasing Router Temperature...\n", stats.StepConfidence*100)
		for _, layer := range ActiveLayers {
			layer.RouterTemperature += 0.1
		}
	}
}

// Validate checks if the model is actually learning or just "mumbling."
// It runs inference on anchor queries and applies quality gates.
func (s *Supervisor) Validate(model *IntentMoE) bool {
	if model.SentenceVocab == nil {
		return true // Cannot validate without vocabulary context
	}

	// Anchor queries to check for structural coherence
	testQueries := []string{"hello", "who are you", "tell me a joke"}
	mumbleDetected := false

	for _, q := range testQueries {
		// Run a quick inference pass (greedy)
		resp, _ := model.GenerateGuidedSentence(q, 15)
		if s.isMumbling(resp) {
			mumbleDetected = true
			log.Printf("🗣️ Supervisor Validate: Mumbling detected for query '%s': '%s'\n", q, resp)
			break
		}
	}

	if mumbleDetected {
		s.MumbleCount++
	} else {
		if s.MumbleCount > 0 {
			s.MumbleCount--
		}
	}

	return !mumbleDetected
}

// expertScore computes a combined health score (higher = healthier/stronger).
// Uses L2 norm of weights * utilization fraction.
func expertScore(expert Expert, utilizationFrac float32) float32 {
	params := expert.Parameters()
	var l2 float32
	for _, p := range params {
		for _, v := range p.Data {
			l2 += v * v
		}
	}
	return l2 * (utilizationFrac + 0.01) // small epsilon to avoid all-zero
}

// PerformSurgery identifies and repairs collapsed experts by cloning better ones.
// 🆕 STAGGERED TRIAGE: Only refreshes the worst-performing ≤50% of experts per layer,
// leaving at least half intact to preserve learned sequences.
func (s *Supervisor) PerformSurgery(model *IntentMoE) {
	log.Println("🏥 Supervisor: Performing Expert Surgery (Staggered Triage)...")

	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	for i, layer := range layers {
		numExperts := len(layer.Experts)
		if numExperts == 0 {
			continue
		}

		// Compute utilization fractions
		totalUtil := 0
		for _, u := range layer.AccumulatedUtilization {
			totalUtil += u
		}

		type expertHealth struct {
			idx   int
			score float32
		}
		scores := make([]expertHealth, numExperts)
		for eIdx, expert := range layer.Experts {
			utilFrac := float32(0)
			if totalUtil > 0 && eIdx < len(layer.AccumulatedUtilization) {
				utilFrac = float32(layer.AccumulatedUtilization[eIdx]) / float32(totalUtil)
			}
			scores[eIdx] = expertHealth{
				idx:   eIdx,
				score: expertScore(expert, utilFrac),
			}
		}

		// Sort ascending (weakest first)
		sort.Slice(scores, func(a, b int) bool {
			return scores[a].score < scores[b].score
		})

		// Only heal the bottom half
		healCount := (numExperts + 1) / 2 // ceil(N/2)

		// Find the alpha (strongest) expert
		alphaIdx := scores[numExperts-1].idx

		healedAny := false
		for k := 0; k < healCount; k++ {
			candidateIdx := scores[k].idx
			if candidateIdx == alphaIdx {
				continue // Don't overwrite the alpha with itself
			}
			// Only heal if genuinely weak (dead or near-dead by L2)
			if scores[k].score < 1e-3 || (totalUtil > 100 && scores[k].score < scores[numExperts-1].score*0.01) {
				log.Printf("🧬 Triage (Layer %d): Expert E%d (score=%.4f) refreshed from E%d (score=%.4f)",
					i, candidateIdx, scores[k].score, alphaIdx, scores[numExperts-1].score)
				layer.PerformSurgery(alphaIdx, candidateIdx)
				s.JustPerformedSurgery = true
				healedAny = true
			}
		}
		if !healedAny {
			log.Printf("✅ Layer %d: No dead experts detected, skipping surgery.", i)
		}
	}
}

// ReflectSparse is a specialized version of Reflect for the high-performance SparseModel.
func (s *Supervisor) ReflectSparse(stats TrainingStats, gater *SparseGater, lr *float32) {
	if stats.MaxDominance > 0.85 {
		log.Printf("🤖 Supervisor Reflect: Expert Dominance too high (%.2f%%). Nudging Router Weights...\n", stats.MaxDominance*100)
		// Nudge the gater weights slightly to break the monopoly
		for i := range gater.Weights {
			gater.Weights[i] += (rand.Float32() - 0.5) * 0.01
		}
	}

	if stats.CurrentLoss < s.BestPerplexity {
		s.BestPerplexity = stats.CurrentLoss
		s.PlateauCount = 0
	} else {
		s.PlateauCount++
	}

	if s.PlateauCount > 1000 {
		*lr *= 0.9
		log.Printf("📉 Supervisor Reflect: Sparse training plateaued. Reducing LR to %e\n", *lr)
		s.PlateauCount = 0
	}
}

// PerformSurgerySparse handles expert repair for SparseModel architectures.
func (s *Supervisor) PerformSurgerySparse(model *SparseModel) {
	log.Println("🏥 Supervisor: Performing Sparse Expert Surgery...")

	alphaID := -1
	sinkID := -1
	maxL2 := float32(-1.0)
	minL2 := float32(1e9)

	for i, expert := range model.Experts {
		var l2 float32
		for _, w := range expert.Weights {
			l2 += w * w
		}
		if l2 < 1e-4 {
			sinkID = i
		}
		if l2 > maxL2 {
			maxL2 = l2
			alphaID = i
		}
		if l2 < minL2 && l2 > 1e-4 {
			minL2 = l2
		}
	}

	if sinkID != -1 && alphaID != -1 && alphaID != sinkID {
		log.Printf("🧬 Surgery: Cloning Sparse Expert E%d (Alpha) -> E%d (Sink)\n", alphaID, sinkID)
		copy(model.Experts[sinkID].Weights, model.Experts[alphaID].Weights)
		copy(model.Experts[sinkID].Bias, model.Experts[alphaID].Bias)

		s.JustPerformedSurgery = true

		// Add tiny mutation
		for j := range model.Experts[sinkID].Weights {
			model.Experts[sinkID].Weights[j] += (rand.Float32() - 0.5) * 0.001
		}
	}
}

// isMumbling implements structural checks for "word salad" detection.
func (s *Supervisor) isMumbling(response string) bool {
	if response == "" {
		return true
	}
	tokens := strings.Fields(response)
	if len(tokens) < 3 {
		return false // Too short to judge coherence accurately
	}

	// 1. Diversity Check (Repetition detection)
	unique := make(map[string]bool)
	for _, t := range tokens {
		unique[t] = true
	}
	uniqueRatio := float32(len(unique)) / float32(len(tokens))
	if uniqueRatio < 0.35 {
		return true // Too much repetition (e.g. "is is is is")
	}

	// 2. Average Word Length (Word Salad Detection)
	totalLen := 0
	for _, t := range tokens {
		totalLen += len(t)
	}
	avgLen := float32(totalLen) / float32(len(tokens))
	if avgLen < 2.1 {
		return true // Mostly 1-2 char tokens (e.g. "a b c to .")
	}

	// 3. Leading Garbage
	first := tokens[0]
	if len(first) == 1 && unicode.IsPunct(rune(first[0])) {
		return true // Starts with punctuation soup
	}

	return false
}

// SuperviseSentenceCreation performs multi-pass generation to ensure a complete sentence.
func (s *Supervisor) SuperviseSentenceCreation(model *IntentMoE, query string) (string, []string) {
	maxPasses := 3
	bestResponse := ""
	bestTokens := []string{}

	currentTemp := float32(0.8)
	currentBias := float32(8.0)

	for pass := 0; pass < maxPasses; pass++ {
		log.Printf("🤖 Supervisor: Pass %d for query '%s' (Temp: %.2f, Bias: %.2f)", pass+1, query, currentTemp, currentBias)

		// Temporarily adjust model parameters
		oldTemp := model.GetGateTemperature()
		model.SetGateTemperature(currentTemp)

		// Run generation
		response, tokens := model.GenerateGuidedSentence(query, 20)

		// Restore old parameters
		model.SetGateTemperature(oldTemp)

		if s.IsCompleteSentence(tokens) {
			log.Printf("✅ Supervisor: Sentence validated on pass %d: '%s'", pass+1, response)
			return response, tokens
		}

		log.Printf("❌ Supervisor: Pass %d produced incomplete sentence: '%s'", pass+1, response)

		// Save best attempt so far
		if len(tokens) > len(bestTokens) {
			bestResponse = response
			bestTokens = tokens
		}

		// Nudge parameters for next pass
		currentTemp += 0.15
		currentBias += 2.0
	}

	log.Printf("⚠️ Supervisor: Failed to create perfect sentence after %d passes. Returning best attempt.", maxPasses)
	return bestResponse, bestTokens
}

// IsCompleteSentence uses heuristics and temporal attention to judge sentence quality.
func (s *Supervisor) IsCompleteSentence(tokens []string) bool {
	if len(tokens) < 3 {
		return false
	}

	// 1. Basic Heuristics
	hasVerb := false
	hasSubject := false
	for _, t := range tokens {
		role := MapWordToGrammarType(t)
		if role == "VERB" || role == "AUX" {
			hasVerb = true
		}
		if role == "PRON" || role == "NOUN" {
			hasSubject = true
		}
	}

	if !hasVerb || !hasSubject {
		return false
	}

	// 2. Ending check
	last := tokens[len(tokens)-1]
	lastChar := last[len(last)-1]
	if lastChar != '.' && lastChar != '!' && lastChar != '?' {
		// If it doesn't end with punctuation, it might be truncated
		return false
	}

	// 3. Temporal Attention Check (Conceptual)
	// In a real implementation, we would pass the hidden states to s.TemporalAttention.
	// For now, we use the heuristics as a proxy for the attention's "judgment".

	return true
}

// SanitizeTensors acts as a circuit breaker for hardware-level or numerical failures.
func (s *Supervisor) SanitizeTensors(output []float32) bool {
	for _, val := range output {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			log.Println("🛑 CRITICAL: MatMul produced NaN! Emergency Brake Engaged.")
			return false
		}
	}

	// Check for "Dead Output" (The MatMul bug)
	sum := float64(0.0)
	for _, v := range output {
		sum += math.Abs(float64(v))
	}
	if sum == 0 && len(output) > 0 {
		log.Println("⚠️ WARNING: Zero-sum output detected. MatMul bridge is failing.")
		return false
	}
	return true
}

// ── NEW: Expert Factory ───────────────────────────────────────────────────────

// TrainPair is a single training example the supervisor can inject or modify.
type TrainPair struct {
	Q       string
	A       string
	Intent  string
	Grammar string
	Weight  float32
}

// AddExpertToLayer dynamically creates a new GrammarExpert and appends it to the
// specified MoE layer. It also extends the gating network's weight matrix to
// include the new expert's routing column.
func (s *Supervisor) AddExpertToLayer(model *IntentMoE, layerIdx int, roleID int) error {
	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	if layerIdx < 0 || layerIdx >= len(layers) {
		return fmt.Errorf("AddExpertToLayer: layerIdx %d out of range (have %d layers)", layerIdx, len(layers))
	}
	layer := layers[layerIdx]

	inputDim := model.EmbeddingDim
	newID := len(layer.Experts)
	expert, err := NewGrammarExpert(newID, roleID%len(GrammarRoles), inputDim, inputDim)
	if err != nil {
		return fmt.Errorf("AddExpertToLayer: could not create expert: %w", err)
	}

	// Seed with structural bias if vocab is available
	if model.SentenceVocab != nil {
		expert.SeedGrammarBias(model.SentenceVocab.Size(), model.SentenceVocab.TokenToWord)
	}

	// Append expert
	layer.Experts = append(layer.Experts, expert)
	layer.NumExperts++

	// Extend ExpertOutputScale
	layer.ExpertOutputScale = append(layer.ExpertOutputScale, 1.0)

	// Extend AccumulatedUtilization
	if len(layer.AccumulatedUtilization) < layer.NumExperts {
		layer.AccumulatedUtilization = append(layer.AccumulatedUtilization, 0)
	}

	// Extend gating network: add a new column to Weights [inputDim x (N+1)]
	if layer.GatingNetwork != nil {
		// 1. Resize Main Linear
		if layer.GatingNetwork.Linear != nil {
			gw := layer.GatingNetwork.Linear.Weights
			oldNumExperts := gw.Shape[1]
			newNumExperts := oldNumExperts + 1
			oldData := gw.Data
			newData := make([]float32, gw.Shape[0]*newNumExperts)
			for row := 0; row < gw.Shape[0]; row++ {
				copy(newData[row*newNumExperts:row*newNumExperts+oldNumExperts],
					oldData[row*oldNumExperts:row*oldNumExperts+oldNumExperts])
				newData[row*newNumExperts+oldNumExperts] = (rand.Float32()*2 - 1) * 0.01
			}
			layer.GatingNetwork.Linear.Weights = tensor.NewTensor(
				[]int{gw.Shape[0], newNumExperts}, newData, true)

			// Extend biases
			if layer.GatingNetwork.Linear.Biases != nil {
				oldBias := layer.GatingNetwork.Linear.Biases.Data
				newBias := make([]float32, newNumExperts)
				copy(newBias, oldBias)
				newBias[oldNumExperts] = 0.0
				layer.GatingNetwork.Linear.Biases = tensor.NewTensor(
					[]int{newNumExperts}, newBias, true)
			}
		}

		// 2. Resize Noise Linear (CRITICAL for SIMD stability)
		if layer.GatingNetwork.NoiseLinear != nil {
			nw := layer.GatingNetwork.NoiseLinear.Weights
			oldNumExperts := nw.Shape[1]
			newNumExperts := oldNumExperts + 1
			oldData := nw.Data
			newData := make([]float32, nw.Shape[0]*newNumExperts)
			for row := 0; row < nw.Shape[0]; row++ {
				copy(newData[row*newNumExperts:row*newNumExperts+oldNumExperts],
					oldData[row*oldNumExperts:row*oldNumExperts+oldNumExperts])
				newData[row*newNumExperts+oldNumExperts] = (rand.Float32()*2 - 1) * 0.01
			}
			layer.GatingNetwork.NoiseLinear.Weights = tensor.NewTensor(
				[]int{nw.Shape[0], newNumExperts}, newData, true)

			// Extend noise biases
			if layer.GatingNetwork.NoiseLinear.Biases != nil {
				oldBias := layer.GatingNetwork.NoiseLinear.Biases.Data
				newBias := make([]float32, newNumExperts)
				copy(newBias, oldBias)
				newBias[oldNumExperts] = 0.0
				layer.GatingNetwork.NoiseLinear.Biases = tensor.NewTensor(
					[]int{newNumExperts}, newBias, true)
			}
		}

		// 3. Repair/Resize LayerNorm
		layer.GatingNetwork.RepairArchitecture()
	}

	// 4. Extend auxiliary slices to prevent out-of-bounds in Forward/Backward
	layer.ExpertFrozen = append(layer.ExpertFrozen, false)
	layer.StagnationCounters = append(layer.StagnationCounters, 0)
	layer.ExpertGradMultiplier = append(layer.ExpertGradMultiplier, 1.0)
	// ExpertOutputScale is already appended above

	log.Printf("✨ [Supervisor] Added new GrammarExpert E%d (role=%s) to Layer %d. Total: %d experts.",
		newID, GrammarRoles[roleID%len(GrammarRoles)], layerIdx, layer.NumExperts)
	return nil
}

// ModifyTrainingData allows the supervisor to inject synthetic pairs or hot-swap
// existing ones. Pass append=true to add to existing data, false to replace.
// This operates on a pointer to the slice so callers see the change immediately.
func (s *Supervisor) ModifyTrainingData(pairs *[]TrainPair, newPairs []TrainPair, appendMode bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if appendMode {
		*pairs = append(*pairs, newPairs...)
		log.Printf("📚 [Supervisor] Injected %d synthetic training pairs (total: %d)", len(newPairs), len(*pairs))
	} else {
		*pairs = newPairs
		log.Printf("📚 [Supervisor] Replaced training data with %d pairs", len(newPairs))
	}
}

// SetExpertVariables tunes per-expert output scale and marks overrides in the
// supervisor's registry so they can be applied each epoch.
func (s *Supervisor) SetExpertVariables(model *IntentMoE, layerIdx, expertIdx int, outputScale float32, lrMultiplier float32) {
	s.mu.Lock()
	defer s.mu.Unlock()

	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	if layerIdx < 0 || layerIdx >= len(layers) {
		log.Printf("⚠️ SetExpertVariables: layerIdx %d out of range", layerIdx)
		return
	}
	layer := layers[layerIdx]

	if expertIdx < 0 || expertIdx >= len(layer.Experts) {
		log.Printf("⚠️ SetExpertVariables: expertIdx %d out of range (layer has %d)", expertIdx, len(layer.Experts))
		return
	}

	// Apply output scale immediately
	if expertIdx < len(layer.ExpertOutputScale) {
		layer.ExpertOutputScale[expertIdx] = outputScale
	}

	// Store override for LR multiplier (applied during optimizer step)
	if _, ok := s.expertOverrides[layerIdx]; !ok {
		s.expertOverrides[layerIdx] = make(map[int]*ExpertOverride)
	}
	s.expertOverrides[layerIdx][expertIdx] = &ExpertOverride{
		OutputScale:  outputScale,
		LRMultiplier: lrMultiplier,
	}

	log.Printf("🎛️ [Supervisor] Layer %d Expert E%d: OutputScale=%.2f LRMult=%.2f",
		layerIdx, expertIdx, outputScale, lrMultiplier)
}

// HandleQualityGateFailure acts as the core decision-maker when the Subject-Verb Connection quality gate drops below threshold.
func (s *Supervisor) HandleQualityGateFailure(model *IntentMoE, path string, pair *TrainPair, actualScore float64) {
	log.Printf("🎯 [Supervisor] Adapting system for failing path [%s] on question: '%s'", path, pair.Q)

	// involvedExperts might look like "E1+E8 -> E4+E12"
	parts := strings.FieldsFunc(path, func(r rune) bool {
		return r == '+' || r == '-' || r == '>' || r == ' '
	})

	s.mu.Lock()
	if s.FailureLogs == nil {
		s.FailureLogs = make(map[string]int)
	}
	s.mu.Unlock()

	// 1. MUTATE VARIABLES: Penalize the failed expert combination
	for _, idStr := range parts {
		if !strings.HasPrefix(idStr, "E") {
			continue
		}
		var eid int
		_, err := fmt.Sscanf(idStr, "E%d", &eid)
		if err != nil {
			continue
		}

		// Apply to all layers to ensure we catch the bottleneck expert
		layers := model.Encoder.GetMoELayers()
		if model.Decoder.OutputMoE != nil {
			layers = append(layers, model.Decoder.OutputMoE)
		}

		for lIdx, layer := range layers {
			if eid < len(layer.Experts) {
				// Get current overrides or defaults
				currentScale := float32(1.0)
				currentLR := float32(1.0)

				s.mu.Lock()
				if ov, ok := s.expertOverrides[lIdx][eid]; ok {
					currentScale = ov.OutputScale
					currentLR = ov.LRMultiplier
				}
				s.mu.Unlock()

				// Mutate: weight *= 0.90, LR *= 1.05
				newScale := currentScale * 0.90
				newLR := currentLR * 1.05
				s.SetExpertVariables(model, lIdx, eid, newScale, newLR)

				s.mu.Lock()
				s.FailureLogs[idStr]++
				s.mu.Unlock()
			}
		}
	}

	// 2. CREATE NEW EXPERTS: If an expert fails too often on an intent, isolate it
	for _, idStr := range parts {
		s.mu.Lock()
		failCount := s.FailureLogs[idStr]
		s.mu.Unlock()

		if failCount >= 3 {
			log.Printf("🔥 [Supervisor] Path Component %s failed %d times. Spawning Specialized Expert.", idStr, failCount)

			// Determine role from intent
			roleID := 7 // OTHER
			if pair.Intent == "social" {
				roleID = 6 // GREET
			}

			// Spawn to Layer 0 and Decoder Output MoE
			s.AddExpertToLayer(model, 0, roleID)
			layers := model.Encoder.GetMoELayers()
			if model.Decoder.OutputMoE != nil {
				s.AddExpertToLayer(model, len(layers), roleID)
			}

			s.mu.Lock()
			s.FailureLogs[idStr] = 0
			s.mu.Unlock()
		}
	}
}

// EvolveTrainingData mutates training pairs to match expected syntactic structures.
func (s *Supervisor) EvolveTrainingData(pairs *[]TrainPair, failedPair *TrainPair) {
	for i := range *pairs {
		pair := &(*pairs)[i]
		if pair.Q == failedPair.Q && pair.Intent == failedPair.Intent {
			// If it's a short social utterance failing grammar validation, rewrite or adapt
			if pair.Intent == "social" && len(strings.Fields(pair.Q)) <= 2 {
				log.Printf("📝 [Supervisor Data Update] Augmenting grammar structure for shorthand data: '%s'", pair.Q)

				switch strings.ToLower(pair.Q) {
				case "hello", "hi", "hey":
					pair.Q = "i greet you with " + pair.Q
				case "thanks", "thank you":
					pair.Q = "i offer you my thanks"
				default:
					// Lower the sample weight so it stops poisoning the general gradient
					if pair.Weight == 0 {
						pair.Weight = 1.0
					}
					pair.Weight *= 0.5
				}
			}
		}
	}
}

// RunTriage orchestrates all supervisor interventions based on current metrics.
// Call once per epoch after test evaluation.
// similarityScore: current average similarity [0,1]
// pairs: pointer to training pairs slice for hot-injection
func (s *Supervisor) RunTriage(model *IntentMoE, similarityScore float32, pairs *[]TrainPair) {
	log.Printf("🔬 [Supervisor Triage] Similarity=%.1f%%", similarityScore*100)

	// If similarity is very low: inject synthetic social phrase pairs to force the model
	// to encounter more diverse social patterns.
	if similarityScore < 0.25 && pairs != nil {
		syntheticPairs := []TrainPair{
			{Q: "hi", A: "Hi there! How can I help you today?", Intent: "greeting"},
			{Q: "hello", A: "Hello! Nice to meet you.", Intent: "greeting"},
			{Q: "how are you", A: "I am doing well, thank you for asking!", Intent: "status_check"},
			{Q: "what is your name", A: "My name is Gollemer. I am an AI assistant.", Intent: "identity"},
			{Q: "good morning", A: "Good morning! Hope you have a great day.", Intent: "greeting"},
			{Q: "thanks", A: "You are welcome! Let me know if you need anything else.", Intent: "polite"},
		}
		s.ModifyTrainingData(pairs, syntheticPairs, true)

		// Also add a fresh PRON expert to help with "I am", "you are" patterns
		if err := s.AddExpertToLayer(model, 0, 0 /* PRON role */); err != nil {
			log.Printf("⚠️ Triage: Could not add PRON expert: %v", err)
		}
	}

	// Surgery pass with staggered triage
	s.PerformSurgery(model)
}
