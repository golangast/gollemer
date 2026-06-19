package moe

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
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
	expertOverrides      map[int]map[int]*ExpertOverride
	FailureLogs          map[string]int // Tracks failures per expert ID (e.g., "E1")
	TrainingDataPath     string         // Path to the raw training assets for evolution
	SpawnsThisEpoch      int            // Track and pace expert spawning per epoch
	OverfitMode          bool           // Allow relaxed constraints during overfit mode
	DisableDataEvolution bool           // When true, skip corpus mutation entirely
	mu                   sync.Mutex
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
		BestPerplexity:    1e9,
		TemporalAttention: att,
		GrammarJudge:      judge,
		FailureLogs:       make(map[string]int),
		expertOverrides:   make(map[int]map[int]*ExpertOverride),
	}
}

// Reflect nudges variables (LR, Noise, Temperature) based on training stats.
func (s *Supervisor) Reflect(stats TrainingStats, opt *nn.Adam, model *IntentMoE) {
	// 0. Jump Start Recovery (The "Heat" nudge)
	// Use a relative bump (×2.0) instead of a hard-coded 0.0005 so the heat
	// signal is proportional to the current training regime. The old absolute
	// value was ~50× the new peak LR (1e-5) and would destroy weight geometry
	// right after surgery tried to fix it. Cap at 2e-5 to stay conservative.
	if s.JustPerformedSurgery {
		currentLR := opt.GetLearningRate()
		heatedLR := currentLR * 2.0
		const maxPostSurgeryLR = float32(2e-5)
		if heatedLR > maxPostSurgeryLR {
			heatedLR = maxPostSurgeryLR
		}
		log.Printf("🔥 Surgery detected: Bumping LR %.2e → %.2e to bake in new weights.", currentLR, heatedLR)
		opt.SetLearningRate(heatedLR)
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

	if s.PlateauCount > 5 {
		newLR := opt.GetLearningRate() * 0.75
		log.Printf("📉 Supervisor Reflect: Training plateaued for 5 steps. Reducing LR to %e\n", newLR)
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
// 🆕 STAGGERED TRIAGE: Only refreshes the worst-performing ≤5% of experts per layer,
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

	dummyRule := IntentRule{}

	for i, t := range tokens {
		role := MapWordToGrammarType(t)
		if role == "VERB" || role == "AUX" {
			hasVerb = true
		}
		if role == "PRON" || role == "NOUN" {
			hasSubject = true
		}

		prevType := "BOS"
		if i > 0 {
			prevType = MapWordToGrammarType(tokens[i-1])
		}
		nextType := "EOS"
		if i < len(tokens)-1 {
			nextType = MapWordToGrammarType(tokens[i+1])
		}

		if dummyRule.EvaluateWindow(prevType, role, nextType) > 0 {
			return false // Fails trigram rule
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

	newRole := GrammarRoles[roleID%len(GrammarRoles)]

	// Cap the number of experts to prevent memory explosion (OOM).
	// Policy: (1) Try standard LRU/health-based eviction first.
	//         (2) If blocked (all candidates are pinned), force-evict the absolute
	//             lowest-utility dynamic expert (index >= 8) to guarantee a slot.
	if layer.AtCapacity() {
		evictedIdx, evicted := layer.EvictLeastActive(newRole)
		if evicted {
			// Floor guard: if the evicted expert was a standard/structural expert and
			// removing it would breach the 40% floor, veto the eviction and fall through
			// to the force-evict path which only targets dynamic experts (index >= 48).
			if !s.CanEvict(layer, evictedIdx) {
				// CanEvict already logged the block reason. Fall through to force-evict.
				evicted = false
			} else {
				log.Printf("♻️ [Supervisor] Evicted low-performing Expert E%d (standard) from layer %d to make room.", evictedIdx, layerIdx)
			}
		}
		if !evicted {
			// Fallback: force-evict lowest utility dynamic expert, bypassing pinned flags.
			evictedIdx = layer.ForceEvictLowestUtility(newRole)
			if evictedIdx == -1 {
				return fmt.Errorf("AddExpertToLayer: layer %d reached capacity and no expert could be force-evicted (fewer than 9 experts)", layerIdx)
			}
			log.Printf("⚠️ [Supervisor] Force-evicted Expert E%d from layer %d (all standard candidates were pinned).", evictedIdx, layerIdx)
		}

		// Scale back exploration metrics to stabilize after eviction.
		if RouterNoiseFactor > 0 {
			RouterNoiseFactor -= 0.05
		}
		if layer.RouterTemperature > 0.8 {
			layer.RouterTemperature -= 0.05
		}

		// Sanity check: eviction should have freed a slot.
		if layer.AtCapacity() {
			return fmt.Errorf("AddExpertToLayer: layer %d still at maximum capacity (64 experts) after eviction", layerIdx)
		}
	}

	inputDim := layer.InputDim
	if inputDim <= 0 {
		inputDim = model.EmbeddingDim // Fallback
	}
	outputDim := layer.OutputDim

	s.mu.Lock()
	newID := len(layer.Experts)
	for _, e := range layer.Experts {
		if e.GetID() >= newID {
			newID = e.GetID() + 1
		}
	}
	s.mu.Unlock()

	expert, err := NewGrammarExpert(newID, roleID%len(GrammarRoles), inputDim, outputDim)
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

	// Extend dynamic parallel tracking slices
	layer.ExpertHealth = append(layer.ExpertHealth, 1.0)
	layer.ExpertLastUsedAt = append(layer.ExpertLastUsedAt, time.Now())
	layer.ExpertPinned = append(layer.ExpertPinned, false) // Dynamically spawned experts should NOT be pinned (prevent choking out baseline)
	layer.ExpertRole = append(layer.ExpertRole, GrammarRoles[roleID%len(GrammarRoles)])

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

	// Step 2: Clamped bounds — prevents weight-frying from extreme supervisor adjustments.
	// MaxLRMult capped at 1.15 (down from 1.50): suppresses runaway gradient escalation
	// on failing paths without starving valid experts of learning signal.
	// MinOutputScale raised to 0.85 (up from 0.70): prevents over-suppressing weights
	// on cross-layer bottleneck experts (e.g. E5) that are legitimately load-bearing.
	if lrMultiplier > 1.15 {
		lrMultiplier = 1.15
	}
	if lrMultiplier < 0.1 {
		lrMultiplier = 0.1
	}
	if outputScale < 0.85 {
		outputScale = 0.85
	}
	if outputScale > 1.0 {
		outputScale = 1.0
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

	// Atomic pacing safety: if the epoch spawn cap is already saturated, skip ALL
	// topology mutations for this path. Mutating hyperparameters for a topology that
	// hasn't been built yet creates an unstable feedback loop where weights drift
	// without any new structural anchors to absorb the gradient shift.
	limit := 32
	if s.OverfitMode {
		limit = 5
	}
	s.mu.Lock()
	currentSpawns := s.SpawnsThisEpoch
	s.mu.Unlock()
	if currentSpawns >= limit {
		log.Printf("⏳ [Supervisor] Pacing limit reached (%d/%d). Deferring spawning AND topology mutations for path [%s].", currentSpawns, limit, path)
		return
	}

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

		expertFound := false
		for lIdx, layer := range layers {
			if eid < len(layer.Experts) {
				expertFound = true
				// Get current overrides or defaults
				currentScale := float32(1.0)
				currentLR := float32(1.0)

				s.mu.Lock()
				if ov, ok := s.expertOverrides[lIdx][eid]; ok {
					currentScale = ov.OutputScale
					currentLR = ov.LRMultiplier
				}
				s.mu.Unlock()

				// Incremental nudge: −0.05 to scale, +0.05 to LR per failure event.
				// Replaces the old multiplicative (×0.85 / ×1.05) jumps which caused
				// violent weight oscillations in bottleneck experts like E5 and
				// destroyed adjacent paths that were actually converging correctly.
				newScale := currentScale - 0.05
				if newScale < 0.85 {
					newScale = 0.85
				}
				newLR := currentLR + 0.05
				s.SetExpertVariables(model, lIdx, eid, newScale, newLR)
			}
		}

		if expertFound {
			s.mu.Lock()
			s.FailureLogs[idStr]++
			s.mu.Unlock()
		}
	}

	// 2. CREATE NEW EXPERTS: If an expert fails too often on an intent, isolate it
	for _, idStr := range parts {
		s.mu.Lock()
		failCount := s.FailureLogs[idStr]
		spawns := s.SpawnsThisEpoch
		s.mu.Unlock()

		if failCount >= 3 {
			limit := 32
			if s.OverfitMode {
				limit = 5
			}
			if spawns >= limit {
				log.Printf("⏳ [Supervisor] Component %s reached failure threshold (%d), but pacing limits expert spawning to %d per epoch. Deferring...", idStr, failCount, limit)
				continue
			}

			log.Printf("🔥 [Supervisor] Path Component %s failed %d times. Spawning Specialized Expert.", idStr, failCount)

			// Determine role from token context, not just intent label.
			// Technical multi-word targets must NOT default to GREET — spawning
			// a GREET expert for a complex query floods the router with useless
			// specialists and destroys the semantic precision of existing paths.
			roleID := 7 // OTHER — safe default for complex/unknown targets
			if pair.Intent == "social" && !isTechnicalPayload(pair.Q) {
				roleID = 6 // GREET — only for genuine short social phrases
			}

			// Spawn to Layer 0 and Decoder Output MoE
			if err := s.AddExpertToLayer(model, 0, roleID); err != nil {
				log.Printf("⚠️ Supervisor: %v", err)
			}
			layers := model.Encoder.GetMoELayers()
			if model.Decoder.OutputMoE != nil {
				if err := s.AddExpertToLayer(model, len(layers), roleID); err != nil {
					log.Printf("⚠️ Supervisor: %v", err)
				}
			}

			s.mu.Lock()
			s.FailureLogs[idStr] = 0
			s.SpawnsThisEpoch++
			s.mu.Unlock()
		}
	}

	// 3. EVOLVE DATASET: Permanently rewrite training assets for this failure
	if !s.DisableDataEvolution {
		s.EvolveDataset(pair.Q)
	}
}

// EvolveDataset mutates the underlying training asset files to replace weak
// linguistic structures with standard syntactic ones.
func (s *Supervisor) EvolveDataset(targetQuestion string) {
	s.mu.Lock()
	path := s.TrainingDataPath
	s.mu.Unlock()

	if path == "" {
		return
	}

	log.Printf("📝 [Supervisor] Scanning training assets for data evolution: '%s'", targetQuestion)

	file, err := os.Open(path)
	if err != nil {
		log.Printf("⚠️ [Supervisor] Could not open data asset for evolution: %v", err)
		return
	}

	var lines []string
	scanner := bufio.NewScanner(file)
	mutatedCount := 0

	for scanner.Scan() {
		line := scanner.Text()

		// Target checking: both Gollemer internal markers and raw CSV lines
		match := false
		if strings.Contains(line, "__ques__ "+targetQuestion+" __ans__") {
			match = true
		} else if strings.HasPrefix(line, targetQuestion+",") {
			match = true
		}

		if match {
			// Mutate short token fragments into rich syntactic representations
			var replacement string
			switch strings.ToLower(targetQuestion) {
			case "hello", "hi", "hey":
				replacement = "i greet you with " + targetQuestion
			case "thanks", "thank you":
				replacement = "i offer you my thanks"
			case "i am sad":
				replacement = "i feel very sad today"
			default:
				if strings.HasPrefix(strings.ToLower(targetQuestion), "") {
					replacement = targetQuestion // Already wrapped — don't double-wrap
				} else if isTechnicalPayload(targetQuestion) {
					// Preserve technical multi-word queries verbatim.
					replacement = targetQuestion
				} else {
					replacement = "" + targetQuestion
				}
			}

			// Atomic replacement
			var newLine string
			if strings.Contains(line, "__ques__") {
				oldMarker := "__ques__ " + targetQuestion
				newMarker := "__ques__ " + replacement
				newLine = strings.Replace(line, oldMarker, newMarker, 1)
			} else {
				// CSV Case: Replace leading question token
				newLine = strings.Replace(line, targetQuestion+",", replacement+",", 1)
			}
			lines = append(lines, newLine)
			mutatedCount++
		} else {
			lines = append(lines, line)
		}
	}
	file.Close()

	if mutatedCount > 0 {
		// Flush changes back cleanly to prevent broken buffers
		outFile, err := os.Create(path)
		if err != nil {
			log.Printf("⚠️ [Supervisor] Could not write evolved dataset: %v", err)
			return
		}
		defer outFile.Close()

		writer := bufio.NewWriter(outFile)
		for _, line := range lines {
			_, _ = writer.WriteString(line + "\n")
		}
		_ = writer.Flush()
		log.Printf("✅ [Supervisor] Data Evolution Success. Mutated %d corpus references in %s", mutatedCount, path)
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
// expert: the HyperparameterExpert that owns the rolling history (may be nil to skip the gate)
func (s *Supervisor) RunTriage(model *IntentMoE, similarityScore float32, pairs *[]TrainPair) {
	log.Printf("🔬 [Supervisor Triage] Similarity=%.1f%%", similarityScore*100)

	// Always run a light surgery pass (clones dead experts from healthy ones).
	// This is a structural repair, not a policy change, so it is not gated.
	s.PerformSurgery(model)
}

// RunTriageGated is the decay-aware version of RunTriage. It adds synthetic pairs
// and structural interventions ONLY when the rolling 100-epoch window shows that
// metrics are genuinely declining. Pass the HyperparameterExpert so the gate
// can read the same history that AnalyzeAndAdjust uses.
func (s *Supervisor) RunTriageGated(model *IntentMoE, similarityScore float32, pairs *[]TrainPair, expert interface{ MetricsAreDeclining() bool }) {
	log.Printf("🔬 [Supervisor Triage] Similarity=%.1f%%", similarityScore*100)

	// Warmup + direction guard: only intervene when things are getting worse.
	if expert == nil || !expert.MetricsAreDeclining() {
		log.Printf("✅ [Supervisor Triage] Metrics stable or improving — skipping synthetic injection and surgery.")
		return
	}

	log.Printf("⚠️  [Supervisor Triage] Metrics declining — running full triage.")

	// If similarity is very low: inject synthetic social phrase pairs.
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

// SeedSystemExperts initializes a base layer of structural processing units
// (e.g. E0/PRON, E1/AUX, E2/INTERROGATIVE) with locked attributes:
// LRMultiplier = 1.0, OutputScale = 0.5, Pinned = true.
func (s *Supervisor) SeedSystemExperts(model *IntentMoE) {
	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	for lIdx, layer := range layers {
		// Ensure tracking arrays are sized
		if len(layer.ExpertPinned) != len(layer.Experts) {
			newPinned := make([]bool, len(layer.Experts))
			copy(newPinned, layer.ExpertPinned)
			layer.ExpertPinned = newPinned
		}
		if len(layer.ExpertRole) != len(layer.Experts) {
			newRole := make([]string, len(layer.Experts))
			copy(newRole, layer.ExpertRole)
			layer.ExpertRole = newRole
		}
		if len(layer.ExpertHealth) != len(layer.Experts) {
			newHealth := make([]float64, len(layer.Experts))
			for i := range newHealth {
				newHealth[i] = 1.0
			}
			copy(newHealth, layer.ExpertHealth)
			layer.ExpertHealth = newHealth
		}
		if len(layer.ExpertLastUsedAt) != len(layer.Experts) {
			newTimes := make([]time.Time, len(layer.Experts))
			now := time.Now()
			for i := range newTimes {
				newTimes[i] = now
			}
			copy(newTimes, layer.ExpertLastUsedAt)
			layer.ExpertLastUsedAt = newTimes
		}

		// Seed and lock parameters for structural experts E0 (PRON), E1 (VERB), E2 (AUX), E3 (ADJ), E4 (NOUN), E5 (PREP), E6 (GREET), E7 (OTHER)
		for i := 0; i < 8 && i < len(layer.Experts); i++ {
			layer.ExpertPinned[i] = true
			if layer.ExpertRole[i] == "" && i < len(GrammarRoles) {
				layer.ExpertRole[i] = GrammarRoles[i]
			}
			s.SetExpertVariables(model, lIdx, i, 0.5, 1.0)
		}

		// Apply specialised routing biases only when the layer is large enough.
		// These experts are spawned dynamically, so smaller layers simply skip them.
		if len(layer.Experts) > 9 {
			s.SetExpertVariables(model, lIdx, 9, 0.8, 1.0) // E9 → VERB/AUX
		}
		if len(layer.Experts) > 10 {
			s.SetExpertVariables(model, lIdx, 10, 0.8, 1.0) // E10 → PRON
		}
		if len(layer.Experts) > 13 {
			s.SetExpertVariables(model, lIdx, 13, 0.8, 1.0) // E13 → CONJ/PREP
		}
	}
	log.Printf("🧬 [Supervisor] SeedSystemExperts complete. Seeded structural experts with OutputScale=0.5, LRMult=1.0, Pinned=true across all %d layers.", len(layers))
}

// GetSpawnsThisEpoch returns the current spawns count this epoch in a thread-safe manner.
func (s *Supervisor) GetSpawnsThisEpoch() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.SpawnsThisEpoch
}

// IncrementSpawnsThisEpoch increments the spawns count this epoch in a thread-safe manner.
func (s *Supervisor) IncrementSpawnsThisEpoch() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.SpawnsThisEpoch++
}

// SpawnSpecializedExpert manually registers a specialized expert in the given layer.
func (s *Supervisor) SpawnSpecializedExpert(model *IntentMoE, layerIdx int, roleName string, expertID int) {
	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	if layerIdx < 0 || layerIdx >= len(layers) {
		log.Printf("⚠️ SpawnSpecializedExpert: layerIdx %d out of range", layerIdx)
		return
	}
	layer := layers[layerIdx]

	roleID := 0 // Default to PRON for IDENTITY
	if roleName != "IDENTITY" {
		roleID = GrammarRoleIndex(roleName)
	}

	for len(layer.Experts) <= expertID {
		currRoleID := roleID
		if len(layer.Experts) < expertID {
			currRoleID = 7 // OTHER for intermediate ones
		}
		err := s.AddExpertToLayer(model, layerIdx, currRoleID)
		if err != nil {
			log.Printf("⚠️ SpawnSpecializedExpert error: %v", err)
			break
		}
	}

	// Ensure the spawned expert's role name matches
	if expertID < len(layer.ExpertRole) {
		layer.ExpertRole[expertID] = roleName
	}
	if expertID < len(layer.Experts) {
		if ge, ok := layer.Experts[expertID].(*GrammarExpert); ok {
			ge.RoleName = roleName
		}
	}
	log.Printf("🧬 [Supervisor] SpawnSpecializedExpert: Expert E%d (role=%s) spawned in layer %d", expertID, roleName, layerIdx)
}

// AdjustRoutingAffinity manually adjusts routing affinity for a given token to an expert.
func (s *Supervisor) AdjustRoutingAffinity(model *IntentMoE, tokenStr string, expertID int, affinityWeight float32) {
	if model.SentenceVocab == nil || model.Embedding == nil {
		log.Printf("⚠️ AdjustRoutingAffinity: SentenceVocab or Embedding is nil")
		return
	}

	tokenID := model.SentenceVocab.GetTokenID(tokenStr)
	if tokenID < 0 || tokenID >= model.Embedding.VocabSize {
		log.Printf("⚠️ AdjustRoutingAffinity: token '%s' not found in vocab/embedding", tokenStr)
		return
	}

	embeddingDim := model.EmbeddingDim
	tokenEmbedding := model.Embedding.Weight.Data[tokenID*embeddingDim : (tokenID+1)*embeddingDim]

	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	for lIdx, layer := range layers {
		if layer.GatingNetwork == nil || layer.GatingNetwork.Linear == nil {
			continue
		}
		weights := layer.GatingNetwork.Linear.Weights
		numExperts := weights.Shape[1]

		// 🛡️ Layer-expert-count guard: if this layer has fewer physical experts
		// than the requested expertID (e.g. Layer 1 with only 16 experts), map
		// the affinity to expert E0 (base fallback) so the gating column is
		// always valid rather than silently skipping the adjustment.
		targetExpertID := expertID
		if targetExpertID >= len(layer.Experts) {
			log.Printf("⚠️ AdjustRoutingAffinity: expertID %d >= len(Experts) %d for layer %d — remapping to base expert E0",
				expertID, len(layer.Experts), lIdx)
			targetExpertID = 0
		}
		// Also guard against gating-matrix width misalignment.
		if targetExpertID >= numExperts {
			log.Printf("⚠️ AdjustRoutingAffinity: targetExpertID %d out of gating range for layer %d (gating has %d cols)", targetExpertID, lIdx, numExperts)
			continue
		}

		// Adjust column targetExpertID in weights: shape [inputDim, numExperts]
		for r := 0; r < embeddingDim; r++ {
			weights.Data[r*numExperts+targetExpertID] += affinityWeight * tokenEmbedding[r]
		}

		if layer.GatingNetwork.NoiseLinear != nil {
			nWeights := layer.GatingNetwork.NoiseLinear.Weights
			for r := 0; r < embeddingDim; r++ {
				nWeights.Data[r*numExperts+targetExpertID] += affinityWeight * tokenEmbedding[r]
			}
		}
		if targetExpertID != expertID {
			log.Printf("🧬 [Supervisor] Routing Affinity for token '%s': E%d remapped to E%d in Layer %d (strength=%.2f)", tokenStr, expertID, targetExpertID, lIdx, affinityWeight)
		} else {
			log.Printf("🧬 [Supervisor] Adjusted Routing Affinity for token '%s' to Expert E%d in Layer %d (strength=%.2f)", tokenStr, expertID, lIdx, affinityWeight)
		}
	}
}

// ClearFailureLogs clears the failure history counters and overrides for all experts.
func (s *Supervisor) ClearFailureLogs(model *IntentMoE) {
	s.mu.Lock()
	s.FailureLogs = make(map[string]int)
	s.expertOverrides = make(map[int]map[int]*ExpertOverride)
	s.mu.Unlock()

	// Reset all expert output scales and LR multipliers
	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}
	for _, layer := range layers {
		for eIdx := range layer.Experts {
			defaultScale := float32(1.0)
			if eIdx < 8 {
				defaultScale = float32(0.5)
			}
			layer.ExpertOutputScale[eIdx] = defaultScale
			if eIdx < len(layer.ExpertGradMultiplier) {
				layer.ExpertGradMultiplier[eIdx] = 1.0
			}
		}
	}
	log.Println("♻️ [Supervisor] Cleared failure history counters and reset expert scales.")
}

// isTechnicalPayload returns true for multi-word or domain-specific token targets that
// should NOT be collapsed into generic social roles like GREET. Short single-word social
// phrases (hello, hi, thanks, etc.) are explicitly NOT technical.
//
// The key invariant: if a multi-turn technical conversation about encryption loops or
// cache hits is being scanned, its token target is multi-word and must route to OTHER,
// not to GREET. A GREET expert spawned for technical tokens is a pure noise injection.
func isTechnicalPayload(targetToken string) bool {
	lower := strings.ToLower(strings.TrimSpace(targetToken))
	// Explicit social phrase allowlist — these are never "technical"
	socialPhrases := []string{
		"hello", "hi", "hey", "thanks", "thank you",
		"bye", "goodbye", "good morning", "good night",
		"how are you", "nice to meet you",
	}
	for _, p := range socialPhrases {
		if lower == p {
			return false
		}
	}
	// Any input with 3+ words that isn't on the social allowlist is technical context
	return len(strings.Fields(lower)) >= 3
}

// MinStandardExpertRatio is the minimum fraction of layer experts that must remain
// structural/standard experts at all times. Evictions that would breach this floor
// are blocked to preserve the baseline linguistic capability of the network.
// At 64 experts, this protects a floor of at least 26 standard experts.
const MinStandardExpertRatio = 0.40

// CanEvict returns true if the expert at expertIdx in the given layer can be safely
// evicted without breaching the 40% standard-expert floor. Non-structural experts
// (GREET, OTHER, etc.) are always evictable. Structural experts (PRON, VERB, AUX,
// ADJ, NOUN, PREP, or untagged) are blocked from eviction once the floor is reached.
func (s *Supervisor) CanEvict(layer *MoELayer, expertIdx int) bool {
	if expertIdx < 0 || expertIdx >= len(layer.Experts) {
		return false
	}

	role := ""
	if expertIdx < len(layer.ExpertRole) {
		role = layer.ExpertRole[expertIdx]
	}
	isStandard := (role == "PRON" || role == "VERB" || role == "AUX" ||
		role == "ADJ" || role == "NOUN" || role == "PREP" || role == "")

	if !isStandard {
		return true // Non-structural experts (GREET, OTHER) can always be evicted
	}

	// Count how many standard experts currently exist
	standardCount := 0
	for i := range layer.Experts {
		r := ""
		if i < len(layer.ExpertRole) {
			r = layer.ExpertRole[i]
		}
		if r == "PRON" || r == "VERB" || r == "AUX" ||
			r == "ADJ" || r == "NOUN" || r == "PREP" || r == "" {
			standardCount++
		}
	}

	floorCount := int(float64(len(layer.Experts)) * MinStandardExpertRatio)
	if standardCount <= floorCount {
		log.Printf("🛡️ [Supervisor] Eviction of E%d (role=%s) blocked: standard expert floor (%.0f%% = %d/%d experts) would be breached.",
			expertIdx, role, MinStandardExpertRatio*100, standardCount, len(layer.Experts))
		return false
	}
	return true
}
