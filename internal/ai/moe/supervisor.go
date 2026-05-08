package moe

import (
	"log"
	"math"
	"math/rand"
	"strings"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
)

// Supervisor monitors the training state and performs autonomous repairs on MoE layers.
type Supervisor struct {
	BestPerplexity float32
	PlateauCount   int
	MumbleCount    int
	LastHealStep   int
	JustPerformedSurgery bool
}

// NewSupervisor initializes a new training supervisor.
func NewSupervisor() *Supervisor {
	return &Supervisor{
		BestPerplexity: 1e9,
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

// PerformSurgery identifies and repairs collapsed experts by cloning better ones.
func (s *Supervisor) PerformSurgery(model *IntentMoE) {
	log.Println("🏥 Supervisor: Performing Expert Surgery...")
	
	layers := model.Encoder.GetMoELayers()
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	for i, layer := range layers {
		alphaID := -1
		sinkID := -1
		maxStrength := float32(-1.0)
		minStrength := float32(1e9)
		
		// Use weight magnitude (L2 norm) as a proxy for expert "strength" / specialization.
		for eIdx, expert := range layer.Experts {
			params := expert.Parameters()
			var l2 float32
			for _, p := range params {
				for _, v := range p.Data {
					l2 += v * v
				}
			}
			
			// Dead expert check
			if l2 < 1e-4 {
				sinkID = eIdx
			}
			
			if l2 > maxStrength {
				maxStrength = l2
				alphaID = eIdx
			}
			if l2 < minStrength && l2 > 1e-4 {
				minStrength = l2
			}
		}
		
		// If we found a collapsed expert, clone the strongest one (alpha) into it.
		if sinkID != -1 && alphaID != -1 && alphaID != sinkID {
			log.Printf("🧬 Surgery (Layer %d): Expert E%d (Collapsed) replaced by E%d (Alpha).\n", i, sinkID, alphaID)
			layer.PerformSurgery(alphaID, sinkID)
			s.JustPerformedSurgery = true
		} else if maxStrength > 500.0 && i == 0 {
			// If an expert is becoming too dominant, we might want to clone it to 
			// the weakest non-dead expert to encourage competition.
			weakestID := -1
			weakestL2 := float32(1e9)
			for eIdx, expert := range layer.Experts {
				params := expert.Parameters()
				var l2 float32
				for _, p := range params {
					for _, v := range p.Data {
						l2 += v * v
					}
				}
				if l2 < weakestL2 {
					weakestL2 = l2
					weakestID = eIdx
				}
			}
			if weakestID != -1 && weakestID != alphaID {
				log.Printf("🧬 Surgery (Layer %d): Expert E%d (Alpha) cloned to E%d (Weak) to increase diversity.\n", i, alphaID, weakestID)
				layer.PerformSurgery(alphaID, weakestID)
			}
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
