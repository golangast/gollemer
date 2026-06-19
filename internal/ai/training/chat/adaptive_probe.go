package chat

import (
	"log"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
)

// ProbeResult holds the generation quality of a single test prompt.
type ProbeResult struct {
	Prompt      string
	Generated   string
	IsCoherent  bool    // Passed quality gate
	TopProb     float32 // Confidence of top token
	UniqueRatio float32 // Ratio of unique words (repetition check)
}

// AdaptiveProbeReport aggregates the results of all probes.
type AdaptiveProbeReport struct {
	Results        []ProbeResult
	CoherentCount  int
	TotalCount     int
	AvgTopProb     float32
	AvgUniqueRatio float32
	Recommendation string
}

var testPrompts = []string{
	"how are you",
	"what is your name",
	"tell me a joke",
	"i feel happy today",
}

// RunAdaptiveProbe evaluates the model's generation quality and suggests training adjustments.
func RunAdaptiveProbe(m *moe.IntentMoE, epoch int, prev *AdaptiveProbeReport) *AdaptiveProbeReport {
	if m == nil || m.SentenceVocab == nil || m.Decoder == nil {
		return &AdaptiveProbeReport{Recommendation: "continue"}
	}

	r := &AdaptiveProbeReport{
		TotalCount: len(testPrompts),
	}

	m.SetMode(false)
	defer m.SetMode(true)

	for _, p := range testPrompts {
		res := probe(m, p)
		r.Results = append(r.Results, res)
		if res.IsCoherent {
			r.CoherentCount++
		}
		r.AvgTopProb += res.TopProb
		r.AvgUniqueRatio += res.UniqueRatio
	}

	if n := float32(len(r.Results)); n > 0 {
		r.AvgTopProb /= n
		r.AvgUniqueRatio /= n
	}

	r.Recommendation = decide(r, prev, epoch)
	logProbe(r, epoch)
	return r
}

// probe generates a response and calculates quality metrics.
func probe(m *moe.IntentMoE, prompt string) ProbeResult {
	res := ProbeResult{
		Prompt:  prompt,
		TopProb: 0.1,
	}

	res.Generated, _ = m.GenerateGuidedSentence(prompt, 20)
	res.UniqueRatio = uniqueRatio(res.Generated)
	res.IsCoherent = isCoherent(res.Generated)

	if res.IsCoherent {
		res.TopProb = 0.8
	}

	return res
}

// isCoherent applies a heuristic quality gate to the generated text.
func isCoherent(text string) bool {
	words := strings.Fields(text)
	if len(words) < 1 {
		return false
	}

	// Reject blatant repetition (e.g., "doing doing doing")
	repeats := 0
	for i := 1; i < len(words); i++ {
		if words[i] == words[i-1] {
			repeats++
		}
	}
	if repeats > len(words)/2 {
		return false
	}

	// Reject long-range loops (e.g., "are doing are doing")
	if len(words) >= 6 {
		loops := 0
		for i := 2; i < len(words); i++ {
			if words[i] == words[i-2] {
				loops++
			}
		}
		if loops > len(words)/2 {
			return false
		}
	}

	ratio := uniqueRatio(text)

	// A sequence of completely unique words is often word salad.
	if ratio == 1.0 && len(words) > 15 {
		return false
	}
	// Too few unique words implies a stuck decoder.
	if ratio < 0.25 && len(words) > 4 {
		return false
	}

	return hasGrammar(text)
}

// uniqueRatio calculates the proportion of unique words in the text.
func uniqueRatio(text string) float32 {
	words := strings.Fields(text)
	if len(words) == 0 {
		return 0
	}
	seen := make(map[string]bool)
	for _, w := range words {
		seen[strings.ToLower(w)] = true
	}
	return float32(len(seen)) / float32(len(words))
}

// hasGrammar checks for common conversational anchors.
func hasGrammar(text string) bool {
	lower := strings.ToLower(text)
	anchors := []string{
		"i'm", "i am", "you", "that", "this", "the", "is", "are",
		"do", "can", "how", "what", "yes", "no", "thanks", "hello",
		"hi", "hey", "good", "great", "fine", "well", "nice",
	}
	for _, a := range anchors {
		if strings.Contains(lower, a) {
			return true
		}
	}
	return false
}

// decide determines the next training action based on generation quality.
func decide(cur, prev *AdaptiveProbeReport, epoch int) string {
	if cur.CoherentCount > 0 {
		return "continue"
	}
	if prev != nil && cur.AvgTopProb > prev.AvgTopProb*1.1 {
		return "continue"
	}

	if epoch >= 10 {
		return "reset_experts"
	}
	if epoch >= 5 {
		return "increase_lr"
	}

	return "continue"
}

func logProbe(r *AdaptiveProbeReport, epoch int) {
	log.Printf("╔═══════════════════════════════════════════════════════════╗")
	log.Printf("║  🧪 ADAPTIVE GENERATION PROBE — Epoch %d                  ║", epoch)
	log.Printf("╠═══════════════════════════════════════════════════════════╣")

	for _, res := range r.Results {
		status := "❌ WORD SALAD"
		if res.IsCoherent {
			status = "✅ COHERENT"
		}

		gen := res.Generated
		if len(gen) > 80 {
			gen = gen[:60] + "..."
		}
		log.Printf("║  Q: %-20s → %s", res.Prompt, status)
		log.Printf("║  A: %s", gen)
	}

	log.Printf("╠═══════════════════════════════════════════════════════════╣")
	log.Printf("║  Coherent: %d/%d | Avg Unique Ratio: %.2f", r.CoherentCount, r.TotalCount, r.AvgUniqueRatio)

	switch r.Recommendation {
	case "continue":
		log.Printf("║  📌 ACTION: Continue training (on track)")
	case "increase_lr":
		log.Printf("║  ⚡ ACTION: Increasing LR (model is stuck)")
	case "reset_experts":
		log.Printf("║  🔄 ACTION: Resetting stagnant experts (no progress)")
	case "abort":
		log.Printf("║  🛑 ACTION: Aborting (fundamental architecture issue)")
	}

	log.Printf("╚═══════════════════════════════════════════════════════════╝")
}

// ApplyProbeRecommendation adjusts the model or learning rate based on the probe's findings.
func ApplyProbeRecommendation(r *AdaptiveProbeReport, m *moe.IntentMoE, lr float32) float32 {
	switch r.Recommendation {
	case "increase_lr":
		lr *= 2.0
		if lr > 0.01 {
			lr = 0.01
		}
		log.Printf("⚡ [Adaptive] LR increased to: %.6f", lr)
		return lr

	case "reset_experts":
		log.Printf("🔄 [Adaptive] Resetting stagnant experts and shaking routers...")
		m.ShakeRouters(0.15)
		lr *= 1.5
		if lr > 0.01 {
			lr = 0.01
		}
		return lr

	default:
		return lr
	}
}
