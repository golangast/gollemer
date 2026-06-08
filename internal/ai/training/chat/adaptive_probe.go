package chat

import (
	"log"
	"math"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// ProbeResult captures the quality of generation from a single test prompt.
type ProbeResult struct {
	Prompt      string
	Generated   string
	IsCoherent  bool    // passes quality gate
	TopProb     float32 // probability of the top-1 token at step 0
	UniqueRatio float32 // unique words / total words (low = repetition)
}

// AdaptiveProbeReport is the aggregate result of running all test probes.
type AdaptiveProbeReport struct {
	Results        []ProbeResult
	CoherentCount  int
	TotalCount     int
	AvgTopProb     float32
	AvgUniqueRatio float32
	Recommendation string // "continue", "increase_lr", "reset_experts", "abort"
}

// TestPrompts are the prompts used to evaluate the model during training.
var TestPrompts = []string{
	"how are you",
	"what is your name",
	"tell me a joke",
	"i feel happy today",
}

// RunAdaptiveProbe runs the EXACT same inference path as the -llm flag
// and returns a report with recommendations for training adjustments.
func RunAdaptiveProbe(model *moe.IntentMoE, epoch int, prevReport *AdaptiveProbeReport) *AdaptiveProbeReport {
	if model == nil || model.SentenceVocab == nil || model.Decoder == nil {
		return &AdaptiveProbeReport{Recommendation: "continue"}
	}

	report := &AdaptiveProbeReport{
		TotalCount: len(TestPrompts),
	}

	// Switch to inference mode
	model.SetMode(false)
	defer model.SetMode(true)

	for _, prompt := range TestPrompts {
		result := probeOnePrompt(model, prompt)
		report.Results = append(report.Results, result)
		if result.IsCoherent {
			report.CoherentCount++
		}
		report.AvgTopProb += result.TopProb
		report.AvgUniqueRatio += result.UniqueRatio
	}

	if len(report.Results) > 0 {
		report.AvgTopProb /= float32(len(report.Results))
		report.AvgUniqueRatio /= float32(len(report.Results))
	}

	// Decide recommendation
	report.Recommendation = decideAction(report, prevReport, epoch)

	// Log the report
	logProbeReport(report, epoch)

	return report
}

// probeOnePrompt runs a single prompt through the SAME path as the LLM inference.
func probeOnePrompt(model *moe.IntentMoE, prompt string) ProbeResult {
	result := ProbeResult{Prompt: prompt}

	// Use the new Guided Generation logic that applies Intent Boosting and Grammar Pressure
	generatedText, _ := model.GenerateGuidedSentence(prompt, 20)
	result.Generated = generatedText

	// Quality gate
	result.IsCoherent = isCoherentOutput(result.Generated)
	result.UniqueRatio = calcUniqueRatio(result.Generated)

	// Rough confidence measure
	result.TopProb = 0.1
	if result.IsCoherent {
		result.TopProb = 0.8
	}

	return result
}

// isCoherentOutput applies the same quality gate as the LLM client.
func isCoherentOutput(text string) bool {
	words := strings.Fields(text)
	if len(words) < 1 {
		return false
	}

	// 1. Check for blatant repetition (loops)
	repeatCount := 0
	for i := 1; i < len(words); i++ {
		if words[i] == words[i-1] {
			repeatCount++
		}
	}
	if repeatCount > len(words)/2 {
		return false // Too much "doing doing doing"
	}

	// 2. Check for long-range loops (A B A B A B)
	if len(words) >= 6 {
		loopCount := 0
		for i := 2; i < len(words); i++ {
			if words[i] == words[i-2] {
				loopCount++
			}
		}
		if loopCount > len(words)/2 {
			return false // Too much "are doing are doing"
		}
	}

	// 3. Check unique word ratio
	unique := make(map[string]bool)
	for _, w := range words {
		unique[strings.ToLower(w)] = true
	}
	uniqueRatio := float64(len(unique)) / float64(len(words))

	// Word salad cannot be reliably detected by high unique ratio in short sentences,
	// because "i am happy today" has a unique ratio of 1.0. We only penalize extremely
	// long sequences of completely unique words, or we skip this heuristic.
	if uniqueRatio == 1.0 && len(words) > 15 {
		return false // 15+ completely random words with no stop-word repetition
	}

	// Too repetitive: very few unique words
	if uniqueRatio < 0.25 && len(words) > 4 {
		return false
	}

	// 4. Check for common conversational patterns
	lower := strings.ToLower(text)
	hasGrammar := false
	grammarIndicators := []string{
		"i'm", "i am", "you", "that", "this", "the", "is", "are",
		"do", "can", "how", "what", "yes", "no", "thanks", "hello",
		"hi", "hey", "good", "great", "fine", "well", "nice",
	}
	for _, g := range grammarIndicators {
		if strings.Contains(lower, g) {
			hasGrammar = true
			break
		}
	}

	return hasGrammar
}

// calcUniqueRatio returns the ratio of unique words to total words.
func calcUniqueRatio(text string) float32 {
	words := strings.Fields(text)
	if len(words) == 0 {
		return 0
	}
	unique := make(map[string]bool)
	for _, w := range words {
		unique[strings.ToLower(w)] = true
	}
	return float32(len(unique)) / float32(len(words))
}

// decideAction determines what the training loop should do based on the probe.
func decideAction(report, prevReport *AdaptiveProbeReport, epoch int) string {
	// If any prompt produces coherent output, we're on the right track
	if report.CoherentCount > 0 {
		return "continue"
	}

	// Check if we're making progress compared to last probe
	if prevReport != nil {
		improving := report.AvgTopProb > prevReport.AvgTopProb*1.1 // 10% improvement
		if improving {
			return "continue" // Getting better, keep going
		}
	}

	// No coherent output and no improvement
	if epoch >= 10 {
		// After 10 epochs with no coherent output, try drastic measures
		return "reset_experts"
	}
	if epoch >= 5 {
		// After 5 epochs, try increasing LR
		return "increase_lr"
	}

	// Too early to judge
	return "continue"
}

// logProbeReport prints a clear report to the training log.
func logProbeReport(report *AdaptiveProbeReport, epoch int) {
	log.Printf("╔═══════════════════════════════════════════════════════════╗")
	log.Printf("║  🧪 ADAPTIVE GENERATION PROBE — Epoch %d                  ║", epoch)
	log.Printf("╠═══════════════════════════════════════════════════════════╣")

	for _, r := range report.Results {
		status := "❌ WORD SALAD"
		if r.IsCoherent {
			status = "✅ COHERENT"
		}
		// Truncate output for readability
		gen := r.Generated
		if len(gen) > 80 {
			gen = gen[:60] + "..."
		}
		log.Printf("║  Q: %-20s → %s", r.Prompt, status)
		log.Printf("║  A: %s", gen)
	}

	log.Printf("╠═══════════════════════════════════════════════════════════╣")
	log.Printf("║  Coherent: %d/%d | Avg Unique Ratio: %.2f", report.CoherentCount, report.TotalCount, report.AvgUniqueRatio)

	switch report.Recommendation {
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

// ApplyProbeRecommendation applies the probe's recommendation to the training state.
// Returns the new learning rate.
func ApplyProbeRecommendation(report *AdaptiveProbeReport, model *moe.IntentMoE, currentLR float32) float32 {
	switch report.Recommendation {
	case "increase_lr":
		newLR := currentLR * 2.0
		if newLR > 0.01 {
			newLR = 0.01
		}
		log.Printf("⚡ [Adaptive] LR increased: %.6f → %.6f", currentLR, newLR)
		return newLR

	case "reset_experts":
		log.Printf("🔄 [Adaptive] Resetting stagnant experts and shaking routers...")
		model.ShakeRouters(0.15)
		// Also bump LR
		newLR := currentLR * 1.5
		if newLR > 0.01 {
			newLR = 0.01
		}
		return newLR

	default:
		return currentLR
	}
}

// RunMLMAdaptiveProbe runs a quick MLM-specific probe during pre-training.
// It checks if the model can predict simple masked words.
func RunMLMAdaptiveProbe(model *moe.IntentMoE, mlmHead *MLMHead, epoch int) bool {
	if model == nil || model.SentenceVocab == nil || mlmHead == nil {
		return true // can't probe, continue training
	}

	type testCase struct {
		tokens  []string
		maskIdx int
		expect  string
	}

	tests := []testCase{
		{[]string{"how", "are", "[MASK]"}, 2, "you"},
		{[]string{"i", "am", "[MASK]"}, 2, "good"},
	}

	correctCount := 0
	for _, tc := range tests {
		tokenIDs := make([]float32, len(tc.tokens))
		for i, t := range tc.tokens {
			id := model.SentenceVocab.GetTokenID(t)
			if id < 0 {
				id = 1
			}
			tokenIDs[i] = float32(id)
		}

		inputT := tensor.NewTensor([]int{1, len(tc.tokens)}, tokenIDs, false)
		emb, err := model.Embedding.Forward(inputT)
		if err != nil {
			continue
		}
		enc, err := model.Encoder.Forward(emb)
		if err != nil {
			continue
		}
		if model.EncoderNorm != nil {
			enc, _ = model.EncoderNorm.Forward(enc)
		}
		logits, err := mlmHead.Forward(enc)
		if err != nil {
			continue
		}

		if len(logits.Shape) == 3 {
			vs := logits.Shape[2]
			offset := tc.maskIdx * vs
			if offset+vs <= len(logits.Data) {
				maskLogits := logits.Data[offset : offset+vs]
				// Find argmax
				bestIdx := 0
				bestVal := float32(-math.MaxFloat32)
				for i, v := range maskLogits {
					if v > bestVal {
						bestVal = v
						bestIdx = i
					}
				}
				predicted := model.SentenceVocab.GetWord(bestIdx)
				if predicted == tc.expect {
					correctCount++
				}
				log.Printf("🧪 [MLM Probe] \"%s\" → predicted: \"%s\" (expected: \"%s\")",
					strings.Join(tc.tokens, " "), predicted, tc.expect)
			}
		}
	}

	return correctCount > 0 // true = making progress
}
