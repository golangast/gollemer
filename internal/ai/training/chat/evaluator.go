package chat

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func ValidateModelHealth(model *moe.IntentMoE) bool {
	fmt.Println(" Performing Pre-Flight Health Check...")
	isHealthy := true

	for i, param := range model.Parameters() {
		var maxVal float32 = -1e18
		var minVal float32 = 1e18
		nanCount := 0

		for _, v := range param.Data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				nanCount++
			}
			if v > maxVal {
				maxVal = v
			}
			if v < minVal {
				minVal = v
			}
		}

		// Check for NaNs
		if nanCount > 0 {
			fmt.Printf(" Param %d: Found %d NaN/Inf values!\n", i, nanCount)
			isHealthy = false
		}

		// Check for Weight Saturation
		if maxVal > 100.0 || minVal < -100.0 {
			fmt.Printf("  Param %d: High saturation detected (Range: %.2f to %.2f)\n", i, minVal, maxVal)
		}
	}

	if isHealthy {
		fmt.Println(" Model weights are within safe numerical bounds.")
	}
	return isHealthy
}

func VerifyModelIntegrity(m *moe.IntentMoE) {
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		w := m.Decoder.OutputMoE.GatingNetwork.Linear.Weights.Data
		var sum float32 = 0.0
		for _, v := range w {
			sum += float32(math.Abs(float64(v)))
		}
		if sum == 0 {
			fmt.Println(" CRITICAL: Decoder Router is empty! Resetting weights...")
			// Initialize with small random values to break the E0 tie
			for i := range w {
				w[i] = float32((rand.Float64() - 0.5) * 0.1)
			}
		}
	}
}

func ValidateChat(model *moe.IntentMoE, valPairs []moe.TrainPair, useGPU bool) float32 {
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

	// Ensure we have the UNK ID for validation
	unkID := model.SentenceVocab.GetTokenID("UNK")
	if unkID == -1 {
		unkID = 0 // Fallback
	}

	for _, pair := range valPairs {
		// Tokenize
		qTokens := cleanTokenize(pair.Q)
		qIDs := make([]float32, max(1, len(qTokens)))
		for i, t := range qTokens {
			id := lookupVocab(t, model.SentenceVocab)
			qIDs[i] = float32(id)
		}
		if len(qTokens) == 0 {
			qIDs[0] = 0 // Pad
		}

		aTokens := cleanTokenize(pair.A)
		aIDs := make([]int, len(aTokens)+2)
		aIDs[0] = model.SentenceVocab.BosID
		for i, t := range aTokens {
			id := model.SentenceVocab.GetTokenID(t)
			if id == -1 || (id == 0 && t != "<pad>") {
				aIDs[i+1] = unkID
			} else {
				aIDs[i+1] = id
			}
		}
		aIDs[len(aIDs)-1] = model.SentenceVocab.EosID

		inputT := tensor.NewTensor([]int{1, len(qIDs)}, qIDs, false)
		// Validation targets (all except BOS)
		targets := aIDs[1:]

		// Forward pass (inference mode)
		targetTIDs := make([]float32, len(aIDs))
		for i, id := range aIDs {
			targetTIDs[i] = float32(id)
		}
		targetT := tensor.NewTensor([]int{1, len(targetTIDs)}, targetTIDs, false)

		if useGPU {
			inputT.ToGPU()
			targetT.ToGPU()
		}

		logits, _, err := model.Forward(0.0, inputT, targetT, nil)
		if err != nil || len(logits) == 0 {
			continue
		}

		// Recreate loss weights for validation context
		valWeights := make([]float32, model.SentenceVocab.Size())
		for i := range valWeights {
			valWeights[i] = 1.0
		}
		valWeights[unkID] = 0.01
		valWeights[model.SentenceVocab.PaddingTokenID] = 0.0

		loss, _ := WeightedCrossEntropy(logits[0], targets, valWeights, 0.0, 0.0)
		totalLoss += float64(loss)
		tokenCount++

		model.Detach()
	}

	if tokenCount == 0 {
		return 0.0
	}

	avgLoss := totalLoss / float64(tokenCount)
	perplexity := float32(math.Exp(avgLoss))
	return perplexity
}

func calculateSequenceReward(predictedIDs []int, vocab *mainvocab.Vocabulary) float32 {
	if vocab == nil || len(predictedIDs) == 0 {
		return 1.0
	}
	var reward float32 = 1.0

	// Find the first meaningful token (skip BOS)
	firstMeaningful := -1
	meaningfulCount := 0
	for _, id := range predictedIDs {
		if id != vocab.BosID && id != vocab.PaddingTokenID && id != vocab.EosID {
			if firstMeaningful == -1 {
				firstMeaningful = id
			}
			meaningfulCount++
		}
	}

	// Extreme penalty for single-word responses
	if meaningfulCount <= 1 {
		reward -= 1.5
	}

	// Bonus: sentence starts with a greeting word
	if firstMeaningful >= 0 {
		w := strings.ToLower(vocab.GetWord(firstMeaningful))
		switch w {
		case "hello", "hi", "hey", "morning", "evening", "greetings", "howdy":
			reward += 0.3
		}
	}

	// Find the last meaningful token (skip EOS/PAD)
	lastMeaningful := -1
	for i := len(predictedIDs) - 1; i >= 0; i-- {
		id := predictedIDs[i]
		if id != vocab.EosID && id != vocab.PaddingTokenID {
			lastMeaningful = id
			break
		}
	}

	// Bonus: sentence ends with terminal punctuation
	if lastMeaningful >= 0 {
		w := vocab.GetWord(lastMeaningful)
		if w == "." || w == "!" || w == "?" {
			reward += 0.5
		}
	}

	// Penalty: adjacent duplicate tokens (word salad signal)
	for i := 0; i < len(predictedIDs)-1; i++ {
		if predictedIDs[i] == predictedIDs[i+1] &&
			predictedIDs[i] != vocab.PaddingTokenID {
			reward -= 0.8
		}
	}

	// Clamp to safe range: never zero out the gradient signal entirely
	if reward < 0.01 {
		reward = 0.01
	}
	if reward > 2.0 {
		reward = 2.0
	}
	return reward
}

func scoreSentenceHeuristic(text string) float32 {
	if text == "[Still Silent]" || text == "" {
		return 0
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return 0
	}

	var score float32 = 10.0

	// 1. Length Penalty
	if len(words) == 1 {
		score -= 8.0 // Heavy penalty for single-token responses
	} else if len(words) < 3 {
		score -= 5.0
	} else if len(words) < 5 {
		score -= 3.0
	}

	// 2. Repetition Penalty
	seen := make(map[string]int)
	for _, w := range words {
		seen[w]++
	}
	var repeatCost float32 = 0.0
	for _, count := range seen {
		if count > 1 {
			repeatCost += float32(math.Pow(float64(count), 1.5))
		}
	}
	score -= repeatCost * 0.5

	// 3. Variety Reward
	score += float32(len(seen)) * 0.2

	// 4. Punctuation Reward
	if strings.ContainsAny(text, ".!?") {
		score += 2.0
	}

	if score < 0 {
		score = 0
	}
	return score
}

func scoreGrammarHeuristic(text string) float32 {
	if text == "[Still Silent]" || text == "" {
		return 0
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return 0
	}
	var score float32 = 0.0

	// Reward just for having content (prevents always-zero on short outputs)
	if len(words) >= 2 {
		score += 1.0
	}

	var lastPos string
	for _, w := range words {
		pos := moe.MapWordToGrammarType(w)
		if pos != "OTHER" {
			score += 1.0 // reward known vocabulary
		}
		// Penalty for consecutive identical POS (e.g. VERB-VERB) unless AUXVERB
		if pos == lastPos && pos != "OTHER" && pos != "PREP" && pos != "ADJ" {
			score -= 1.5
		}
		// === Syntactical rewards for natural sequences ===
		// PRON  VERB: "I am", "you want", "it is"
		if lastPos == "PRON" && (pos == "VERB" || pos == "AUX") {
			score += 2.0
		}
		// AUX  VERB: "can do", "should try", "would love"
		if lastPos == "AUX" && pos == "VERB" {
			score += 2.0
		}
		// VERB  PREP/NOUN/ADJ/PRON: "doing well", "going to", "want to"
		if lastPos == "VERB" && (pos == "PREP" || pos == "NOUN" || pos == "ADJ" || pos == "PRON") {
			score += 1.5
		}
		// GREET  anything: natural greeting start
		if lastPos == "GREET" && pos != "OTHER" {
			score += 1.0
		}
		// INTERROGATIVE  PRON/AUX/VERB: "how are", "what do", "how's"
		if lastPos == "INTERROGATIVE" && (pos == "PRON" || pos == "AUX" || pos == "VERB") {
			score += 2.0
		}
		// ADJ  NOUN: "good day", "nice weather"
		if lastPos == "ADJ" && pos == "NOUN" {
			score += 1.0
		}
		// PREP  NOUN/PRON/ADJ: "to the", "with you", "of course"
		if lastPos == "PREP" && (pos == "NOUN" || pos == "PRON" || pos == "ADJ") {
			score += 0.5
		}
		lastPos = pos
	}
	if score < 0 {
		return 0
	}
	if score > 30.0 {
		return 30.0
	}
	return score
}
