package chat

import (
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// TrainingMetric is a snapshot of one training step, compatible with WASM dashboards.
type TrainingMetric struct {
	Step              int     `json:"step"`
	Loss              float32 `json:"loss"`
	LoadBalanceLoss   float32 `json:"lb_loss"`
	LearningRate      float32 `json:"lr"`
	ActiveExperts     []int   `json:"active_experts"`  // Expert IDs used this batch
	IsCooling         bool    `json:"is_cooling"`      // CoolingOptimizer state
	CircuitBreaker    bool    `json:"circuit_breaker"` // True if a router shake was triggered
	Temperature       float32 `json:"temperature"`     // ThawScheduler temperature
	ThawedExpertCount int     `json:"thawed_count"`    // Number of active expert clusters
}

// PrepareTrainingSequence flattens a ConversationSample into token IDs and a loss mask.
//
// Speaker turns are delimited by <|im_start|> and <|im_end|> control tokens.
// Loss masking:
//
//   - Control/role prefix tokens → 0.0 (never train on these)
//   - User content tokens        → 0.0
//   - Assistant content tokens   → 1.0 (only these drive the gradient)
//   - Padding tokens             → 0.0
//
// If windowSize > 0, the slices are padded or truncated to that length.
func PrepareTrainingSequence(
	conv ConversationSample,
	vocab *mainvocab.Vocabulary,
	windowSize int,
) (tokens []int32, lossMask []float32) {
	lookup := func(word string) int32 {
		if id, ok := vocab.WordToToken[word]; ok {
			return int32(id)
		}
		return int32(vocab.GetTokenID("UNK"))
	}

	imStart := lookup("<|im_start|>")
	imEnd := lookup("<|im_end|>")
	newline := lookup("\n")

	for _, turn := range conv.Dialogue {
		// Role prefix: <|im_start|> ROLE \n
		for _, id := range []int32{imStart, lookup(string(turn.Role)), newline} {
			tokens = append(tokens, id)
			lossMask = append(lossMask, 0.0)
		}

		// Content tokens — only assistant turns contribute to the gradient.
		var maskVal float32
		if turn.Role == RoleAssistant {
			maskVal = 1.0
		}
		for _, word := range cleanTokenize(turn.Content) {
			tokens = append(tokens, lookup(word))
			lossMask = append(lossMask, maskVal)
		}

		tokens = append(tokens, imEnd)
		lossMask = append(lossMask, 0.0)
	}

	// Pad or truncate to the requested window size for SIMD-aligned batching.
	if windowSize > 0 {
		padID := int32(vocab.PaddingTokenID)
		for len(tokens) < windowSize {
			tokens = append(tokens, padID)
			lossMask = append(lossMask, 0.0)
		}
		if len(tokens) > windowSize {
			tokens = tokens[:windowSize]
			lossMask = lossMask[:windowSize]
		}
	}

	return tokens, lossMask
}
