package llm

import (
	"math/rand"
	"sort"
)

// ScoredIndex represents a class index paired with its model probability score.
type ScoredIndex struct {
	Index int
	Score float64
}

// TopKSample identifies the K most probable indices and randomly selects one 
// based on their relative weighted probabilities. This prevents the model from 
// getting stuck in repetitive "greedy" loops (e.g., punctuation cycles).
func TopKSample(probabilities []float64, k int) int {
	if len(probabilities) == 0 {
		return -1
	}
	if k <= 1 {
		// Degenerate case: behave like ArgMax
		maxIdx := 0
		maxVal := probabilities[0]
		for i, p := range probabilities {
			if p > maxVal {
				maxVal = p
				maxIdx = i
			}
		}
		return maxIdx
	}
	
	if k > len(probabilities) {
		k = len(probabilities)
	}

	// 1. Create a list of indexed scores
	pairs := make([]ScoredIndex, len(probabilities))
	for i, p := range probabilities {
		pairs[i] = ScoredIndex{i, p}
	}

	// 2. Sort by score descending to find the top candidates
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Score > pairs[j].Score
	})

	// 3. Keep only the top K most likely outputs
	topK := pairs[:k]

	// 4. Renormalize the top K scores so they sum is relative to each other
	var sum float64
	for _, p := range topK {
		sum += p.Score
	}

	if sum <= 0 {
		return topK[0].Index
	}

	// 5. Randomly sample from the top K based on normalized weights
	r := rand.Float64() * sum
	var cumulative float64
	for _, p := range topK {
		cumulative += p.Score
		if r <= cumulative {
			return p.Index
		}
	}

	return topK[0].Index
}
