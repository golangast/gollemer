package seq2seq

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// BeamHypothesis is a single hypothesis in beam search.
type BeamHypothesis struct {
	Tokens      []int
	Score       float32 // cumulative log probability
	Hidden      *tensor.Tensor
	Cell        *tensor.Tensor
	HistStates  [][][]float32
	HistTokens  [][]int
	SeenBigrams map[[2]int]bool
	Finished    bool
}

// copyTensor creates a deep copy of a tensor.
func copyTensor(t *tensor.Tensor) *tensor.Tensor {
	if t == nil {
		return nil
	}
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	return tensor.NewTensor(shape, data, t.RequiresGrad)
}

// copyHistStates deep-copies the history states.
func copyHistStates(h [][][]float32) [][][]float32 {
	out := make([][][]float32, len(h))
	for i := range h {
		out[i] = make([][]float32, len(h[i]))
		for j := range h[i] {
			out[i][j] = make([]float32, len(h[i][j]))
			copy(out[i][j], h[i][j])
		}
	}
	return out
}

// copyHistTokens deep-copies the history tokens.
func copyHistTokens(h [][]int) [][]int {
	out := make([][]int, len(h))
	for i := range h {
		out[i] = make([]int, len(h[i]))
		copy(out[i], h[i])
	}
	return out
}

// logSoftmax computes log(softmax(x)) for numerical stability.
func logSoftmax(logits []float32) []float32 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	out := make([]float32, len(logits))
	sumExp := float32(0)
	for i, v := range logits {
		out[i] = v - maxVal
		sumExp += float32(math.Exp(float64(out[i])))
	}
	logSumExp := float32(math.Log(float64(sumExp))) + maxVal
	for i := range out {
		out[i] = logits[i] - logSumExp
	}
	return out
}

// lengthPenalty applies the standard length penalty: score / (len^alpha)
func lengthPenalty(score float32, length int, alpha float64) float32 {
	if length == 0 {
		return score
	}
	return score / float32(math.Pow(float64(length), alpha))
}

// PredictBeam generates text using beam search.
// beamWidth=1 is equivalent to greedy decoding.
// Returns the best sequence by length-penalized log probability.
func (m *Seq2Seq) PredictBeam(query string, maxLen, beamWidth int) (string, error) {
	if beamWidth < 1 {
		beamWidth = 1
	}

	// Exact match shortcut (same as greedy)
	if m != nil && len(m.ExactMap) > 0 {
		if answer, ok := m.ExactMap[strings.ToLower(strings.TrimSpace(query))]; ok {
			return strings.TrimSpace(answer), nil
		}
	}

	if m.Tokenizer == nil {
		return "", fmt.Errorf("seq2seq tokenizer is nil")
	}

	query = NormalizeQuery(query)
	inputTokenIDs, err := m.Tokenizer.Encode(query)
	if err != nil {
		return "", fmt.Errorf("failed to tokenize query: %w", err)
	}

	inputTensorData := make([]float32, len(inputTokenIDs))
	for i, id := range inputTokenIDs {
		inputTensorData[i] = float32(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputTokenIDs)}, inputTensorData, true)

	// Encoder forward pass (once)
	encoderOutputs, encoderHidden, encoderCell, err := m.Encoder.ForwardWithOutputs(inputTensor)
	if err != nil {
		return "", fmt.Errorf("prediction encoder forward failed: %w", err)
	}

	useCopy := m.Decoder.Copy != nil
	var encInputIDs [][]int
	if m.Decoder.EncoderCopy != nil {
		encInputIDs = [][]int{inputTokenIDs}
	}

	// Initialize beams: single hypothesis with BOS
	initialHyp := &BeamHypothesis{
		Tokens:      []int{},
		Score:       0,
		Hidden:      copyTensor(encoderHidden),
		Cell:        copyTensor(encoderCell),
		HistStates:  [][][]float32{{}},
		HistTokens:  [][]int{{}},
		SeenBigrams: map[[2]int]bool{},
		Finished:    false,
	}
	beams := []*BeamHypothesis{initialHyp}

	for step := 0; step < maxLen; step++ {
		var candidates []*BeamHypothesis
		allFinished := true

		for _, hyp := range beams {
			if hyp.Finished {
				candidates = append(candidates, hyp)
				continue
			}
			allFinished = false

			// Determine input token: BOS for first step, else last token
			var inputID int
			if len(hyp.Tokens) == 0 {
				inputID = m.OutputVocab.BosID
			} else {
				inputID = hyp.Tokens[len(hyp.Tokens)-1]
			}

			decoderInput := tensor.NewTensor([]int{1, 1}, []float32{float32(inputID)}, true)
			prediction, hidden, cell, err := m.Decoder.ForwardCopy(
				decoderInput, hyp.Hidden, hyp.Cell,
				hyp.HistStates, hyp.HistTokens,
				encoderOutputs, encInputIDs,
			)
			if err != nil {
				return "", fmt.Errorf("beam decoder forward failed at step %d: %w", step, err)
			}

			// Apply anti-repetition penalties (same as greedy)
			logits := make([]float32, len(prediction.Data))
			copy(logits, prediction.Data)
			if len(hyp.Tokens) > 0 {
				prev := hyp.Tokens[len(hyp.Tokens)-1]
				if prev >= 0 && prev < len(logits) {
					logits[prev] = -1e30
				}
				for c := 0; c < len(logits); c++ {
					if hyp.SeenBigrams[[2]int{prev, c}] {
						logits[c] = -1e30
					}
				}
			}

			// Convert to log probabilities
			logProbs := logSoftmax(logits)

			// Get top-k candidates for this hypothesis
			// We take top beamWidth to limit branching
			type tokenScore struct {
				id    int
				score float32
			}
			scores := make([]tokenScore, len(logProbs))
			for i, lp := range logProbs {
				scores[i] = tokenScore{i, lp}
			}
			sort.Slice(scores, func(a, b int) bool {
				return scores[a].score > scores[b].score
			})

			k := beamWidth
			if k > len(scores) {
				k = len(scores)
			}
			for i := 0; i < k; i++ {
				ts := scores[i]
				newTokens := append(append([]int{}, hyp.Tokens...), ts.id)
				newHyp := &BeamHypothesis{
					Tokens:      newTokens,
					Score:       hyp.Score + ts.score,
					Hidden:      copyTensor(hidden),
					Cell:        copyTensor(cell),
					HistStates:  copyHistStates(hyp.HistStates),
					HistTokens:  copyHistTokens(hyp.HistTokens),
					SeenBigrams: make(map[[2]int]bool, len(hyp.SeenBigrams)+1),
					Finished:    ts.id == m.OutputVocab.EosID,
				}
				for bg := range hyp.SeenBigrams {
					newHyp.SeenBigrams[bg] = true
				}
				if len(hyp.Tokens) > 0 {
					newHyp.SeenBigrams[[2]int{hyp.Tokens[len(hyp.Tokens)-1], ts.id}] = true
				}

				// Update copy history
				if useCopy {
					hs := append([]float32{}, hidden.Data...)
					newHyp.HistStates[0] = append(newHyp.HistStates[0], hs)
					newHyp.HistTokens[0] = append(newHyp.HistTokens[0], ts.id)
				}

				candidates = append(candidates, newHyp)
			}
		}

		if allFinished {
			break
		}

		// Keep top beamWidth by score
		sort.Slice(candidates, func(a, b int) bool {
			return candidates[a].Score > candidates[b].Score
		})
		if len(candidates) > beamWidth {
			candidates = candidates[:beamWidth]
		}
		beams = candidates
	}

	// Select best by length-penalized score
	best := beams[0]
	bestPenalized := lengthPenalty(best.Score, len(best.Tokens), 0.6)
	for _, hyp := range beams[1:] {
		penalized := lengthPenalty(hyp.Score, len(hyp.Tokens), 0.6)
		if penalized > bestPenalized {
			best = hyp
			bestPenalized = penalized
		}
	}

	return m.OutputVocab.Decode(best.Tokens), nil
}
