package chat

import (
	"fmt"
	"math"

	"github.com/golangast/gollemer/internal/ai/moe"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// IsStuck detects a stuck decoder: consecutive repetition or all-unique word salad.
func IsStuck(tokens []string, threshold float32) bool {
	if len(tokens) < 5 {
		return false
	}

	// Repetition: same token dominates the last 10 positions.
	last := tokens[len(tokens)-1]
	repeats := 0
	for i := len(tokens) - 2; i >= max(0, len(tokens)-10); i-- {
		if tokens[i] == last {
			repeats++
		}
	}
	if float32(repeats) >= threshold*10 && len(tokens) >= 10 {
		return true
	}

	// Word salad: high-entropy, very short unique tokens.
	if len(tokens) >= 8 {
		seen := make(map[string]bool)
		total := 0
		for _, t := range tokens {
			seen[t] = true
			total += len(t)
		}
		avgLen := float32(total) / float32(len(tokens))
		uniqueRatio := float32(len(seen)) / float32(len(tokens))
		if avgLen < 2.0 && uniqueRatio > 0.85 {
			return true
		}
	}

	return false
}

// InspectRouterWeights prints the L1 magnitude of each layer's gating network.
// Zero magnitude means the router will pin all tokens to Expert 0.
func InspectRouterWeights(_ *moe.IntentMoE) {
	fmt.Println(" Inspecting Router Integrity...")
	for i, layer := range moe.ActiveLayers {
		var sum float32
		for _, v := range layer.GatingNetwork.Linear.Weights.Data {
			sum += float32(math.Abs(float64(v)))
		}
		if sum == 0 {
			fmt.Printf("  LAYER %d ALERT: Router weights are all ZEROS! (Inference will pin to E0)\n", i)
		} else {
			fmt.Printf("  Layer %d: Router weight magnitude = %.4f\n", i, sum)
		}
	}
}

// PrepareTrainingWeights initialises vocabulary-level punctuation loss weights.
func PrepareTrainingWeights(vocab *mainvocab.Vocabulary) {
	resolvePunctuationWeights(vocab)
}

// findMoELayers returns all MoE layers in the model (encoder layers + decoder output layer).
func findMoELayers(m *moe.IntentMoE) []*moe.MoELayer {
	if m == nil {
		return nil
	}
	layers := m.Encoder.GetMoELayers()
	if m.Decoder != nil && m.Decoder.OutputMoE != nil {
		layers = append(layers, m.Decoder.OutputMoE)
	}
	return layers
}

// toFloat32 hashes string tokens to float32 IDs for tensor construction.
func toFloat32(tokens []string) []float32 {
	out := make([]float32, len(tokens))
	for i, t := range tokens {
		var h uint32
		for _, ch := range t {
			h = h*31 + uint32(ch)
		}
		out[i] = float32(h % 10000)
	}
	return out
}

// extractExpertRoutingInfo collects expert IDs and ACTUAL vocabulary token IDs from all MoE layers.
func extractExpertRoutingInfo(intentModel *moe.IntentMoE, inputTensor *tensor.Tensor, targetTensor *tensor.Tensor) ([]int, []int) {
	expertIDs := make([]int, 0)
	tokenIDs := make([]int, 0)

	// Process Encoder Layers (mapped to inputTensor)
	for _, layer := range intentModel.Encoder.GetMoELayers() {
		if layer != nil && layer.ExpertTokenIndices != nil {
			for expertIdx, seqIndices := range layer.ExpertTokenIndices {
				for _, seqIdx := range seqIndices {
					if inputTensor != nil && seqIdx < len(inputTensor.Data) {
						vocabID := int(inputTensor.Data[seqIdx])
						expertIDs = append(expertIDs, expertIdx)
						tokenIDs = append(tokenIDs, vocabID)
					}
				}
			}
		}
	}

	// Process Decoder Layers (mapped to targetTensor)
	if intentModel.Decoder.OutputMoE != nil {
		layer := intentModel.Decoder.OutputMoE
		if layer.ExpertTokenIndices != nil {
			for expertIdx, seqIndices := range layer.ExpertTokenIndices {
				for _, seqIdx := range seqIndices {
					if targetTensor != nil && seqIdx < len(targetTensor.Data) {
						vocabID := int(targetTensor.Data[seqIdx])
						expertIDs = append(expertIDs, expertIdx)
						tokenIDs = append(tokenIDs, vocabID)
					}
				}
			}
		}
	}

	return expertIDs, tokenIDs
}
