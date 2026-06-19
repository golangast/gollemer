package chat

import (
	"log"
	"math"
	"math/rand"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func WeightedCrossEntropy(logits *tensor.Tensor, targets []int, weights []float32, labelSmoothing float32, entropyWeight float32) (float32, *tensor.Tensor) {
	// Flatten batch and sequence dimensions to handle 3D tensors [Batch, Seq, Vocab]
	vocabSize := logits.Shape[len(logits.Shape)-1]
	numClasses := vocabSize
	numRows := len(logits.Data) / numClasses
	grad := tensor.NewTensor(logits.Shape, make([]float32, len(logits.Data)), false)

	var totalLoss float32
	var count float32
	lsLabel := labelSmoothing / float32(numClasses)

	for i := 0; i < numRows; i++ {
		if i >= len(targets) {
			break
		}
		targetID := targets[i]

		// 1. Skip if weight is 0 (Padding)
		if weights[targetID] == 0.0 {
			continue
		}

		offset := i * numClasses
		row := logits.Data[offset : offset+numClasses]

		// 2. Optimized Softmax via SIMD (includes max, exp, sum, and normalization)
		sumExp := moe.SimdSoftmaxF32(row)

		//  NUMERICAL SAFETY: Check for NaNs
		if math.IsNaN(float64(sumExp)) || math.IsInf(float64(sumExp), 0) {
			if rand.Float32() < 0.01 {
				log.Printf(" [WeightedCrossEntropy] NaNs in row %d! Skipping.", i)
			}
			continue
		}

		// 3. Loss (log-prob of target)
		prob := row[targetID]
		loss := -float32(math.Log(float64(prob + 1e-12)))

		currentWeight := weights[targetID]
		weightCap := float32(5.0)
		if currentWeight >= 50.0 {
			weightCap = 100.0
		}
		if currentWeight > weightCap {
			currentWeight = weightCap
		}
		if puncWeight, ok := ResolvedPunctuationWeights[targetID]; ok {
			currentWeight *= puncWeight
		}
		if weights[targetID] < 0.1 {
			currentWeight *= 0.5
		}

		totalLoss += loss * currentWeight
		count++

		// 4. FUSED gradient + entropy
		var rowEntropy float32
		if entropyWeight > 0 {
			for j := 0; j < numClasses; j++ {
				sj := row[j]
				if sj > 1e-12 {
					rowEntropy -= sj * float32(math.Log(float64(sj)))
				}
			}
		}

		gradOut := grad.Data[offset : offset+numClasses]
		if labelSmoothing > 0 {
			if entropyWeight > 0 {
				for j := 0; j < numClasses; j++ {
					sj := row[j]
					targetProb := lsLabel
					if j == targetID {
						targetProb += (1.0 - labelSmoothing)
					}
					g := sj - targetProb
					if sj > 1e-12 {
						g -= entropyWeight * sj * (rowEntropy + float32(math.Log(float64(sj))))
					}
					gradOut[j] = g * currentWeight
				}
			} else {
				for j := 0; j < numClasses; j++ {
					sj := row[j]
					targetProb := lsLabel
					if j == targetID {
						targetProb += (1.0 - labelSmoothing)
					}
					gradOut[j] = (sj - targetProb) * currentWeight
				}
			}
		} else {
			if entropyWeight > 0 {
				for j := 0; j < numClasses; j++ {
					sj := row[j]
					g := sj
					if j == targetID {
						g -= 1.0
					}
					if sj > 1e-12 {
						g -= entropyWeight * sj * (rowEntropy + float32(math.Log(float64(sj))))
					}
					gradOut[j] = g * currentWeight
				}
			} else {
				// Fast path: SIMD multiplication for the bulk of the gradient
				moe.SimdMulScalarF32(gradOut, row, currentWeight)
				gradOut[targetID] = (row[targetID] - 1.0) * currentWeight
			}
		}
	}

	if count > 0 {
		avgLoss := totalLoss / count
		for i := range grad.Data {
			grad.Data[i] /= count
		}
		return avgLoss, grad
	}
	return 0, grad
}
