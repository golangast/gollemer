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

	// Pre-allocate a shifted-row scratch buffer for the SIMD label-smoothing path.
	// Reusing across rows avoids per-row heap allocations for the common case.
	var lsBuf []float32
	if labelSmoothing > 0 && entropyWeight == 0 {
		lsBuf = make([]float32, numClasses)
	}

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
		if puncWeight, ok := ResolvedPunctuationWeights[targetID]; ok && puncWeight > 0 {
			currentWeight *= puncWeight
		}
		if currentWeight > weightCap {
			currentWeight = weightCap
		}
		if weights[targetID] < 0.1 {
			currentWeight *= 0.5
		}

		totalLoss += loss * currentWeight
		count++

		gradOut := grad.Data[offset : offset+numClasses]

		// 4. Gradient computation
		if entropyWeight > 0 {
			// Slow path: entropy regularisation requires per-element log — stay scalar.
			var rowEntropy float32
			for j := 0; j < numClasses; j++ {
				sj := row[j]
				if sj > 1e-12 {
					rowEntropy -= sj * float32(math.Log(float64(sj)))
				}
			}
			if labelSmoothing > 0 {
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
					g := sj
					if j == targetID {
						g -= 1.0
					}
					if sj > 1e-12 {
						g -= entropyWeight * sj * (rowEntropy + float32(math.Log(float64(sj))))
					}
					gradOut[j] = g * currentWeight
				}
			}
		} else if labelSmoothing > 0 {
			// SIMD label-smoothing fast path.
			// gradOut[j] = (row[j] - lsLabel) * w   for j != targetID
			// gradOut[targetID] = (row[targetID] - (lsLabel + 1 - ls)) * w
			//
			// Implementation:
			//   lsBuf = row - lsLabel  (i.e. copy row then subtract uniform shift)
			//   gradOut = lsBuf * w
			//   then patch targetID: subtract (1-ls)*w
			moe.SimdMulScalarF32(lsBuf, row, 1.0) // copy row into lsBuf
			// shift all by -lsLabel: lsBuf[j] = row[j] - lsLabel
			for j := range lsBuf { // scalar shift, cheap, 1 pass
				lsBuf[j] -= lsLabel
			}
			// scale: gradOut = lsBuf * currentWeight (SIMD, unrolled 4-way)
			moe.SimdMulScalarF32(gradOut, lsBuf, currentWeight)
			// Patch target: gradOut[tgt] -= (1.0 - labelSmoothing) * currentWeight
			gradOut[targetID] -= (1.0 - labelSmoothing) * currentWeight
		} else {
			// Fastest path: SIMD scale + single scalar patch
			moe.SimdMulScalarF32(gradOut, row, currentWeight)
			gradOut[targetID] = (row[targetID] - 1.0) * currentWeight
		}
	}

	if count > 0 {
		avgLoss := totalLoss / count
		// SIMD: normalise all gradients in one 4-way unrolled vectorised pass.
		moe.SimdScaleF32(grad.Data, 1.0/count)
		return avgLoss, grad
	}
	return 0, grad
}
