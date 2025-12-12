package tensor

import (
	"fmt"
	"math"
)

// EmbeddingLookupOperation represents an embedding lookup operation for autograd.
type EmbeddingLookupOperation struct {
	InputIDs *Tensor // Tensor of shape [batch_size, sequence_length]
	Weights  *Tensor // Tensor of shape [vocab_size, embedding_dim]
	Output   *Tensor // Tensor of shape [batch_size, sequence_length, embedding_dim]
}

// Softmax applies the softmax function to the last dimension of the tensor.
func Softmax(tensor *Tensor) *Tensor {
	shape := tensor.Shape
	lastDim := shape[len(shape)-1]
	output := NewTensor(shape, make([]float64, len(tensor.Data)), false)

	for i := 0; i < len(tensor.Data); i += lastDim {
		maxVal := math.Inf(-1)
		for j := 0; j < lastDim; j++ {
			if tensor.Data[i+j] > maxVal {
				maxVal = tensor.Data[i+j]
			}
		}

		sumExp := 0.0
		for j := 0; j < lastDim; j++ {
			sumExp += math.Exp(tensor.Data[i+j] - maxVal)
		}

		for j := 0; j < lastDim; j++ {
			output.Data[i+j] = math.Exp(tensor.Data[i+j]-maxVal) / sumExp
		}
	}

	return output
}

// CrossEntropyLoss calculates the cross-entropy loss with optional label smoothing.
// labelSmoothing: value between 0.0 and 1.0. When > 0, distributes probability mass
// from the target class to all classes to prevent overconfidence.
func CrossEntropyLoss(logits *Tensor, targetIDs []int, padID int, labelSmoothing float64) (float64, *Tensor) {
	// Reshape logits to 2D if it's 3D (batch_size * seq_len, vocab_size)
	originalShape := logits.Shape
	var reshapedLogits *Tensor
	var numClasses int

	if len(originalShape) == 3 {
		batchSize := originalShape[0]
		seqLen := originalShape[1]
		numClasses = originalShape[2]
		var err error
		reshapedLogits, err = logits.Reshape([]int{batchSize * seqLen, numClasses})
		if err != nil {
			panic(fmt.Sprintf("Failed to reshape logits: %v", err))
		}
	} else if len(originalShape) == 2 {
		reshapedLogits = logits
		numClasses = originalShape[1]
	} else {
		// Handle other dimensions or return an error
		panic("Unsupported logits dimension for CrossEntropyLoss")
	}

	probs := Softmax(reshapedLogits)
	loss := 0.0
	activeTokens := 0
	epsilon := 1e-9 // Small value to avoid log(0)

	grad := NewTensor(reshapedLogits.Shape, make([]float64, len(reshapedLogits.Data)), false)

	// Calculate smoothed target distribution
	smoothValue := labelSmoothing / float64(numClasses)
	targetConfidence := 1.0 - labelSmoothing + smoothValue

	for i := 0; i < reshapedLogits.Shape[0]; i++ {
		targetID := targetIDs[i]
		if targetID == padID {
			continue
		}
		activeTokens++

		smoothedLoss := 0.0
		unrollFactor := 8
		j := 0
		baseIndex := i * numClasses

		// Unrolled loop
		for ; j <= numClasses-unrollFactor; j += unrollFactor {
			// Unroll 1
			p1 := probs.Data[baseIndex+j]
			t1 := smoothValue
			if j == targetID {
				t1 = targetConfidence
			}
			smoothedLoss -= t1 * math.Log(p1+epsilon)
			grad.Data[baseIndex+j] = p1 - t1

			// Unroll 2
			p2 := probs.Data[baseIndex+j+1]
			t2 := smoothValue
			if j+1 == targetID {
				t2 = targetConfidence
			}
			smoothedLoss -= t2 * math.Log(p2+epsilon)
			grad.Data[baseIndex+j+1] = p2 - t2

			// Unroll 3
			p3 := probs.Data[baseIndex+j+2]
			t3 := smoothValue
			if j+2 == targetID {
				t3 = targetConfidence
			}
			smoothedLoss -= t3 * math.Log(p3+epsilon)
			grad.Data[baseIndex+j+2] = p3 - t3

			// Unroll 4
			p4 := probs.Data[baseIndex+j+3]
			t4 := smoothValue
			if j+3 == targetID {
				t4 = targetConfidence
			}
			smoothedLoss -= t4 * math.Log(p4+epsilon)
			grad.Data[baseIndex+j+3] = p4 - t4

			// Unroll 5
			p5 := probs.Data[baseIndex+j+4]
			t5 := smoothValue
			if j+4 == targetID {
				t5 = targetConfidence
			}
			smoothedLoss -= t5 * math.Log(p5+epsilon)
			grad.Data[baseIndex+j+4] = p5 - t5

			// Unroll 6
			p6 := probs.Data[baseIndex+j+5]
			t6 := smoothValue
			if j+5 == targetID {
				t6 = targetConfidence
			}
			smoothedLoss -= t6 * math.Log(p6+epsilon)
			grad.Data[baseIndex+j+5] = p6 - t6

			// Unroll 7
			p7 := probs.Data[baseIndex+j+6]
			t7 := smoothValue
			if j+6 == targetID {
				t7 = targetConfidence
			}
			smoothedLoss -= t7 * math.Log(p7+epsilon)
			grad.Data[baseIndex+j+6] = p7 - t7

			// Unroll 8
			p8 := probs.Data[baseIndex+j+7]
			t8 := smoothValue
			if j+7 == targetID {
				t8 = targetConfidence
			}
			smoothedLoss -= t8 * math.Log(p8+epsilon)
			grad.Data[baseIndex+j+7] = p8 - t8
		}

		// Handle remaining elements
		for ; j < numClasses; j++ {
			p := probs.Data[baseIndex+j]
			var targetProb float64
			if j == targetID {
				targetProb = targetConfidence
			} else {
				targetProb = smoothValue
			}
			smoothedLoss -= targetProb * math.Log(p+epsilon)
			grad.Data[baseIndex+j] = p - targetProb
		}
		loss += smoothedLoss
	}

	if activeTokens > 0 {
		loss /= float64(activeTokens)
		for i := range grad.Data {
			grad.Data[i] /= float64(activeTokens)
		}
	}

	// Reshape grad back to original shape if it was reshaped
	if len(originalShape) == 3 {
		var err error
		grad, err = grad.Reshape(originalShape)
		if err != nil {
			panic(fmt.Sprintf("Failed to reshape gradient: %v", err))
		}
	}

	return loss, grad
}
