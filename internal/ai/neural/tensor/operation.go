package tensor

import (
	"fmt"
	"math"
	"runtime"
	"sync"
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
	numRows := len(tensor.Data) / lastDim
	output := NewTensor(shape, make([]float32, len(tensor.Data)), false)

	numWorkers := runtime.NumCPU()
	if numWorkers > 8 {
		numWorkers = 8
	}
	rowsPerWorker := (numRows + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * rowsPerWorker
		end := (w + 1) * rowsPerWorker
		if start >= numRows {
			break
		}
		if end > numRows {
			end = numRows
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				offset := i * lastDim
				maxVal := float32(math.Inf(-1))
				for j := 0; j < lastDim; j++ {
					if tensor.Data[offset+j] > maxVal {
						maxVal = tensor.Data[offset+j]
					}
				}

				var sumExp float64
				for j := 0; j < lastDim; j++ {
					sumExp += math.Exp(float64(tensor.Data[offset+j] - maxVal))
				}

				for j := 0; j < lastDim; j++ {
					output.Data[offset+j] = float32(math.Exp(float64(tensor.Data[offset+j]-maxVal)) / sumExp)
				}
			}
		}(start, end)
	}
	wg.Wait()

	return output
}

// CrossEntropyLoss calculates the cross-entropy loss with optional label smoothing.
// labelSmoothing: value between 0.0 and 1.0. When > 0, distributes probability mass
// from the target class to all classes to prevent overconfidence.
func CrossEntropyLoss(logits *Tensor, targetIDs []int, padID int, labelSmoothing float32) (float32, *Tensor) {
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
	var loss float64
	activeTokens := 0

	grad := NewTensor(reshapedLogits.Shape, make([]float32, len(reshapedLogits.Data)), false)

	// Calculate smoothed target distribution
	// Exclude padID from sharing smoothing mass
	smoothValue := labelSmoothing / float32(numClasses-1)
	if numClasses <= 1 {
		smoothValue = 0
	}
	targetConfidence := float32(1.0) - labelSmoothing

	numWorkers := runtime.NumCPU()
	if numWorkers > 8 {
		numWorkers = 8
	}
	rowsPerWorker := (reshapedLogits.Shape[0] + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	losses := make([]float64, numWorkers)
	tokenCounts := make([]int, numWorkers)

	for w := 0; w < numWorkers; w++ {
		start := w * rowsPerWorker
		end := (w + 1) * rowsPerWorker
		if start >= reshapedLogits.Shape[0] {
			break
		}
		if end > reshapedLogits.Shape[0] {
			end = reshapedLogits.Shape[0]
		}

		wg.Add(1)
		go func(workerID, s, e int) {
			defer wg.Done()
			var localLoss float64
			localActive := 0
			epsilon := float32(1e-9)

			for i := s; i < e; i++ {
				targetID := targetIDs[i]
				if targetID == padID {
					continue
				}
				localActive++

				baseIndex := i * numClasses
				for j := 0; j < numClasses; j++ {
					p := probs.Data[baseIndex+j]
					var t float32
					if j == targetID {
						t = targetConfidence
					} else if j == padID {
						t = 0
					} else {
						t = smoothValue
					}
					localLoss -= float64(t) * math.Log(float64(p+epsilon))
					grad.Data[baseIndex+j] = p - t
				}
			}
			losses[workerID] = localLoss
			tokenCounts[workerID] = localActive
		}(w, start, end)
	}
	wg.Wait()

	for w := 0; w < numWorkers; w++ {
		loss += losses[w]
		activeTokens += tokenCounts[w]
	}

	if activeTokens > 0 {
		loss /= float64(activeTokens)
		for i := range grad.Data {
			grad.Data[i] /= float32(activeTokens)
		}
	}

	if len(originalShape) == 3 {
		var err error
		grad, err = grad.Reshape(originalShape)
		if err != nil {
			panic(fmt.Sprintf("Failed to reshape gradient: %v", err))
		}
	}

	return float32(loss), grad
}

// ArgMax returns the index and confidence (max value) of the highest probability in the tensor.
func ArgMax(tensor *Tensor) (int, float32) {
	if tensor == nil || len(tensor.Data) == 0 {
		return -1, 0.0
	}

	maxIdx := 0
	maxVal := tensor.Data[0]

	for i, val := range tensor.Data {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx, maxVal
}
