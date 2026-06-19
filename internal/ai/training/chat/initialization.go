package chat

import (
	"math"
	"math/rand"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func InitializeXavier(p *tensor.Tensor) {
	if p == nil || len(p.Shape) == 0 {
		return
	}
	fanIn := p.Shape[0]
	fanOut := 0
	if len(p.Shape) > 1 {
		fanOut = p.Shape[1]
	} else {
		fanOut = fanIn
	}
	limit := float32(math.Sqrt(6.0 / float64(fanIn+fanOut)))
	for i := range p.Data {
		p.Data[i] = (float32(rand.Float64()) * 2 * limit) - limit
	}
}

func InitializeHeNormal(p *tensor.Tensor) {
	if p == nil || len(p.Shape) < 2 {
		// Skip biases, LayerNorm params, and scalars.
		// They are usually initialized to 0 or 1 elsewhere.
		return
	}
	// He initialization uses fan-in (input dimension).
	// Most layers in this repo use [inputDim, outputDim] format.
	fanIn := float64(p.Shape[0])
	if fanIn == 0 {
		fanIn = 1
	}
	scale := float32(math.Sqrt(2.0 / fanIn))
	for i := range p.Data {
		p.Data[i] = float32(rand.NormFloat64()) * scale
	}
}

func InitializeRouterGating(weights, biases *tensor.Tensor) {
	if weights == nil {
		return
	}
	scale := float32(0.5)
	for i := range weights.Data {
		weights.Data[i] = (float32(rand.Float64())*2.0 - 1.0) * scale
	}
	if biases != nil {
		for i := range biases.Data {
			if i == 3 && len(biases.Data) > 3 {
				biases.Data[i] = -0.2
			} else {
				biases.Data[i] = 0.1
			}
		}
	}
}

func InitializeOrthogonal(param *tensor.Tensor, gain float32) {
	if param == nil || len(param.Shape) < 2 {
		InitializeXavier(param)
		return
	}

	// 1. Fill with random normal distribution
	for i := range param.Data {
		param.Data[i] = float32(rand.NormFloat64())
	}

	rows := param.Shape[0]
	cols := len(param.Data) / rows
	if cols == 0 {
		return
	}

	// 2. Gram-Schmidt orthogonalization
	for i := 0; i < rows; i++ {
		rowI := param.Data[i*cols : (i+1)*cols]
		for j := 0; j < i; j++ {
			rowJ := param.Data[j*cols : (j+1)*cols]
			// Compute dot product of rowI and rowJ
			dot := 0.0
			for k := range rowI {
				dot += float64(rowI[k] * rowJ[k])
			}
			// Subtract projection: rowI -= dot * rowJ
			for k := range rowI {
				rowI[k] -= float32(dot) * rowJ[k]
			}
		}
		// Normalize the row
		norm := 0.0
		for _, v := range rowI {
			norm += float64(v * v)
		}
		norm = math.Sqrt(norm + 1e-8)
		for k := range rowI {
			rowI[k] = (rowI[k] / float32(norm)) * gain
		}
	}
}

func InitializeLSTMBias(param *tensor.Tensor, hiddenSize int) {
	if param == nil {
		return
	}
	// Zero everything first
	for i := range param.Data {
		param.Data[i] = 0
	}
	// Set Forget Gate bias to 1.0  the 2nd gate in the [f, i, c, o] ordering
	forgetStart := hiddenSize
	forgetEnd := 2 * hiddenSize
	if forgetEnd > len(param.Data) {
		forgetEnd = len(param.Data)
	}
	for i := forgetStart; i < forgetEnd; i++ {
		param.Data[i] = 1.0
	}
	// For BiLSTM: if bias is the backward pass (second half), set that forget gate too
	if len(param.Data) == 8*hiddenSize {
		backwardForgetStart := 5 * hiddenSize
		backwardForgetEnd := 6 * hiddenSize
		for i := backwardForgetStart; i < backwardForgetEnd; i++ {
			param.Data[i] = 1.0
		}
	}
}

func isLSTMWeight(param *tensor.Tensor) bool {
	if param == nil || len(param.Shape) != 2 {
		return false
	}
	// Heuristic: LSTM weight matrices are square or have a larger first dimension
	// (inputSize + hiddenSize > hiddenSize). They are never 1-row (bias) tensors.
	return param.Shape[0] > 1 && param.Shape[1] > 1
}

func isLSTMBias(param *tensor.Tensor, hiddenSize int) bool {
	if param == nil {
		return false
	}
	size := len(param.Data)
	// LSTM bias is exactly hiddenSize elements (one bias per gate, 4 gates total if stacked)
	// or hiddenSize for individual gate biases (Bf/Bi/Bc/Bo each [1, hiddenSize]).
	return size == hiddenSize || size == 4*hiddenSize || size == 8*hiddenSize
}
