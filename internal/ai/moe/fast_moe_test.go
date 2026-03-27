package moe

import (
	"testing"
)

// Mock Layers for Benchmarking
type DenseLayer struct {
	Weights []float32
	Bias    []float32
}

func NewDenseLayer(inputSize, outputSize int) *DenseLayer {
	return &DenseLayer{
		Weights: make([]float32, inputSize*outputSize),
		Bias:    make([]float32, outputSize),
	}
}

func (d *DenseLayer) Forward(input []float32) []float32 {
	outputSize := len(d.Bias)
	inputSize := len(input)
	res := make([]float32, outputSize)
	for i := 0; i < outputSize; i++ {
		dot := float32(0)
		offset := i * inputSize
		for j := 0; j < inputSize; j++ {
			dot += d.Weights[offset+j] * input[j]
		}
		res[i] = dot + d.Bias[i]
	}
	return res
}

func TestLoadBalancingLoss(t *testing.T) {
	numExperts := 4
	hiddenSize := 512
	moe := NewMoELayerFast(numExperts, hiddenSize, hiddenSize, 2)
	moe.AuxLossWeight = 0.01

	// Scenario A: Perfectly Unbalanced (All tokens to Expert 0)
	countsUnbalanced := []int64{8, 0, 0, 0}
	probsUnbalanced := []float32{0.9, 0.03, 0.03, 0.04}
	
	lossHigh := moe.CalculateAuxLoss(8, countsUnbalanced, probsUnbalanced)

	// Scenario B: Perfectly Balanced (2 tokens per expert)
	countsBalanced := []int64{2, 2, 2, 2}
	probsBalanced := []float32{0.25, 0.25, 0.25, 0.25}
	
	lossLow := moe.CalculateAuxLoss(8, countsBalanced, probsBalanced)

	if lossLow >= lossHigh {
		t.Errorf("Expected balanced loss (%f) to be lower than unbalanced loss (%f)", lossLow, lossHigh)
	}
}

func BenchmarkLayerThroughput(b *testing.B) {
	hiddenSize := 512
	numExperts := 8
	input := make([]float32, hiddenSize) // Mock input token

	// Initialize Layers
	dense := NewDenseLayer(hiddenSize, hiddenSize*numExperts)
	moe := NewMoELayerFast(numExperts, hiddenSize, hiddenSize, 2)

	b.Run("Dense-Layer", func(b *testing.B) {
		b.SetBytes(int64(hiddenSize * 4)) // 4 bytes per float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = dense.Forward(input)
		}
	})

	b.Run("Sparse-MoE", func(b *testing.B) {
		b.SetBytes(int64(hiddenSize * 4)) // 4 bytes per float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = moe.Forward(input)
		}
	})
}
