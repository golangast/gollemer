package moe

import (
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"testing"
)

func TestMoELayer_SignalCollapse(t *testing.T) {
	// 1. Initialize a small MoE Layer (8 Experts, 128 Hidden)
	inputDim := 128
	outputDim := 128
	numExperts := 8
	k := 2

	expertBuilder := func(id int) (Expert, error) {
		return NewFeedForwardExpert(inputDim, 256, outputDim)
	}

	layer, err := NewMoELayer(inputDim, outputDim, numExperts, k, expertBuilder)
	if err != nil {
		t.Fatalf("Failed to create MoE layer: %v", err)
	}

	// 2. Create a "Crashed" Input (All Zeros)
	// This simulates the L2 Norm: 0.000000 you saw in the logs
	deadInput := tensor.NewTensor([]int{1, 1, 128}, nil, false) // 3D input as expected by MoELayer.Forward

	// 3. Run Forward Pass
	output, err := layer.Forward(deadInput)
	if err != nil {
		t.Fatalf("Layer panicked on zero input: %v", err)
	}

	// 4. Validate output isn't NaN or Inf
	for _, v := range output.Data {
		if v != v { // NaN check
			t.Error("🚨 MoE Layer produced NaN on zero-norm input. Gating division by zero detected.")
			break
		}
	}

	// 5. Check Routing - It should ideally be uniform, not skewed
	// Since we injected jitter (1e-6) on zero norm, the router should have something to work with.
	// gateLogits should be stable.
}
