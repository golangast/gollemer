package moe

import (
	"fmt"
	"math"
	"github.com/golangast/gollemer/neural/tensor"
)

// VerifyGradients performs numerical gradient checking on a model.
// It compares the analytical gradients from Backward() with numerical approximations.
func VerifyGradients(m *IntentMoE, inputIDs, targetIDs *tensor.Tensor) {
	epsilon := 1e-4
	tolerance := 1e-2

	fmt.Println("🧪 --- Numerical Gradient Checker ---")

	// 1. Get analytical gradient from Backward pass
	// (Assumes Forward and Backward have already been implemented correctly)
	logits, _, err := m.Forward(0.0, inputIDs, targetIDs)
	if err != nil {
		fmt.Printf("❌ Forward failed: %v\n", err)
		return
	}
	
	// Create dummy loss and gradient for testing
	// In a real check, we'd use the actual training loss function
	const stepIdx = 0
	targetLogit := logits[stepIdx]
	
	// Simplified analytical pass
	m.Backward(targetLogit) // Just pass the output as grad for simplicity of check-setup
	
	// Pick a parameter to check: e.g., the first expert's weights in Layer 0
	var testParam *tensor.Tensor
	if stack, ok := m.Encoder.(*HybridLLMGNNEncoder).LLMEncoder.(*MoEStack); ok {
		if len(stack.Layers) > 0 {
			l0 := stack.Layers[0]
			if len(l0.Experts) > 0 {
				if fe, ok := l0.Experts[0].(*FeedForwardExpert); ok {
					testParam = fe.Layer1.Weights
				}
			}
		}
	}

	if testParam == nil || testParam.Grad == nil {
		fmt.Println("⚠️  Could not find target parameter or its gradient for verification.")
		return
	}

	simdGrad := testParam.Grad.Data[0]
	originalWeight := testParam.Data[0]

	// 2. Get numerical gradient: (L(w+e) - L(w-e)) / 2e
	// Note: For this to work perfectly, we must use the same loss function.
	// Here we use Sum(Output) as a simple 'loss' for verification of the graph.
	
	fetchLoss := func() float64 {
		l, _, _ := m.Forward(0.0, inputIDs, targetIDs)
		sum := 0.0
		for _, v := range l[stepIdx].Data {
			sum += v
		}
		return sum
	}

	testParam.Data[0] = originalWeight + epsilon
	lossPlus := fetchLoss()

	testParam.Data[0] = originalWeight - epsilon
	lossMinus := fetchLoss()

	numGrad := (lossPlus - lossMinus) / (2 * epsilon)

	// 3. Compare
	diff := math.Abs(simdGrad - numGrad)
	if diff > tolerance {
		fmt.Printf("❌ GRADIENT MISMATCH: SIMD: %f, Numerical: %f, Diff: %f\n", simdGrad, numGrad, diff)
	} else {
		fmt.Printf("✅ GRADIENT MATCH: Diff: %e\n", diff)
	}

	// Reset weight!
	testParam.Data[0] = originalWeight
}
