package main

import (
	"fmt"
	"log"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func main() {
	// Create a 3D input [2, 3, 4] (batch, seq, hidden)
	inputData := make([]float32, 2*3*4)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}
	input := tensor.NewTensor([]int{2, 3, 4}, inputData, true)

	// Create a Linear layer [4, 5] (hidden, output)
	l, err := nn.NewLinear(4, 5)
	if err != nil {
		log.Fatalf("Failed to create linear layer: %v", err)
	}
	
	// Initialize weights to 1.0 for simplicity
	for i := range l.Weights.Data {
		l.Weights.Data[i] = 1.0
	}
	l.Weights.RequiresGrad = true

	// Forward pass
	output, err := l.Forward(input)
	if err != nil {
		log.Fatalf("Forward failed: %v", err)
	}
	fmt.Printf("Output shape: %v\n", output.Shape)

	// Create a gradient tensor [2, 3, 5]
	gradData := make([]float32, 2*3*5)
	for i := range gradData {
		gradData[i] = 1.0
	}
	grad := tensor.NewTensor([]int{2, 3, 5}, gradData, false)

	// Backward pass
	err = l.Backward(grad)
	if err != nil {
		log.Fatalf("Backward failed: %v", err)
	}

	// Check weight gradients
	fmt.Printf("Weight Grad Data (first 10): %v\n", l.Weights.Grad.Data[:10])
	
	sum := float32(0)
	for _, v := range l.Weights.Grad.Data {
		sum += v
	}
	fmt.Printf("Weight Grad Sum: %.4f\n", sum)
}
