package planner

import (
	"context"
	"fmt"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// GollemerNativeEngine bridges the LLMEngine interface to Gollemer native MoE inference.
type GollemerNativeEngine struct {
	moeStack *moe.MoEStack
	tok      *tokenizer.Tokenizer
}

// NewGollemerNativeEngine creates a bridge between the planner LLM interface and Gollemer native MoE.
func NewGollemerNativeEngine(stack *moe.MoEStack, tok *tokenizer.Tokenizer) *GollemerNativeEngine {
	return &GollemerNativeEngine{
		moeStack: stack,
		tok:      tok,
	}
}

// Generate implements LLMEngine by tokenizing prompt, running MoE forward pass, and decoding output.
func (e *GollemerNativeEngine) Generate(ctx context.Context, prompt string) (string, error) {
	if e.moeStack == nil || e.tok == nil {
		return "", fmt.Errorf("native gollemer model uninitialized")
	}

	inputIDs, err := e.tok.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("encoding failed: %w", err)
	}

	tensorInput := inputIDsToTensor(inputIDs)
	outputTensor, err := e.moeStack.Forward(tensorInput)
	if err != nil {
		return "", fmt.Errorf("forward pass failed: %w", err)
	}

	outputIDs := tensorToOutputIDs(outputTensor)
	resultStr, err := e.tok.Decode(outputIDs)
	if err != nil {
		return "", fmt.Errorf("decoding failed: %w", err)
	}

	return resultStr, nil
}
func inputIDsToTensor(ids []int) *tensor.Tensor {
	floatData := make([]float32, len(ids))
	for i, id := range ids {
		floatData[i] = float32(id)
	}
	return tensor.NewTensor([]int{1, len(ids)}, floatData, false)
}

// tensorToOutputIDs extracts integer token IDs from output logits/tensors.
// tensorToOutputIDs extracts integer token IDs from output logits/tensors.
func tensorToOutputIDs(t *tensor.Tensor) []int {
	if t == nil {
		return nil
	}
	// Direct field access without ()
	data := t.Data
	ids := make([]int, len(data))
	for i, val := range data {
		ids[i] = int(val)
	}
	return ids
}

func tensorToIDs(t interface{}) []int {
	return nil
}
