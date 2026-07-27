package errors

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSynthesizerNeuralNetwork(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "gollemer_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	prompt := "Create a feedforward neural network with backpropagation from scratch"
	fileName := "neural.go"

	// HandleGenerativePrompt returns a single error value
	err = HandleGenerativePrompt(prompt, tempDir, fileName)
	if err != nil {
		t.Fatalf("generative prompt failed: %v", err)
	}

	generatedFilePath := filepath.Join(tempDir, fileName)
	if _, err := os.Stat(generatedFilePath); os.IsNotExist(err) {
		t.Errorf("expected generated file %s to exist", generatedFilePath)
	}

	t.Logf("Successfully synthesized and compiled code!")
}
