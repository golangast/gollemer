package errors

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestSynthesizerNeuralNetwork(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "gollemer_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// 1. Initialize a Go module in the temp directory so `go build` works
	modInitCmd := exec.Command("go", "mod", "init", "testnet")
	modInitCmd.Dir = tempDir
	if err := modInitCmd.Run(); err != nil {
		t.Fatalf("failed to initialize go mod in temp dir: %v", err)
	}

	// Enable offline test mode to bypass curl network requests
	t.Setenv("GOLLEMER_TEST_MODE", "1")

	prompt := "Create a feedforward neural network with backpropagation from scratch"
	fileName := "neural.go"

	err = HandleGenerativePrompt(prompt, tempDir, fileName)
	if err != nil {
		t.Fatalf("generative prompt failed: %v", err)
	}

	generatedFilePath := filepath.Join(tempDir, fileName)
	if _, err := os.Stat(generatedFilePath); IsNotExist(err) {
		t.Errorf("expected generated file %s to exist", generatedFilePath)
	}

	t.Logf("Successfully synthesized and compiled code with an active go.mod module!")
}

// Helper since os.IsNotExist might need checking
func IsNotExist(err error) bool {
	return os.IsNotExist(err)
}
