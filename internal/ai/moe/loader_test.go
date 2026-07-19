package moe

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"os"
	"testing"
)

// TestLoadIntentMoEModelWithFallback demonstrates the fallback loader works with both formats
func TestLoadIntentMoEModelWithFallback(t *testing.T) {
	// Create a minimal test model
	testModel := &IntentMoE{
		EmbeddingDim:      256,
		SentenceVocabSize: 50,
	}

	// Test 1: Write and read gzip-compressed checkpoint format
	t.Run("GzipCheckpointFormat", func(t *testing.T) {
		tmpFile := "/tmp/test_gzip_checkpoint.gob"
		defer os.Remove(tmpFile)

		// Write as gzip checkpoint
		file, err := os.Create(tmpFile)
		if err != nil {
			t.Fatalf("Failed to create test file: %v", err)
		}
		defer file.Close()

		gz := gzip.NewWriter(file)
		encoder := gob.NewEncoder(gz)
		checkpoint := &Checkpoint{
			Model:     testModel,
			StepCount: 100,
		}
		if err := encoder.Encode(checkpoint); err != nil {
			t.Fatalf("Failed to encode checkpoint: %v", err)
		}
		gz.Close()
		file.Close()

		// Read with fallback loader
		loaded, err := LoadIntentMoEModelWithFallback(tmpFile)
		if err != nil {
			t.Fatalf("Failed to load gzip checkpoint: %v", err)
		}
		if loaded.EmbeddingDim != 256 {
			t.Errorf("Expected EmbeddingDim=256, got %d", loaded.EmbeddingDim)
		}
	})

	// Test 2: Write and read raw gob format
	t.Run("RawGobLegacyFormat", func(t *testing.T) {
		tmpFile := "/tmp/test_raw_gob.gob"
		defer os.Remove(tmpFile)

		// Write as raw gob
		file, err := os.Create(tmpFile)
		if err != nil {
			t.Fatalf("Failed to create test file: %v", err)
		}
		defer file.Close()

		encoder := gob.NewEncoder(file)
		if err := encoder.Encode(testModel); err != nil {
			t.Fatalf("Failed to encode model: %v", err)
		}
		file.Close()

		// Read with fallback loader
		loaded, err := LoadIntentMoEModelWithFallback(tmpFile)
		if err != nil {
			t.Fatalf("Failed to load raw gob model: %v", err)
		}
		if loaded.EmbeddingDim != 256 {
			t.Errorf("Expected EmbeddingDim=256, got %d", loaded.EmbeddingDim)
		}
	})

	// Test 3: Empty file rejection
	t.Run("EmptyFileRejection", func(t *testing.T) {
		tmpFile := "/tmp/test_empty.gob"
		defer os.Remove(tmpFile)

		// Create empty file
		file, _ := os.Create(tmpFile)
		file.Close()

		// Should fail with empty file error
		_, err := LoadIntentMoEModelWithFallback(tmpFile)
		if err == nil {
			t.Fatal("Expected error for empty file, got nil")
		}
		if !bytes.Contains([]byte(err.Error()), []byte("empty")) {
			t.Errorf("Expected 'empty' in error message, got: %v", err)
		}
	})
}
