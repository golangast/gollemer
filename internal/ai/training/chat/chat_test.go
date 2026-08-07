package chat

import (
	"testing"

	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func TestLookupVocabPreservesSpecialHardwareTokens(t *testing.T) {
	vocab := mainvocab.NewVocabulary()
	want := vocab.AddToken("<INTENT_CAMERA_CAPTURE>")

	got := lookupVocab("<INTENT_CAMERA_CAPTURE>", vocab)
	if got != want {
		t.Fatalf("lookupVocab() = %d, want %d", got, want)
	}
}

func TestLoadConversingCSV(t *testing.T) {
	pairs, err := LoadConversingCSV("../../../../data/training/trainingdata/conversing.csv")
	if err != nil {
		t.Fatalf("LoadConversingCSV error: %v", err)
	}
	if len(pairs) == 0 {
		t.Fatalf("LoadConversingCSV returned 0 pairs")
	}
}

func TestLoadConversationCSV_AutoDetectsConversingCSV(t *testing.T) {
	pairs, err := LoadConversationCSV("../../../../data/training/trainingdata/conversing.csv")
	if err != nil {
		t.Fatalf("LoadConversationCSV error: %v", err)
	}
	if len(pairs) == 0 {
		t.Fatalf("LoadConversationCSV on conversing.csv returned 0 pairs")
	}
}
