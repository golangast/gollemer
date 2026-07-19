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
