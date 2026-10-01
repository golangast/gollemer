package memory

import (
	"math"
	"testing"
)

func TestEmbedTextDeterministic(t *testing.T) {
	a := EmbedText("func Concat(words []string) string", 64)
	b := EmbedText("func Concat(words []string) string", 64)
	if len(a) != 64 || len(b) != 64 {
		t.Fatalf("wrong dim: %d, %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("embedding is not deterministic")
		}
	}
}

func TestEmbedTextUnitNorm(t *testing.T) {
	v := EmbedText("hello world foo bar", 32)
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if math.Abs(math.Sqrt(norm)-1) > 1e-5 {
		t.Fatalf("not unit norm: %f", norm)
	}
}

func TestEmbedTextOverlap(t *testing.T) {
	// Lexical overlap should rank the related text higher.
	query := EmbedText("concatenate words string builder", 64)
	related := EmbedText("func Concat(words []string) string { var s strings.Builder }", 64)
	unrelated := EmbedText("quantum banana telescope", 64)
	if CosineSimilarity(query, related) <= CosineSimilarity(query, unrelated) {
		t.Error("expected related code to score higher than unrelated text")
	}
}

func TestEmbedTextEdgeCases(t *testing.T) {
	if EmbedText("x", 0) != nil {
		t.Error("expected nil for dim 0")
	}
	if EmbedText("x", -1) != nil {
		t.Error("expected nil for negative dim")
	}
	v := EmbedText("", 16)
	for _, x := range v {
		if x != 0 {
			t.Error("expected zero vector for empty text")
		}
	}
}
