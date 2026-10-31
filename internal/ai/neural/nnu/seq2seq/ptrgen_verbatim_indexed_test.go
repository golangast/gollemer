package seq2seq

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// TestEncodeIndexedInputDisambiguatesUnknowns pins the arm-A
// mechanism: two distinct out-of-vocabulary words in one input
// get two DIFFERENT placeholder ids, ranked by first
// appearance, and the restore map carries each one's own
// surface spelling. A repeated unknown reuses its index.
func TestEncodeIndexedInputDisambiguatesUnknowns(t *testing.T) {
	v := vocab.NewVocabulary()
	for _, tok := range []string{"add", "a", "struct", "to", "handler", "in", "the"} {
		v.AddToken(tok)
	}
	for k := 1; k <= MaxIndexedUnk; k++ {
		v.AddToken(IndexedUnkToken(k))
	}
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewSeq2Seq(v.Size(), v.Size(), 8, 16, tok, v)
	if err != nil {
		t.Fatal(err)
	}

	enc := m.EncodeIndexedInput("add a struct Tend to handler Sift in Tend")
	unk1 := v.GetTokenID(IndexedUnkToken(1))
	unk2 := v.GetTokenID(IndexedUnkToken(2))
	if enc.Restore[1] != "Tend" || enc.Restore[2] != "Sift" {
		t.Fatalf("restore map = %v, want Tend/Sift by first appearance", enc.Restore)
	}
	var got []int
	for _, id := range enc.IDs {
		if id == unk1 || id == unk2 {
			got = append(got, id)
		}
	}
	want := []int{unk1, unk2, unk1} // Tend, Sift, Tend again
	if len(got) != len(want) {
		t.Fatalf("placeholder ids = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("placeholder ids = %v, want %v", got, want)
		}
	}
	if unk1 == unk2 || unk1 == v.UnkID || unk2 == v.UnkID {
		t.Fatalf("placeholder ids must differ from each other and from UNK: %d %d unk=%d", unk1, unk2, v.UnkID)
	}
	// Known words keep their real ids.
	if enc.IDs[0] != v.GetTokenID("add") {
		t.Fatalf("known word lost its id: %v", enc.IDs)
	}
	// Case: the surface keeps the user's spelling.
	if !strings.Contains(strings.Join(enc.Surface, " "), "Sift") {
		t.Fatalf("surface lost case: %v", enc.Surface)
	}
}

// TestEncodeIndexedInputCapsPlaceholders: beyond
// MaxIndexedUnk distinct unknowns, words fall back to plain
// UNK rather than inventing a ninth placeholder.
func TestEncodeIndexedInputCapsPlaceholders(t *testing.T) {
	v := vocab.NewVocabulary()
	v.AddToken("add")
	for k := 1; k <= MaxIndexedUnk; k++ {
		v.AddToken(IndexedUnkToken(k))
	}
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewSeq2Seq(v.Size(), v.Size(), 8, 16, tok, v)
	if err != nil {
		t.Fatal(err)
	}
	enc := m.EncodeIndexedInput("add Aa Bb Cc Dd Ee Ff Gg Hh Ii Jj")
	if len(enc.Restore) != MaxIndexedUnk {
		t.Fatalf("restore holds %d entries, want %d", len(enc.Restore), MaxIndexedUnk)
	}
	plain := 0
	for _, id := range enc.IDs {
		if id == v.UnkID {
			plain++
		}
	}
	if plain != 2 { // Ii and Jj overflow
		t.Fatalf("plain UNK count = %d, want 2 (ids %v)", plain, enc.IDs)
	}
}
