package seq2seq

import "testing"

// The verbatim decoder's surface mechanics: tokenization keeps
// the user's case while dropping exactly what NormalizeQuery
// drops, and the word choice substitutes the input's own word
// precisely when copying carried the step.

func TestSurfaceTokensKeepCaseAndShape(t *testing.T) {
	got := VerbatimTokens(`Add a Jim in Jake, saying "the Results" to jim/main.go.`)
	want := []string{"Add", "a", "Jim", "in", "Jake", "saying", `"`, "the", "Results", `"`, "to", "jim/main.go"}
	if len(got) != len(want) {
		t.Fatalf("VerbatimTokens = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("VerbatimTokens[%d] = %q, want %q (all: %v)", i, got[i], want[i], got)
		}
	}
}

func TestVerbatimWordSubstitution(t *testing.T) {
	wordOf := func(id int) string {
		switch id {
		case 1:
			return "UNK"
		case 7:
			return "jim"
		case 9:
			return "add"
		}
		return "?"
	}
	surface := []string{"Add", "a", "Jim", "in", "Jake"}
	inputIDs := []int{9, 3, 7, 4, 8}

	// UNK winner: the most-attended position's surface word.
	c := &ptrGenCache{attn: []float32{0.05, 0.05, 0.8, 0.05, 0.05}}
	if w := verbatimWord(1, c, inputIDs, surface, wordOf, 1); w != "Jim" {
		t.Errorf("UNK step: got %q, want Jim", w)
	}
	// Copy-dominated known token: surface case restored.
	c2 := &ptrGenCache{
		attn:     []float32{0.1, 0.1, 0.7, 0.05, 0.05},
		pGen:     0.2,
		copyDist: []float32{0, 0, 0, 0, 0, 0, 0, 0.9, 0, 0},
		pVocab:   []float32{0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0},
	}
	if w := verbatimWord(7, c2, inputIDs, surface, wordOf, 1); w != "Jim" {
		t.Errorf("copy-dominated step: got %q, want Jim", w)
	}
	// Generation-dominated known token: vocabulary word stands.
	c3 := &ptrGenCache{
		attn:     []float32{0.1, 0.1, 0.7, 0.05, 0.05},
		pGen:     0.95,
		copyDist: []float32{0, 0, 0, 0, 0, 0, 0, 0.9, 0, 0},
		pVocab:   []float32{0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0},
	}
	if w := verbatimWord(7, c3, inputIDs, surface, wordOf, 1); w != "jim" {
		t.Errorf("generation-dominated step: got %q, want jim", w)
	}
}
