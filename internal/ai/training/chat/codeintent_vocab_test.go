package chat

// Regression test for the codeintent -resume failure of 2026-10-04: the
// frequency-capped vocabulary was built by sorting tokens by frequency alone
// over randomized Go map iteration order, so every process built a different
// token ordering. The checkpoint fingerprint hashes that ordering, so
// -resume always refused the checkpoint with "data changed" and a killed run
// could never resume. buildCappedVocab must be fully deterministic.

import (
	"fmt"
	"testing"
)

// TestBuildCappedVocabDeterministic builds the capped vocab repeatedly from
// pair data engineered to have massive frequency ties (the case that broke
// the old frequency-only sort) and requires byte-identical token order every
// time — which is what makes vocabFingerprint stable across processes.
func TestBuildCappedVocabDeterministic(t *testing.T) {
	var pairs []ChatPair
	// 500 distinct tokens, each appearing exactly 3 times: every token ties.
	for i := 0; i < 500; i++ {
		tok := fmt.Sprintf("ident_%03d", i)
		pairs = append(pairs,
			ChatPair{Input: "what does " + tok + " do", Output: tok + " handles it"},
			ChatPair{Input: "why call " + tok, Output: "because " + tok},
		)
	}
	// A few high-frequency tokens to exercise the frequency-descending part.
	for i := 0; i < 50; i++ {
		pairs = append(pairs, ChatPair{Input: "what does func do", Output: "func handles it"})
	}

	first := buildCappedVocab(pairs, 8000)
	if first.Size() == 0 {
		t.Fatal("expected non-empty vocab")
	}
	for run := 0; run < 50; run++ {
		v := buildCappedVocab(pairs, 8000)
		if v.Size() != first.Size() {
			t.Fatalf("run %d: vocab size %d != first run %d", run, v.Size(), first.Size())
		}
		for i := 0; i < v.Size(); i++ {
			if v.GetWord(i) != first.GetWord(i) {
				t.Fatalf("run %d: token %d = %q, want %q (first run)",
					run, i, v.GetWord(i), first.GetWord(i))
			}
		}
		if vocabFingerprint(v) != vocabFingerprint(first) {
			t.Fatalf("run %d: fingerprint mismatch", run)
		}
	}
	// Sanity: frequency-descending with alphabetical tie-break.
	// "do" and "does" each appear 550 times (the tie), "func" 100 times.
	// (IDs 0-3 are the pre-seeded specials: <pad>, <unk>, <s>, </s>.)
	if got := first.GetWord(4); got != "do" {
		t.Fatalf("expected %q at index 4, got %q", "do", got)
	}
	if got := first.GetWord(5); got != "does" {
		t.Fatalf("expected %q at index 5 (tie broken alphabetically), got %q", "does", got)
	}
}

// TestBuildCappedVocabCapRespected ensures the cap bounds the vocab size.
func TestBuildCappedVocabCapRespected(t *testing.T) {
	var pairs []ChatPair
	for i := 0; i < 300; i++ {
		tok := fmt.Sprintf("tok_%03d", i)
		pairs = append(pairs, ChatPair{Input: tok, Output: tok})
	}
	v := buildCappedVocab(pairs, 100)
	// 100 content tokens plus the 4 pre-seeded specials (<pad>, <unk>, <s>, </s>).
	if v.Size() != 104 {
		t.Fatalf("expected vocab size 104, got %d", v.Size())
	}
}
