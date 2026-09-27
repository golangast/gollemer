package chat

// Collision-aware augmentation: a synonym variant that normalizes to another
// base pair's input must be dropped when the outputs differ.

import (
	"strings"
	"testing"
)

func TestAugmentInputsSkipsCollisions(t *testing.T) {
	base := []ChatPair{
		{Input: "I'm sad", Output: "That's tough. I'm here if you want to chat.", Domain: "social"},
		{Input: "I'm unhappy.", Output: "That's hard. I hope things get better soon.", Domain: "social"},
		{Input: "Good morning", Output: "Morning! Have a great day.", Domain: "social"},
		{Input: "Good afternoon", Output: "Afternoon! How is it going?", Domain: "social"},
	}
	// "sad" -> {"unhappy","down"}: variant "i'm unhappy" collides with the
	// "I'm unhappy." base pair (different output) and must be skipped.
	// "morning" -> {"day"}: variant "good day" collides with the
	// "Good afternoon" base pair and must be skipped.
	got := augmentInputs(base, base, socialSynonyms)
	seen := map[string]string{}
	for _, p := range got {
		n := normAugInput(p.Input)
		if prev, dup := seen[n]; dup && prev != p.Output {
			t.Fatalf("conflicting labels for %q: %q vs %q", p.Input, prev, p.Output)
		}
		seen[n] = p.Output
	}
	for _, p := range got {
		if normAugInput(p.Input) == "i'm unhappy" && p.Output != "That's hard. I hope things get better soon." {
			t.Fatalf("colliding variant kept: %q -> %q", p.Input, p.Output)
		}
	}
	// "morning" -> {"day"}: variant "good day" is claimed by exactly one of
	// "Good morning"/"Good afternoon" — ambiguous phrasings get a single
	// consistent label, never two.
	goodDayOutputs := map[string]bool{}
	for _, p := range got {
		if normAugInput(p.Input) == "good day" {
			goodDayOutputs[p.Output] = true
		}
	}
	if len(goodDayOutputs) > 1 {
		t.Fatalf("\"good day\" has conflicting outputs: %v", goodDayOutputs)
	}
	// Non-colliding variants must still be produced.
	foundDown := false
	for _, p := range got {
		if strings.Contains(normAugInput(p.Input), "i'm down") {
			foundDown = true
		}
	}
	if !foundDown {
		t.Fatal("expected non-colliding variant \"i'm down\" to be kept")
	}
	// Every base pair must survive.
	if len(got) < len(base) {
		t.Fatalf("base pairs lost: got %d pairs from %d base", len(got), len(base))
	}
	t.Logf("augmented %d base -> %d train pairs", len(base), len(got))
}

func TestGocodeSynonymsNeverTouchHello(t *testing.T) {
	// The gocode synonym set must never rewrite greeting words: "write a
	// hello world program" (main) vs "code a hello name function" (greet)
	// is the boundary the model struggles with, and social-style swaps like
	// hello->hi would blur it. Regression test for the round-5 collapse.
	for w, syns := range gocodeSynonyms {
		for _, s := range syns {
			for _, banned := range []string{"hello", "hi", "hey", "greetings"} {
				if w == banned || s == banned {
					t.Fatalf("gocodeSynonyms touches greeting word %q (-> %q)", w, s)
				}
			}
		}
	}
	variants := synonymVariants("write a hello world program in go", 9, gocodeSynonyms)
	for _, v := range variants {
		n := normAugInput(v)
		if !strings.Contains(n, "hello world") {
			t.Fatalf("gocode variant mangled the hello-world cue: %q", v)
		}
	}
	if len(variants) == 0 {
		t.Fatal("expected gocode variants for a write/make/create input")
	}
}
