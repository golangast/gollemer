package chat

import "testing"

func TestSmallTrainingPromptSet(t *testing.T) {
	prompts := SmallTestPrompts()
	if len(prompts) == 0 {
		t.Fatal("expected non-empty prompt set for tiny dataset LLM check")
	}
	if got := prompts[0]; got == "" {
		t.Fatal("expected first prompt to be non-empty")
	}
	for _, p := range prompts {
		if p == "" {
			t.Fatal("expected no empty prompt in tiny-data validation set")
		}
	}
}
