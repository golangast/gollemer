package chat

import (
	"path/filepath"
	"testing"
)

// The intent layer's verdict, kept as a regression gate: the
// shipped model must understand handwritten beginner phrasings it
// never trained on. If a future corpus/model change drops this
// below the bar, the front door got dumber — fail loudly.
func TestIntentModelHeldOut(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..")
	m, err := LoadIntentModel(filepath.Join(root, "data", "intent_model.json"))
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	held, err := LoadIntentPairs(filepath.Join(root, "data", "training", "intent_heldout.jsonl"))
	if err != nil {
		t.Fatalf("load held-out: %v", err)
	}
	ok := 0
	for _, p := range held {
		if got, _, _ := m.Predict(p.Input); got == p.Intent {
			ok++
		} else {
			t.Logf("miss: want %s, got %s, %q", p.Intent, got, p.Input)
		}
	}
	acc := float64(ok) / float64(len(held))
	t.Logf("held-out accuracy: %d/%d = %.1f%%", ok, len(held), 100*acc)
	if acc < 0.90 {
		t.Fatalf("held-out accuracy %.1f%% is below the 90%% bar", 100*acc)
	}
}

// Brain-owned lines must NOT be dispatched by the intent layer:
// they belong to the router and the neural brains.
func TestIntentDispatchLeavesBrainsAlone(t *testing.T) {
	for _, line := range []string{
		"what is a goroutine",
		"write a function that reverses a string",
		"how was your day",
		"how do i retrain the model",
	} {
		if tryIntentDispatch(line, nil) {
			t.Errorf("intent layer should not claim %q", line)
		}
	}
}
