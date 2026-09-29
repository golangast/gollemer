package chat

import "testing"

// Makefile recall must return trained commands verbatim for exact
// (normalized) matches, so similar "how do i ..." inputs can't be
// confused by the tiny generative model.
func TestMakefileRecall(t *testing.T) {
	initMakefileRecall("../../../..")
	hits := map[string]string{
		"show the start here guide":      "run make start",
		"where do i begin with gollemer": "run make start",
		"retrain gocli":                  "run make train-gocli",
		"train the go code model":        "run make train-gocode",
		"list the commands":              "run make help",
		"how do i chat with the model":    "run make chat",
	}
	for in, want := range hits {
		got, ok := LookupMakefileRecall(in)
		if !ok {
			t.Errorf("LookupMakefileRecall(%q) missed, want hit", in)
			continue
		}
		if got != want {
			t.Errorf("LookupMakefileRecall(%q) = %q, want %q", in, got, want)
		}
	}
	// Case/punctuation-insensitive.
	if got, ok := LookupMakefileRecall("Show the start here guide?"); !ok || got != "run make start" {
		t.Errorf("normalized lookup failed: %q %v", got, ok)
	}
	for _, in := range []string{"retrain the flux capacitor", "", "???"} {
		if out, ok := LookupMakefileRecall(in); ok {
			t.Errorf("LookupMakefileRecall(%q) hit %q, want miss", in, out)
		}
	}
}
