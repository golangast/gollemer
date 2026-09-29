package chat

import "testing"

// Social recall must return trained answers verbatim for exact
// (normalized) matches, and miss on anything else so the neural
// model still handles generalization.
func TestSocialRecall(t *testing.T) {
	initSocialRecall("../../../..")
	if n := SocialRecallSize(); n < 100 {
		t.Fatalf("SocialRecallSize() = %d, want >= 100 indexed pairs", n)
	}
	hits := map[string]string{
		"have you ever played soccer":  "only for fun with friends. i am better at cheering than scoring.",
		"Have you ever played soccer?": "only for fun with friends. i am better at cheering than scoring.",
	}
	for in, want := range hits {
		got, ok := LookupSocialRecall(in)
		if !ok {
			t.Errorf("LookupSocialRecall(%q) missed, want hit", in)
			continue
		}
		if got != want {
			t.Errorf("LookupSocialRecall(%q) = %q, want %q", in, got, want)
		}
	}
	misses := []string{
		"have you ever played rugby", // not trained
		"flibbertigibbet",
		"",
		"???",
		"before you said hi. now you say hi", // history format, not raw input
	}
	for _, in := range misses {
		if out, ok := LookupSocialRecall(in); ok {
			t.Errorf("LookupSocialRecall(%q) hit %q, want miss", in, out)
		}
	}
}
