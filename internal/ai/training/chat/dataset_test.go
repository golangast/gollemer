package chat

import "testing"

func TestValidatePairAcceptsGoodPair(t *testing.T) {
	seen := map[string]bool{}
	if err := ValidatePair(ChatPair{Input: "hello there", Output: "hi, how are you?"}, seen); err != nil {
		t.Fatalf("good pair rejected: %v", err)
	}
}

func TestValidatePairRejectsBadPairs(t *testing.T) {
	seen := map[string]bool{}
	cases := []ChatPair{
		{Input: "", Output: "hi"},               // empty input
		{Input: "hello", Output: ""},            // empty output
		{Input: "hello", Output: "hello"},       // identical
		{Input: "hi\x00there", Output: "hello"}, // control char
		{Input: "hello", Output: "hi\x07there"}, // control char
		{Input: "word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word", Output: "hi"}, // too long
	}
	for i, p := range cases {
		if err := ValidatePair(p, seen); err == nil {
			t.Fatalf("case %d accepted but should be quarantined: %+v", i, p)
		}
	}
}

func TestValidatePairRejectsDuplicates(t *testing.T) {
	seen := map[string]bool{}
	p := ChatPair{Input: "Hello there", Output: "Hi!"}
	if err := ValidatePair(p, seen); err != nil {
		t.Fatalf("first occurrence rejected: %v", err)
	}
	seen[normalizePairKey(p)] = true
	dup := ChatPair{Input: "  hello   THERE ", Output: "hi!"} // same after normalization
	if err := ValidatePair(dup, seen); err == nil {
		t.Fatalf("normalized duplicate accepted")
	}
}

func TestClassifyDomain(t *testing.T) {
	if d := ClassifyDomain("What is a goroutine?", ""); d != "go" {
		t.Fatalf("go question classified as %q", d)
	}
	if d := ClassifyDomain("How are you doing today?", ""); d != SocialDomain {
		t.Fatalf("social question classified as %q", d)
	}
	// "How did your day go?" is social — the verb "go" must not trigger it.
	if d := ClassifyDomain("How did your day go?", ""); d != SocialDomain {
		t.Fatalf("social 'go' verb classified as %q", d)
	}
}

func TestClassifyDomainGoVariants(t *testing.T) {
	goQs := []string{
		"What is a WaitGroup?", "What is a pointer?", "Why should I use Go?",
		"Where are binary executables saved?", "How does error handling work?",
		"What does new() do?", "How do I handle dependencies?",
	}
	for _, q := range goQs {
		if d := ClassifyDomain(q, ""); d != "go" {
			t.Errorf("%q classified as %q, want go", q, d)
		}
	}
}
