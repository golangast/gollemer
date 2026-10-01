package chat

import (
	"strings"
	"testing"
)

func TestRouteDomainBeginner(t *testing.T) {
	for _, in := range []string{
		"beginner: create an http server",
		"xray a worker pool",
		"explain channels like I'm a beginner",
		"eli5 goroutines",
		"explain simply what a slice is",
		"in simple terms, what is a mutex",
		"show me a worker pool for beginners",
	} {
		if got := routeDomain(in); got != BeginnerDomain {
			t.Errorf("routeDomain(%q) = %q, want beginner", in, got)
		}
	}
	// Ordinary requests must NOT be stolen by the beginner brain.
	stillOther := map[string]string{
		"write a function to sort a slice":      GocodeDomain,
		"what is a channel in go":               GoDomain,
		"create an http server":                 SocialDomain, // unmarked: unchanged existing behavior
		"how do i train the model":              MakefileDomain,
		"analyze this project":                  GoAnalyzeDomain,
		"what does go mod tidy do":              GoDomain,
		"tell me a joke":                        SocialDomain,
		"check if the dependencies are updated": GoCliDomain,
	}
	for in, want := range stillOther {
		if got := routeDomain(in); got != want {
			t.Errorf("routeDomain(%q) = %q, want %q (beginner stole it?)", in, got, want)
		}
	}
}

func TestHandleBeginnerExplain(t *testing.T) {
	out, ok := handleBeginner("beginner: explain channels")
	if !ok {
		t.Fatal("handleBeginner did not answer")
	}
	if !strings.Contains(out, "conveyor belt") {
		t.Errorf("missing analogy: %q", out)
	}
	if !strings.Contains(out, "```go") {
		t.Errorf("missing code fence: %q", out)
	}
}

func TestHandleBeginnerGenerate(t *testing.T) {
	out, ok := handleBeginner("xray: create a worker pool using channels and WaitGroup")
	if !ok {
		t.Fatal("handleBeginner did not answer")
	}
	if !strings.Contains(out, "sync.WaitGroup") {
		t.Errorf("missing generated code: %q", out)
	}
	if !strings.Contains(out, "Why this code:") {
		t.Errorf("missing explanation: %q", out)
	}
}

func TestHandleBeginnerUnknown(t *testing.T) {
	out, ok := handleBeginner("beginner: bake a cake")
	if !ok {
		t.Fatal("handleBeginner did not answer")
	}
	if !strings.Contains(out, "couldn't build that") {
		t.Errorf("want graceful failure, got: %q", out)
	}
}
