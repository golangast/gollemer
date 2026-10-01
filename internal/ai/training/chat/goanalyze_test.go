package chat

import (
	"testing"
)

// TestRouteGoAnalyze verifies the new brain's intents route to
// GoAnalyzeDomain without disturbing the other brains.
func TestRouteGoAnalyze(t *testing.T) {
	goanalyze := []string{
		"analyze this go project",
		"analyze the codebase at ~/projects/foo",
		"can you analyze this repo",
		"give me a tour of the codebase",
		"map out the project structure",
		"walk me through this code",
		"how does this codebase work",
		"how is this repo structured",
		"where do I start in this codebase",
		"read the source at ./cmd/tool",
		"analyze https://github.com/foo/bar",
		"where would I add a new chat command",
		"where do I change the router",
		"where should I put the new eval script",
		// Natural path forms: the analyze verb plus a path-shaped token
		// routes even when the path names no codebase noun.
		"analyze ~/projects/foo",
		"analyze ./cmd/tool",
		// Visual requests aimed at the codebase.
		"show me a visual of this project",
		"draw me a diagram of the repo",
	}
	for _, in := range goanalyze {
		if d := routeDomain(in); d != GoAnalyzeDomain {
			t.Errorf("routeDomain(%q) = %q, want goanalyze", in, d)
		}
	}
	// Near-misses that must NOT route to goanalyze (other brains own them).
	keep := map[string]string{
		"how do i use a map in go":           GoDomain, // bare map: Go builtin
		"what does make chat do":             SocialDomain,
		"analyze the training dataset":       MakefileDomain, // dataset work
		"how do i add training data":         SocialDomain,
		"what is gollemer":                   SocialDomain,
		"write a function to parse json":     GocodeDomain,
		"check if dependencies are updated":  GoCliDomain,
		"show me how to draw a graph in go":  GoDomain, // plotting, not the codebase graph
	}
	for in, want := range keep {
		if d := routeDomain(in); d != want {
			t.Errorf("routeDomain(%q) = %q, want %q", in, d, want)
		}
	}
}

func TestExtractAnalyzePath(t *testing.T) {
	unifiedProjectRoot = "/repo/root"
	defer func() { unifiedProjectRoot = "" }()
	cases := []struct{ in, want string }{
		{`analyze the codebase at ~/projects/foo`, "~/projects/foo"},
		{`analyze "/home/john/my app" please`, "/home/john/my app"},
		{"analyze https://github.com/foo/bar", "https://github.com/foo/bar"},
		{"map out github.com/spf13/cobra", "github.com/spf13/cobra"},
		{"read the code in ./cmd/tool", "./cmd/tool"},
		{"analyze this project", "/repo/root"},
		{"where do I start in this repo", "/repo/root"},
	}
	for _, c := range cases {
		if got := extractAnalyzePath(c.in); got != c.want {
			t.Errorf("extractAnalyzePath(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestParseWhereTask(t *testing.T) {
	task, ok := parseWhereTask("where would I add a retry helper in ~/proj")
	if !ok || task != "a retry helper" {
		t.Fatalf("got %q, %v", task, ok)
	}
	task, ok = parseWhereTask("where do I change the router")
	if !ok || task != "the router" {
		t.Fatalf("got %q, %v", task, ok)
	}
	if _, ok := parseWhereTask("where is the bathroom"); ok {
		t.Fatal("should not match a non-change where-question")
	}
}

// TestVisualFollowupRegex guards the "show me a visual" follow-up shape:
// a show/draw verb aimed at a visual noun. It must not catch "show me
// <symbol>" source questions or Go questions that mention rendering.
func TestVisualFollowupRegex(t *testing.T) {
	yes := []string{
		"show me a visual",
		"draw me a diagram of this project",
		"show me the dependency graph",
		"give me a picture of the codebase",
	}
	for _, in := range yes {
		if !visualFollowup.MatchString(in) {
			t.Errorf("visualFollowup(%q) = false, want true", in)
		}
	}
	no := []string{
		"show me routeDomain",
		"show me greet",
		"how do I render html templates",
		"what does a graph database do",
	}
	for _, in := range no {
		if visualFollowup.MatchString(in) {
			t.Errorf("visualFollowup(%q) = true, want false", in)
		}
	}
}
