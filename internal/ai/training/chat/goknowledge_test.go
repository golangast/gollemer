package chat

import (
	"strings"
	"testing"
)

func TestLookupGoKnowledgeHits(t *testing.T) {
	cases := []struct {
		query string
		want  string // substring expected in the body
	}{
		{"what is a goroutine", "lightweight thread"},
		{"how do channels work in go", "move values between goroutines"},
		{"how do i read a file in go", "os.ReadFile"},
		{"what does the context package do", "cancellation"},
		{"explain interfaces to me", "implicitly"},
		{"what is go vet for", "suspicious code"},
		{"how do i sort a slice", "sort.Slice"},
		{"what is a worker pool", "bounds concurrency"},
		{"why does writing to a nil map panic", "assignment to entry in nil map"},
		{"how do i parse json in go", "json.Unmarshal"},
		{"what is select used for", "multiple channel"},
		{"tell me about sync mutex", "guards shared state"},
		{"what is a function", "multiple values"},
	}
	for _, c := range cases {
		body, ok := LookupGoKnowledge(c.query)
		if !ok {
			t.Errorf("LookupGoKnowledge(%q) missed, want hit", c.query)
			continue
		}
		if !strings.Contains(body, c.want) {
			t.Errorf("LookupGoKnowledge(%q) = %q, want substring %q", c.query, body, c.want)
		}
	}
}

func TestLookupGoKnowledgeMisses(t *testing.T) {
	// Vague, social, or ambiguous queries must fall through to the model.
	// Note: "what is a function" DOES hit — the functions entry answers it
	// well, so it lives in the hits test instead.
	for _, q := range []string{
		"hello",
		"how are you",
		"what is go",
		"i like maps",
		"",
		"???",
		"tell me about stuff",
	} {
		if body, ok := LookupGoKnowledge(q); ok {
			t.Errorf("LookupGoKnowledge(%q) hit %q, want miss", q, body)
		}
	}
}

func TestLookupGoKnowledgeBodiesAreProse(t *testing.T) {
	for _, e := range goKnowledge {
		if strings.Contains(e.body, "```") {
			t.Errorf("entry %q body contains a code block", e.title)
		}
		if len(strings.Fields(e.body)) > 80 {
			t.Errorf("entry %q body is %d words, keep it short", e.title, len(strings.Fields(e.body)))
		}
		if len(e.keywords) < 2 {
			t.Errorf("entry %q has fewer than 2 keywords", e.title)
		}
	}
}
