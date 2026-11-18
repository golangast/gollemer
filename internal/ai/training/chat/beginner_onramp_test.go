package chat

import (
	"strings"
	"testing"
)

// The on-ramp fallbacks: with nothing analyzed and a start folder
// that holds no Go code, a beginner command must land on the
// bundled pocketcounter demo — never on a refusal.
func TestBeginnerFallsBackToDemo(t *testing.T) {
	oldProj, oldRoot := lastAnalyzeProject, unifiedProjectRoot
	defer func() { lastAnalyzeProject, unifiedProjectRoot = oldProj, oldRoot }()
	lastAnalyzeProject = nil
	unifiedProjectRoot = t.TempDir() // empty: no Go files here
	tourStops = nil

	if !handleBeginnerCommand("/tour", nil) {
		t.Fatal("/tour should be handled")
	}
	if lastAnalyzeProject == nil {
		t.Fatal("expected the demo project to be loaded")
	}
	if !strings.Contains(lastAnalyzeRoot, "pocketcounter") {
		t.Fatalf("expected the pocketcounter demo, analyzed: %s", lastAnalyzeRoot)
	}
	if len(tourStops) == 0 || !strings.Contains(strings.Join(tourStops, " "), "Counter") {
		t.Fatalf("tour should walk the demo's Counter, got: %v", tourStops)
	}
}

func TestSentenceTranslations(t *testing.T) {
	cases := map[string]string{
		"tour https://github.com/fgm/izidic": "/tour https://github.com/fgm/izidic",
		"Give me a tour":                     "/tour",
		"what are the main types":            "/glossary",
		"What should I learn first?":         "/learn-path",
		"example Names":                      "/example Names",
		"show me an example of Names":        "/example Names",
		"how is Names used":                  "/example Names",
		"run it":                             "/watch",
		"hello there":                        "hello there", // untouched
	}
	for in, want := range cases {
		if got := translateBeginnerSentence(in); got != want {
			t.Errorf("translate(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestUnknownCommandHintSuggests(t *testing.T) {
	if h := unknownCommandHint("/ture"); !strings.Contains(h, "did you mean") {
		t.Errorf("/ture should suggest a correction, got: %s", h)
	}
	if h := unknownCommandHint("zzzzqq"); !strings.Contains(h, "help") {
		t.Errorf("gibberish should point at help, got: %s", h)
	}
}
