package analyze

import (
	"strings"
	"testing"
)

// The beginner walkthrough runs a real function through the xray
// intelligence engine: analogy, execution walk, safety badge.
func TestExplainBeginner(t *testing.T) {
	p, err := Analyze(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	out, ok := p.ExplainBeginner("main")
	if !ok {
		t.Fatal("ExplainBeginner(main) not ok")
	}
	for _, want := range []string{
		"BEGINNER WALK:",
		"Think of it like this:",
		"Walk-through:",
		"Safety:",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("walkthrough missing %q", want)
		}
	}
}

// Unknown symbols report not-ok so the caller can fall back.
func TestExplainBeginnerUnknown(t *testing.T) {
	p, err := Analyze(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := p.ExplainBeginner("noSuchFunction"); ok {
		t.Errorf("ExplainBeginner(noSuchFunction) = ok, want false")
	}
}

// The new chat patterns route to the walkthrough.
func TestAnswerBeginnerPatterns(t *testing.T) {
	p, err := Analyze(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{
		"walk me through main",
		"walk through main",
		"trace main",
		"explain main for beginners",
		"explain main like i'm a beginner",
	} {
		out, ok := p.Answer(q)
		if !ok {
			t.Errorf("Answer(%q) not ok", q)
			continue
		}
		if !strings.Contains(out, "BEGINNER WALK:") {
			t.Errorf("Answer(%q) did not produce a walkthrough", q)
		}
	}
}

// An unknown symbol still gets the reference fallback, not silence.
func TestAnswerBeginnerFallback(t *testing.T) {
	p, err := Analyze(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	out, ok := p.Answer("walk me through noSuchFunction")
	if ok {
		t.Errorf("Answer(unknown walkthrough) = ok with %q, want not-ok", out)
	}
}
