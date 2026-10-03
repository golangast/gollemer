package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// consistencyFixture builds three game packages, each with its own
// win-check shape, plus a CheckGameOver family member in one.
func consistencyFixture(t *testing.T) *Project {
	t.Helper()
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/cy\n\ngo 1.21\n")
	write("app/snake/snake.go", `package snake

type model struct{}

func (m model) CheckForWin() bool { return false }

func (m model) Update() { _ = m.CheckForWin() }
`)
	write("app/pong/pong.go", `package pong

type model struct{}

func (m model) CheckForWin() int { return 0 }
`)
	write("app/chess/engine/engine.go", `package engine

type Engine struct{}

func (e *Engine) CheckWin() bool { return false }

func (e *Engine) CheckGameOver() bool { return e.CheckWin() }
`)
	write("app/chess/chess.go", `package chess

type model struct{}

func helper() {}
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestConsistencyPlan(t *testing.T) {
	p := consistencyFixture(t)
	c := ExtractIssueConcepts("[FEAT] Consistent win/fail behaviours\n\n" +
		"Unify win handling across games. Each game does it differently.")
	cp := p.buildConsistencyPlan(c)
	if cp == nil {
		t.Fatal("buildConsistencyPlan returned nil")
	}
	out := p.GuideIssue(c)
	for _, want := range []string{
		"SITES",
		"snake", "CheckForWin",
		"pong", "CheckForWin",
		"engine", "CheckWin", "CheckGameOver",
		"shared helper",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("consistency plan missing %q:\n%s", want, out)
		}
	}
	// One anchor is the wrong shape for consistency issues.
	if strings.Contains(out, "BEHAVIOR") {
		t.Errorf("consistency plan should not render BEHAVIOR:\n%s", out)
	}
	if strings.Contains(out, "YOU ARE HERE") {
		t.Errorf("consistency plan should not draw a single chain:\n%s", out)
	}
}

func TestConsistencyPlanNeedsSignal(t *testing.T) {
	p := consistencyFixture(t)
	// No uniformity language -> no consistency plan.
	c := ExtractIssueConcepts("Add win counter\n\nCount wins per game.")
	if cp := p.buildConsistencyPlan(c); cp != nil {
		t.Errorf("no consistency signal — plan should be nil, got %+v", cp)
	}
}

func TestConsistencyPlanNeedsSpan(t *testing.T) {
	p := consistencyFixture(t)
	// Uniformity language but the concept lives in one package only.
	c := ExtractIssueConcepts("Consistent helper naming\n\nUnify helper names across the codebase.")
	if cp := p.buildConsistencyPlan(c); cp != nil {
		t.Errorf("single-package concept — plan should be nil, got %+v", cp)
	}
}
