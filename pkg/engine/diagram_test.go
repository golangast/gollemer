package engine

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// The diagram shows the prompt, all four stages, and the renderer box.
func TestRenderDiagramStages(t *testing.T) {
	r := &PipelineResult{IsSafetyVerified: true, PerformanceDelta: "x"}
	out := r.RenderDiagram("make an http handler")
	for _, want := range []string{
		"[ YOUR PROMPT ]",
		`"make an http handler"`,
		"pkg/engine/pipeline.go",
		"1. Synthesis",
		"2. Proving",
		"3. Auto-tuning",
		"4. Visuals",
		"TERMINAL RENDERER",
		"┌", "┐", "└", "┘", "│", "─", "▼", "┬",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("RenderDiagram missing %q", want)
		}
	}
}

// The safety badge reflects the real per-run verdict.
func TestRenderDiagramBadge(t *testing.T) {
	ok := (&PipelineResult{IsSafetyVerified: true}).RenderDiagram("x")
	if !strings.Contains(ok, "✓") {
		t.Errorf("verified run should show ✓")
	}
	bad := (&PipelineResult{IsSafetyVerified: false}).RenderDiagram("x")
	if !strings.Contains(bad, "⚠") {
		t.Errorf("unverified run should show ⚠")
	}
}

// Every box line is exactly 66 cells wide so the borders line up.
func TestDiagramBoxWidth(t *testing.T) {
	r := &PipelineResult{IsSafetyVerified: true}
	out := r.RenderDiagram("a prompt that is a bit longer than usual, for width")
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		// Only box borders start at column 0; centered │/▼ arrows don't.
		if strings.HasPrefix(line, "┌") || strings.HasPrefix(line, "│") ||
			strings.HasPrefix(line, "├") || strings.HasPrefix(line, "└") {
			if w := utf8.RuneCountInString(line); w != diagramWidth+2 {
				t.Errorf("box line width %d, want %d: %q", w, diagramWidth+2, line)
			}
		}
	}
}

func TestPadHelpers(t *testing.T) {
	if got := padRight("ab", 5); got != "ab   " {
		t.Errorf("padRight = %q", got)
	}
	if got := padRight("abcdef", 5); got != "abcd…" {
		t.Errorf("padRight truncation = %q", got)
	}
	// Multibyte box chars count as one cell.
	if got := padRight("─", 3); got != "─  " {
		t.Errorf("padRight multibyte = %q", got)
	}
	if got := padCenter("ab", 6); got != "  ab  " {
		t.Errorf("padCenter = %q", got)
	}
}
