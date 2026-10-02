package engine

import (
	"strconv"
	"strings"
	"unicode/utf8"
)

// This file draws the /flow run as a box-style pipeline visual in the
// terminal: the prompt flows through the four engine stages into the
// renderer. Stage results (the safety badge, the tuning delta) are the
// real per-run values, not placeholders.

// diagramWidth is the inner width of every box, in characters.
const diagramWidth = 64

// RenderDiagram draws the pipeline visual for one /flow run:
//
//	[ YOUR PROMPT ]
//	  "make an http handler"
//	          │
//	          ▼
//	┌────────────────────────────────────────────────────────────────┐
//	│                    pkg/engine/pipeline.go                      │
//	├────────────────────────────────────────────────────────────────┤
//	│ 1. Synthesis    -> prompt becomes formatted Go code             │
//	│ 2. Proving      -> 0 nil-panics, 0 leaks                   ✓    │
//	│ 3. Auto-tuning  -> ...                                         │
//	│ 4. Visuals      -> execution walk + beginner analogy            │
//	└──────────────────────────────┬─────────────────────────────────┘
//	                               │
//	                               ▼
//	┌────────────────────────────────────────────────────────────────┐
//	│                      TERMINAL RENDERER                         │
//	...
func (r *PipelineResult) RenderDiagram(prompt string) string {
	var b strings.Builder
	b.WriteString("[ YOUR PROMPT ]\n")
	b.WriteString("  " + strconv.Quote(truncateRunes(prompt, 58)) + "\n")
	b.WriteString(diagramArrow() + "\n")
	b.WriteString(diagramBox("pkg/engine/pipeline.go", []string{
		"1. Synthesis    -> prompt becomes formatted Go code",
		"2. Proving      -> " + provingLine(r),
		"3. Auto-tuning  -> " + tuningLine(r),
		"4. Visuals      -> execution walk + beginner analogy",
	}, true))
	b.WriteString("\n" + diagramArrow() + "\n")
	badge := "✓"
	if !r.IsSafetyVerified {
		badge = "⚠"
	}
	b.WriteString(diagramBox("TERMINAL RENDERER", []string{
		"• Production Go Code (go/format)",
		"• Beginner Analogy",
		"• Step-by-Step Execution Walk",
		"• Safety & Allocation Badge    " + badge,
	}, false))
	return b.String()
}

// provingLine is the real stage-2 outcome for the diagram.
func provingLine(r *PipelineResult) string {
	if r.IsSafetyVerified {
		return "0 nil-panics, 0 leaks              ✓"
	}
	return "needs a human read-through      ⚠"
}

// tuningLine is the real stage-3 outcome for the diagram.
func tuningLine(r *PipelineResult) string {
	if d := strings.TrimSpace(r.PerformanceDelta); d != "" {
		return truncateRunes(plainDelta(d), 44)
	}
	return "code already lean"
}

// diagramBox draws a titled box of fixed inner width. When outlet is
// true the bottom border carries a ┬ where the next arrow continues.
func diagramBox(title string, rows []string, outlet bool) string {
	var b strings.Builder
	bar := strings.Repeat("─", diagramWidth)
	b.WriteString("┌" + bar + "┐\n")
	b.WriteString("│" + padCenter(title, diagramWidth) + "│\n")
	b.WriteString("├" + bar + "┤\n")
	for _, r := range rows {
		b.WriteString("│" + padRight(r, diagramWidth) + "│\n")
	}
	if outlet {
		b.WriteString("└" + strings.Repeat("─", diagramWidth/2) + "┬" +
			strings.Repeat("─", diagramWidth/2-1) + "┘")
	} else {
		b.WriteString("└" + bar + "┘")
	}
	return b.String()
}

// diagramArrow is the │/▼ connector, centered under a box.
func diagramArrow() string {
	c := strings.Repeat(" ", diagramWidth/2+1)
	return c + "│\n" + c + "▼"
}

// padRight pads s to w columns, counting runes (box-drawing chars are
// multibyte). Overlong strings are truncated cleanly.
func padRight(s string, w int) string {
	s = truncateRunes(s, w)
	return s + strings.Repeat(" ", w-utf8.RuneCountInString(s))
}

// padCenter centers s in w columns.
func padCenter(s string, w int) string {
	s = truncateRunes(s, w)
	n := utf8.RuneCountInString(s)
	left := (w - n) / 2
	return strings.Repeat(" ", left) + s + strings.Repeat(" ", w-n-left)
}

// truncateRunes shortens s to at most n runes, adding … when cut.
func truncateRunes(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return strings.TrimSpace(string(r[:n-1])) + "…"
}
