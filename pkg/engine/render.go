// Package engine: beginner-friendly rendering of a pipeline result.
//
// RenderBeginner turns a PipelineResult into plain words: the code, a
// one-line real-world analogy, a numbered walk through what the program
// does, and the safety and speed verdicts in plain English. It is the
// shared renderer for the -flow CLI flag and the /flow chat command, so
// both speak the same simple language.
package engine

import (
	"fmt"
	"regexp"
	"strings"
)

// RenderBeginner renders the result as plain words for a beginner: the
// generated Go, a one-line analogy, a numbered execution walk, and the
// safety and performance verdicts without jargon.
func (r *PipelineResult) RenderBeginner() string {
	var b strings.Builder

	b.WriteString("Here's your Go program:\n\n")
	b.WriteString("```go\n")
	b.WriteString(strings.TrimRight(r.FinalCode, "\n"))
	b.WriteString("\n```\n")

	if r.Analogy != "" {
		b.WriteString("\nThink of it like this: ")
		b.WriteString(r.Analogy)
		b.WriteString("\n")
	}

	if len(r.VisualSteps) > 0 {
		b.WriteString("\nWhat it does, step by step:\n")
		for _, s := range r.VisualSteps {
			title := cleanStepTitle(s.Title)
			fmt.Fprintf(&b, "%d. %s", s.StepNumber, title)
			if d := cleanStepDesc(s.Description, title); d != "" {
				fmt.Fprintf(&b, " — %s", d)
			}
			b.WriteString("\n")
		}
	}

	b.WriteString("\n")
	if r.IsSafetyVerified {
		b.WriteString("✅ Safety: checked — no unchecked nil dereferences, no leaked resources.\n")
	} else {
		b.WriteString("⚠️ Safety: couldn't fully verify this one — read it through before running.\n")
	}
	if d := strings.TrimSpace(r.PerformanceDelta); d != "" {
		fmt.Fprintf(&b, "⚡ Speed: %s\n", plainDelta(d))
	}
	return b.String()
}

// plainDelta rewrites the tuner's metric into plain words.
func plainDelta(d string) string {
	lower := strings.ToLower(d)
	if strings.Contains(lower, "no unbounded allocations") || strings.Contains(lower, "already at 0") {
		return "already lean — the tuner found nothing to trim."
	}
	return d
}

// stepPrefix strips a leading "Step N:" label; the renderer numbers the
// steps itself.
var stepPrefix = regexp.MustCompile(`^[Ss]tep \d+:\s*`)

func cleanStepTitle(title string) string {
	return strings.TrimSpace(stepPrefix.ReplaceAllString(title, ""))
}

// cleanStepDesc drops a description that merely echoes the title
// ("Main — calls worker." for title "Main" becomes "calls worker").
func cleanStepDesc(desc, title string) string {
	desc = strings.TrimSpace(desc)
	if desc == "" {
		return ""
	}
	if strings.HasPrefix(desc, title) {
		desc = strings.TrimSpace(strings.TrimPrefix(desc, title))
		desc = strings.TrimPrefix(desc, "—")
		desc = strings.TrimPrefix(desc, "-")
		desc = strings.TrimSpace(desc)
	}
	// A description that was only the title plus punctuation ("Worker.")
	// carries no information.
	if strings.Trim(desc, ".!") == "" {
		return ""
	}
	return desc
}
