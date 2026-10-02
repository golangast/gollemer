package analyze

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/pkg/xray"
)

// This file wires the intelligence engine (pkg/xray) into code
// understanding: "walk me through X" runs a function through the
// X-Ray assistant — a plain-words explanation, a real-world analogy,
// a step-by-step execution walk, and a safety badge. It's the
// beginner's view of a symbol, powered by the same engine as /flow.

// xrayTimeout bounds one beginner walk; the engine is pure AST work,
// so this is generous.
const xrayTimeout = 30 * time.Second

// maxWalkSteps caps the displayed execution steps so the walk stays
// readable; the engine itself may trace more.
const maxWalkSteps = 12

// ExplainBeginner runs symbol through the xray engine and renders the
// beginner walkthrough. It reports ok=false when the symbol doesn't
// resolve to a function, letting the caller fall back to the reference
// answer.
func (p *Project) ExplainBeginner(symbol string) (string, bool) {
	fn, _, _, _ := p.resolve(symbol)
	if fn == nil {
		return "", false
	}
	src, err := os.ReadFile(filepath.Join(p.Root, fn.File))
	if err != nil {
		return "", false
	}
	ctx, cancel := context.WithTimeout(context.Background(), xrayTimeout)
	defer cancel()
	resp, err := xray.SynthesizeWithXRay(ctx, xray.XRayRequest{
		Query:       fmt.Sprintf("trace %q", fn.Name),
		ContextCode: string(src),
	})
	if err != nil {
		return "", false
	}
	return renderXrayWalk(fn, resp), true
}

// renderXrayWalk renders the XRayResponse as the chat's beginner walk.
func renderXrayWalk(fn *Func, resp *xray.XRayResponse) string {
	var b strings.Builder
	fmt.Fprintf(&b, "BEGINNER WALK: %s\n", fn.Display())
	if d := firstSentence(fn.Doc); d != "" {
		fmt.Fprintf(&b, "%s\n", d)
	}
	b.WriteString("\nThink of it like this: ")
	b.WriteString(resp.Analogy)
	b.WriteString("\n\n")
	if resp.BeginnerExplanation != "" {
		b.WriteString(resp.BeginnerExplanation)
		b.WriteString("\n\n")
	}
	steps := resp.VisualSequence
	if len(steps) > 0 {
		b.WriteString("Walk-through:\n")
		if len(steps) > maxWalkSteps {
			steps = steps[:maxWalkSteps]
		}
		for _, s := range steps {
			title := strings.TrimSpace(s.Title)
			if title == "" {
				title = fmt.Sprintf("Step %d", s.StepNumber)
			}
			if d := strings.TrimSpace(s.Description); d != "" {
				fmt.Fprintf(&b, "%d. %s — %s\n", s.StepNumber, title, firstSentence(d))
			} else {
				fmt.Fprintf(&b, "%d. %s\n", s.StepNumber, title)
			}
			if snip := snippetHead(s.CodeSnippet, 8); snip != "" {
				b.WriteString("```go\n")
				b.WriteString(snip)
				b.WriteString("\n```\n")
			}
		}
		if len(resp.VisualSequence) > maxWalkSteps {
			fmt.Fprintf(&b, "… (+%d more steps)\n", len(resp.VisualSequence)-maxWalkSteps)
		}
		b.WriteString("\n")
	}
	b.WriteString("Safety: ")
	b.WriteString(safetyLine(resp.SafetyBadge))
	b.WriteString("\n")
	return b.String()
}

// safetyLine renders the badge in one plain line.
func safetyLine(b xray.SafetyBadge) string {
	status := strings.TrimSpace(b.NullSafetyStatus)
	if status == "" {
		status = "unknown"
	}
	alloc := strings.TrimSpace(b.AllocEstimate)
	if alloc == "" {
		return status
	}
	return status + " · " + alloc
}

// snippetHead keeps the first n lines of a code snippet.
func snippetHead(snippet string, n int) string {
	snippet = strings.TrimRight(snippet, "\n")
	lines := strings.Split(snippet, "\n")
	if len(lines) > n {
		lines = lines[:n]
	}
	return strings.Join(lines, "\n")
}
