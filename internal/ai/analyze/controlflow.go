package analyze

import (
	"fmt"
	"strings"
)

// This file summarizes control flow: branches, loops, error paths. Not just
// "what does it call" but "under what conditions" — the decision structure
// of the code.

// ControlSummary describes the branching structure of a function.
type ControlSummary struct {
	Branches   int      // number of if/switch branches
	Loops      int      // number of loops (from LoopCalls)
	ErrorPaths []string // conditions that check errors, e.g. "err != nil"
	HasDefer   bool     // uses defer
}

// Control returns the control-flow summary for a function.
func (fn *Func) Control() ControlSummary {
	cs := ControlSummary{
		Branches: len(fn.Guards),
		Loops:    len(fn.LoopCalls),
	}
	for _, g := range fn.Guards {
		cond := strings.ToLower(g.Cond)
		if strings.Contains(cond, "err !=") || strings.Contains(cond, "err ==") {
			cs.ErrorPaths = append(cs.ErrorPaths, g.Cond)
		}
	}
	// Check for defer in ExtCalls or via a simple heuristic.
	for _, e := range fn.ExtCalls {
		if strings.Contains(e, "defer") {
			cs.HasDefer = true
			break
		}
	}
	return cs
}

// ControlPhrase renders the control flow as a plain sentence.
func (fn *Func) ControlPhrase() string {
	cs := fn.Control()
	var parts []string
	if cs.Branches > 0 {
		if cs.Branches == 1 {
			parts = append(parts, "1 branch")
		} else {
			parts = append(parts, fmt.Sprintf("%d branches", cs.Branches))
		}
	}
	if cs.Loops > 0 {
		if cs.Loops == 1 {
			parts = append(parts, "1 loop")
		} else {
			parts = append(parts, fmt.Sprintf("%d loops", cs.Loops))
		}
	}
	if len(cs.ErrorPaths) > 0 {
		parts = append(parts, fmt.Sprintf("handles errors (%s)", strings.Join(cs.ErrorPaths, ", ")))
	}
	if cs.HasDefer {
		parts = append(parts, "uses defer for cleanup")
	}
	if len(parts) == 0 {
		return "straight-line code, no branches or loops"
	}
	return strings.Join(parts, ", ")
}
