package analyze

import (
	"fmt"
	"strings"
)

// This file analyzes failure modes: what can go wrong, who handles it, and
// whether it's safe to retry. Deep understanding includes knowing the
// risks, not just the happy path.

// FailureMode describes how a function can fail and what happens.
type FailureMode struct {
	CanFail      bool     // returns error or panics
	ErrorChecks  []string // conditions that check errors
	HandledBy    []string // callers that check its error
	RetrySafe    bool     // heuristic: no writes, idempotent-looking
	Panics       []string // panic calls found
}

// AnalyzeFailures computes the failure modes for a function.
func (fn *Func) AnalyzeFailures(p *Project) FailureMode {
	fm := FailureMode{}

	// Does it return an error? Check signature.
	if strings.Contains(fn.Sig, "error") {
		fm.CanFail = true
	}

	// What errors does it check?
	for _, g := range fn.Guards {
		cond := strings.ToLower(g.Cond)
		if strings.Contains(cond, "err !=") {
			fm.ErrorChecks = append(fm.ErrorChecks, g.Cond)
		}
	}

	// Who handles its errors? (Callers that check err after calling it.)
	// Heuristic: caller has a guard mentioning err.
	for _, callerID := range fn.Callers {
		if caller := p.byID[callerID]; caller != nil {
			for _, g := range caller.Guards {
				if strings.Contains(strings.ToLower(g.Cond), "err") {
					fm.HandledBy = append(fm.HandledBy, caller.Name)
					break
				}
			}
		}
	}

	// Retry safe? No writes and no external mutations.
	fm.RetrySafe = len(fn.Writes) == 0 && !fn.MutatesAll

	// Panics?
	for _, e := range fn.ExtCalls {
		if e == "panic" {
			fm.Panics = append(fm.Panics, "calls panic")
		}
	}

	return fm
}

// FailurePhrase renders failure modes as plain sentences.
func (fn *Func) FailurePhrase(p *Project) string {
	fm := fn.AnalyzeFailures(p)
	var parts []string

	if !fm.CanFail && len(fm.ErrorChecks) == 0 {
		return "no error handling — cannot fail (or failures are unchecked)"
	}

	if fm.CanFail {
		parts = append(parts, "can return an error")
	}
	if len(fm.ErrorChecks) > 0 {
		parts = append(parts, fmt.Sprintf("checks %d error condition(s)", len(fm.ErrorChecks)))
	}
	if len(fm.HandledBy) > 0 {
		// Dedupe.
		seen := map[string]bool{}
		var unique []string
		for _, h := range fm.HandledBy {
			if !seen[h] {
				seen[h] = true
				unique = append(unique, h)
			}
		}
		parts = append(parts, fmt.Sprintf("errors handled by: %s", strings.Join(unique, ", ")))
	} else if fm.CanFail {
		parts = append(parts, "errors NOT handled by callers (may propagate up)")
	}
	if fm.RetrySafe {
		parts = append(parts, "safe to retry (no writes)")
	} else {
		parts = append(parts, "NOT safe to retry (has side effects)")
	}
	if len(fm.Panics) > 0 {
		parts = append(parts, "can panic")
	}

	return strings.Join(parts, "; ")
}
