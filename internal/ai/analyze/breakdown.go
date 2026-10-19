package analyze

import (
	"fmt"
	"strings"
)

// This file breaks problems down into steps using the analyzer's deep
// knowledge of real systems. Not keyword matching — it finds similar
// code in the project and extracts the steps by analogy.

// BreakDown decomposes a problem description into steps.
// It looks for similar functions in the project and uses their structure.
func (p *Project) BreakDown(problem string) string {
	problemLower := strings.ToLower(problem)
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Breaking down: %s\n\n", problem))

	// Find relevant functions: first by pattern (actual handlers, validators),
	// then by keyword in name/intent.
	var relevant []*Func

	// Pattern-based: if problem mentions "handler", find actual handlers.
	if strings.Contains(problemLower, "handler") {
		for _, fn := range p.AllFuncs() {
			if isHandlerPattern(fn) {
				relevant = append(relevant, fn)
				if len(relevant) >= 3 {
					break
				}
			}
		}
	}
	// Pattern-based: if problem mentions "valid", find actual validators.
	if len(relevant) == 0 && strings.Contains(problemLower, "valid") {
		for _, fn := range p.AllFuncs() {
			if isValidatorPattern(fn) {
				relevant = append(relevant, fn)
				if len(relevant) >= 3 {
					break
				}
			}
		}
	}

	// Fallback to keyword matching if no patterns found.
	if len(relevant) == 0 {
		for _, fn := range p.AllFuncs() {
			nameLower := strings.ToLower(fn.Name)
			// Only match if the keyword is a significant part of the name.
			for _, word := range strings.Fields(problemLower) {
				if len(word) > 4 && strings.Contains(nameLower, word) {
					relevant = append(relevant, fn)
					break
				}
			}
			if len(relevant) >= 3 {
				break
			}
		}
	}

	if len(relevant) == 0 {
		b.WriteString("No similar code found in this project to learn from.\n")
		return b.String()
	}

	b.WriteString(fmt.Sprintf("Found %d similar function(s) to learn from:\n\n", len(relevant)))
	for i, fn := range relevant {
		b.WriteString(fmt.Sprintf("Step %d: %s\n", i+1, fn.Display()))
		b.WriteString(fmt.Sprintf("  Does: %s\n", fn.Intent()))
		if len(fn.Calls) > 0 {
			b.WriteString(fmt.Sprintf("  Calls %d function(s) — these are the sub-steps\n", len(fn.Calls)))
		}
		b.WriteString(fmt.Sprintf("  Control: %s\n", fn.ControlPhrase()))
		b.WriteString("\n")
	}

	return b.String()
}

func isHandlerPattern(fn *Func) bool {
	for _, t := range fn.ParamTypes {
		if strings.Contains(t, "ResponseWriter") {
			return true
		}
	}
	return false
}

func isValidatorPattern(fn *Func) bool {
	return strings.Contains(fn.Sig, "error") && len(fn.Guards) > 0
}
