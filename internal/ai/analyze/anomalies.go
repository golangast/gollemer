package analyze

import (
	"fmt"
	"strings"
)

// This file detects bugs and anomalies: when code doesn't do what it
// claims, when errors are ignored, when code is dead. This is deeper
// understanding — not just what the code does, but whether it's *right*.

// Anomaly is a potential bug or code smell.
type Anomaly struct {
	Kind     string // "name-mismatch", "ignored-error", "dead-code", "unchecked-return"
	Function string // function name
	Detail   string // what was found
	Severity string // "warning", "info"
}

// DetectAnomalies finds bugs and smells in the project.
func (p *Project) DetectAnomalies() []Anomaly {
	var anomalies []Anomaly

	for _, fn := range p.byID {
		// 1. Name vs behavior: function named "Validate" but no checks.
		if isValidatorName(fn.Name) && len(fn.Guards) == 0 {
			anomalies = append(anomalies, Anomaly{
				Kind:     "name-mismatch",
				Function: fn.Display(),
				Detail:   fmt.Sprintf("Named like a validator (%s) but has no condition checks", fn.Name),
				Severity: "warning",
			})
		}

		// 2. Ignored errors: calls a function returning error but doesn't check.
		// Heuristic: calls something, but no error guards.
		if len(fn.Calls) > 0 && strings.Contains(fn.Sig, "error") {
			hasErrCheck := false
			for _, g := range fn.Guards {
				if strings.Contains(strings.ToLower(g.Cond), "err") {
					hasErrCheck = true
					break
				}
			}
			// If it calls error-returning functions but never checks err, flag it.
			// (Simplified: if no err checks at all but calls exist.)
			if !hasErrCheck && len(fn.Calls) >= 2 {
				anomalies = append(anomalies, Anomaly{
					Kind:     "ignored-error",
					Function: fn.Display(),
					Detail:   "Calls multiple functions but never checks errors",
					Severity: "info",
				})
			}
		}

		// 3. Dead code: no callers, not exported, not main/init/test.
		if len(fn.Callers) == 0 && !fn.Exported && !fn.IsMain && !fn.IsInit && !fn.IsTest {
			anomalies = append(anomalies, Anomaly{
				Kind:     "dead-code",
				Function: fn.Display(),
				Detail:   "No callers in project — possibly dead code",
				Severity: "info",
			})
		}

		// 4. Empty function: no calls, no logic (just a stub?).
		if len(fn.Calls) == 0 && len(fn.Guards) == 0 && !fn.IsMain && !fn.IsTest {
			// Check if it's a simple getter (ok) vs truly empty.
			if !isGetter(fn) && fn.Name != "String" {
				anomalies = append(anomalies, Anomaly{
					Kind:     "empty-function",
					Function: fn.Display(),
					Detail:   "Does nothing — no calls, no branches (stub or incomplete?)",
					Severity: "info",
				})
			}
		}
	}

	return anomalies
}

func isValidatorName(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "valid") || strings.Contains(lower, "check") ||
		strings.Contains(lower, "verify") || strings.HasPrefix(lower, "is") ||
		strings.HasPrefix(lower, "has")
}

// AnomalyReport renders anomalies as readable text.
func (p *Project) AnomalyReport() string {
	anomalies := p.DetectAnomalies()
	if len(anomalies) == 0 {
		return "No anomalies detected."
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Found %d anomalie(s):\n\n", len(anomalies)))
	for _, a := range anomalies {
		b.WriteString(fmt.Sprintf("[%s] %s: %s\n  %s\n\n", a.Severity, a.Kind, a.Function, a.Detail))
	}
	return b.String()
}
