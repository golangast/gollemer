package analyze

import (
	"fmt"
	"strings"
)

// This file builds dataflow understanding: where values come from and where
// they go. Not just "X calls Y" but "X passes validated input to Y" — the
// causal chain that makes code understandable.

// Flow is one value-flow through a function: a parameter or value, where
// it's used, and what happens to it.
type Flow struct {
	Source string   // e.g. "param url", "field cache", "return of tryFetch"
	Uses   []string // e.g. ["passed to tryFetch", "checked for error"]
}

// Flows returns the dataflow summary for a function: for each parameter,
// where it goes; for each significant value, how it's used.
func (fn *Func) Flows(p *Project) []Flow {
	var flows []Flow

	// Parameter flows: where does each param go?
	for paramName, paramType := range fn.ParamTypes {
		// Skip receiver.
		if fn.Receiver != "" && paramName == strings.ToLower(fn.Receiver[:1]) {
			continue
		}
		var uses []string
		// Passed to which calls?
		for _, ca := range fn.CallArgs {
			for _, arg := range ca.Args {
				if arg == paramName {
					target := ca.Target
					if target == "" {
						target = "external call"
					} else if callee := p.byID[target]; callee != nil {
						target = callee.Name
					}
					uses = append(uses, fmt.Sprintf("passed to %s", target))
				}
			}
		}
		// Used in guards?
		for _, g := range fn.Guards {
			if strings.Contains(g.Cond, paramName) {
				uses = append(uses, fmt.Sprintf("checked in condition `%s`", g.Cond))
			}
		}
		if len(uses) > 0 {
			flows = append(flows, Flow{
				Source: fmt.Sprintf("param `%s` (%s)", paramName, shortType(paramType)),
				Uses:   uses,
			})
		}
	}

	// Field flows: what fields are read/written?
	for _, r := range fn.Reads {
		flows = append(flows, Flow{
			Source: fmt.Sprintf("reads `%s`", r),
			Uses:   []string{"value used in computation"},
		})
	}
	for _, w := range fn.Writes {
		flows = append(flows, Flow{
			Source: fmt.Sprintf("writes `%s`", w),
			Uses:   []string{"value stored"},
		})
	}

	return flows
}

// FlowSummary renders the dataflow as plain sentences.
func (fn *Func) FlowSummary(p *Project) string {
	flows := fn.Flows(p)
	if len(flows) == 0 {
		return ""
	}
	var parts []string
	for _, f := range flows {
		parts = append(parts, fmt.Sprintf("%s → %s", f.Source, strings.Join(f.Uses, ", ")))
	}
	return strings.Join(parts, ". ") + "."
}

func shortType(t string) string {
	// "main.Server" -> "Server", "net/http.ResponseWriter" -> "ResponseWriter".
	if i := strings.LastIndex(t, "."); i >= 0 {
		return t[i+1:]
	}
	if i := strings.LastIndex(t, "/"); i >= 0 {
		return t[i+1:]
	}
	return t
}
