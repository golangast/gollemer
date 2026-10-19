package analyze

import (
	"fmt"
	"strings"
)

// This file computes contextual intent: not just "what does this do" but
// "why does it exist in this system." It looks at callers and callees to
// infer the function's role — what it enables, what depends on it, and
// where it sits in the purpose chain.

// Role is the architectural role of a function in the system.
type Role string

const (
	RoleEntryPoint  Role = "entry point"  // called by main/init or external
	RoleCoordinator Role = "coordinator"  // calls many, called by few — orchestrates
	RoleWorker      Role = "worker"        // called by many, calls few — does the work
	RoleLeaf        Role = "leaf"         // calls nothing — pure computation or I/O
	RoleHub         Role = "hub"          // called by many and calls many — central
	RoleIsolated    Role = "isolated"     // no callers or callees in project
)

// ContextualRole determines the function's role from the call graph.
func (fn *Func) ContextualRole() Role {
	callers := len(fn.Callers)
	callees := len(fn.Calls)

	if fn.IsMain || fn.IsInit {
		return RoleEntryPoint
	}
	if callers == 0 && callees == 0 {
		return RoleIsolated
	}
	if callers == 0 {
		return RoleEntryPoint
	}
	if callees == 0 {
		return RoleLeaf
	}
	// Coordinator: fans out to many, few call it.
	if callees >= 3 && callers <= 2 {
		return RoleCoordinator
	}
	// Worker: many call it, it calls few.
	if callers >= 3 && callees <= 2 {
		return RoleWorker
	}
	// Hub: central, many both ways.
	if callers >= 3 && callees >= 3 {
		return RoleHub
	}
	// Default: worker (does something for its caller).
	return RoleWorker
}

// WhyExists explains the function's purpose in system context: what it
// enables for its callers, based on what they do with it.
func (fn *Func) WhyExists(p *Project) string {
	role := fn.ContextualRole()
	var b strings.Builder

	switch role {
	case RoleEntryPoint:
		b.WriteString("Entry point — starts the work. ")
	case RoleCoordinator:
		b.WriteString(fmt.Sprintf("Coordinator — orchestrates %d operations. ", len(fn.Calls)))
	case RoleLeaf:
		b.WriteString("Leaf — does pure work, no further delegation. ")
	case RoleHub:
		b.WriteString("Hub — central, used by many parts of the system. ")
	case RoleIsolated:
		b.WriteString("Isolated — no connections in this project (maybe dead code or external API). ")
	}

	// What do callers do with it? Look at caller intent.
	if len(fn.Callers) > 0 && len(fn.Callers) <= 3 {
		var callerPurposes []string
		for _, callerID := range fn.Callers {
			if caller := p.byID[callerID]; caller != nil {
				// What does the caller do? Use its verb phrase.
				purpose := verbPhrase(caller.Name, true)
				callerPurposes = append(callerPurposes, fmt.Sprintf("%s (%s)", caller.Display(), purpose))
			}
		}
		if len(callerPurposes) > 0 {
			b.WriteString(fmt.Sprintf("Enables: %s. ", strings.Join(callerPurposes, ", ")))
		}
	} else if len(fn.Callers) > 3 {
		b.WriteString(fmt.Sprintf("Used by %d callers — a shared utility. ", len(fn.Callers)))
	}

	// What does it enable downstream? (What do its callees do?)
	if len(fn.Calls) > 0 && len(fn.Calls) <= 5 {
		var downstream []string
		seen := map[string]bool{}
		for _, calleeID := range fn.Calls {
			if callee := p.byID[calleeID]; callee != nil && !seen[callee.Name] {
				seen[callee.Name] = true
				downstream = append(downstream, verbPhrase(callee.Name, false))
			}
		}
		if len(downstream) > 0 {
			b.WriteString(fmt.Sprintf("It works by: %s.", strings.Join(downstream, ", ")))
		}
	}

	return strings.TrimSpace(b.String())
}
