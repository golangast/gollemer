package analyze

import (
	"fmt"
	"strings"
)

// ImpactReport shows what depends on a function across files.
type ImpactReport struct {
	Target      string
	DirectCalls []string // functions that call target directly
	Transitive  []string // all upstream callers (transitive closure)
	Files       []string // files affected
}

// AnalyzeImpact finds all callers of a function, transitively, across files.
func (p *Project) AnalyzeImpact(funcID string) *ImpactReport {
	r := &ImpactReport{Target: funcID}
	if p == nil {
		return r
	}

	// Build caller map if not already done
	p.BuildGraph()

	visited := map[string]bool{}
	queue := []string{funcID}
	visited[funcID] = true

	fileSet := map[string]bool{}

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		fn := p.FuncByID(cur)
		if fn == nil {
			continue
		}
		for _, callerID := range fn.Callers {
			if !visited[callerID] {
				visited[callerID] = true
				queue = append(queue, callerID)
				caller := p.FuncByID(callerID)
				if caller != nil {
					r.Transitive = append(r.Transitive, callerID)
					fileSet[caller.File] = true
					// Direct callers are those that call the target directly
					if cur == funcID {
						r.DirectCalls = append(r.DirectCalls, callerID)
					}
				}
			}
		}
		if fn.File != "" {
			fileSet[fn.File] = true
		}
	}

	for f := range fileSet {
		r.Files = append(r.Files, f)
	}
	return r
}

// Describe returns a plain-words impact summary.
func (r *ImpactReport) Describe() string {
	if len(r.Transitive) == 0 {
		return fmt.Sprintf("%s has no callers (dead code or entry point).", r.Target)
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Changing %s affects:\n", r.Target))
	b.WriteString(fmt.Sprintf("- %d direct caller(s): %s\n", len(r.DirectCalls), strings.Join(r.DirectCalls, ", ")))
	b.WriteString(fmt.Sprintf("- %d total upstream (transitive)\n", len(r.Transitive)))
	b.WriteString(fmt.Sprintf("- %d file(s): %s\n", len(r.Files), strings.Join(r.Files, ", ")))
	if len(r.Transitive) > 5 {
		b.WriteString("Blast radius is LARGE — change carefully.")
	} else if len(r.Transitive) > 0 {
		b.WriteString("Blast radius is small.")
	}
	return b.String()
}

// FindPath finds a call chain from one function to another (BFS).
func (p *Project) FindPath(fromID, toID string) []string {
	if p == nil {
		return nil
	}
	p.BuildGraph()

	// BFS from 'from' following callees (reverse of callers)
	// We need callee map; build it from caller info
	callees := map[string][]string{}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			for _, callerID := range fn.Callers {
				callees[callerID] = append(callees[callerID], fn.ID)
			}
		}
	}

	visited := map[string]bool{fromID: true}
	parent := map[string]string{}
	queue := []string{fromID}

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		if cur == toID {
			// Reconstruct path
			path := []string{toID}
			for path[0] != fromID {
				path = append([]string{parent[path[0]]}, path...)
			}
			return path
		}
		for _, next := range callees[cur] {
			if !visited[next] {
				visited[next] = true
				parent[next] = cur
				queue = append(queue, next)
			}
		}
	}
	return nil
}
