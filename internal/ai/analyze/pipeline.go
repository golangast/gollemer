package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// This file answers the beginner's hardest question: "what happens when
// I run this?" Pipeline draws the execution flow as an ASCII tree
// starting at each func main — main calls X, X calls Y — with each
// step's doc comment so the story reads in plain words. Story renders
// the same flow as a short paragraph.

const (
	maxPipelineDepth   = 4  // levels below main
	maxPipelineKids    = 5  // callees shown per function
	maxPipelineNodes   = 40 // total nodes before truncation
	maxPipelineDocLen  = 64 // doc snippet per node
)

// Pipeline draws the call flow starting at every func main. Cycles are
// cut with a ↺ marker; depth, breadth, and total nodes are capped so
// the tree stays readable. For libraries (no main) it starts at the
// call-graph roots: exported functions nothing calls — the API surface.
func (p *Project) Pipeline() string {
	var b strings.Builder
	b.WriteString("PIPELINE: what happens when you run it\n")
	b.WriteString("Follow the arrows — each step calls the next.\n\n")
	mains := p.EntryPoints()
	if len(mains) == 0 {
		b.WriteString("No func main: this is a library, not a runnable program.\n")
		b.WriteString("Its story starts at the functions nothing else calls:\n")
		roots := p.callRoots(3)
		for _, fn := range roots {
			fmt.Fprintf(&b, "  %s — %s\n", fn.Display(), shortDoc(fn, maxPipelineDocLen))
		}
		if len(roots) == 0 {
			b.WriteString("  (no clear entry functions found)\n")
		}
		return b.String()
	}
	pw := &pipelineWalker{p: p}
	for i, m := range mains {
		if i > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "%s  (%s:%d)\n", m.Name, m.File, m.Line)
		if d := shortDoc(m, maxPipelineDocLen); d != "" {
			fmt.Fprintf(&b, "  %s\n", d)
		}
		pw.walk(&b, m, []string{m.ID}, 1, "")
		if pw.truncated {
			b.WriteString("  … (deeper calls truncated)\n")
		}
	}
	return b.String()
}

// Story renders the first two levels of the pipeline as a plain-words
// paragraph: the beginner's "what does this program do" answer.
func (p *Project) Story() string {
	var b strings.Builder
	b.WriteString("STORY: what happens when you run it, in plain words\n\n")
	mains := p.EntryPoints()
	if len(mains) == 0 {
		b.WriteString("This is a library — there's no main program to run. ")
		roots := p.callRoots(3)
		if len(roots) > 0 {
			names := make([]string, 0, len(roots))
			for _, fn := range roots {
				names = append(names, fn.Display()+"()")
			}
			fmt.Fprintf(&b, "Other code starts from %s.\n", joinWords(names))
		}
		return b.String()
	}
	m := mains[0]
	fmt.Fprintf(&b, "Everything starts at main() in %s.\n", m.File)
	kids := p.topCallees(m, 4)
	if len(kids) == 0 {
		b.WriteString("main() does its work directly, without calling other project functions.\n")
		return b.String()
	}
	for _, k := range kids {
		if d := shortDoc(k, 90); d != "" {
			fmt.Fprintf(&b, "main() calls %s(): %s\n", k.Name, d)
		} else {
			fmt.Fprintf(&b, "main() calls %s().\n", k.Name)
		}
	}
	// One level deeper through the most important first call.
	grandkids := p.topCallees(kids[0], 3)
	if len(grandkids) > 0 {
		names := make([]string, 0, len(grandkids))
		for _, g := range grandkids {
			names = append(names, g.Name+"()")
		}
		fmt.Fprintf(&b, "From there, %s() calls %s.\n", kids[0].Name, joinWords(names))
	}
	return b.String()
}

// pipelineWalker holds the traversal budget shared across entry points.
type pipelineWalker struct {
	p         *Project
	nodes     int
	truncated bool
}

// walk prints the callee tree of fn. prefix carries the tree-drawing
// indentation; path holds the IDs on the current branch for cycle cuts.
func (w *pipelineWalker) walk(b *strings.Builder, fn *Func, path []string, depth int, prefix string) {
	if depth > maxPipelineDepth || w.truncated {
		return
	}
	kids := w.p.topCallees(fn, maxPipelineKids+1)
	shown := kids
	more := 0
	if len(kids) > maxPipelineKids {
		shown = kids[:maxPipelineKids]
		more = len(kids) - maxPipelineKids
	}
	for i, k := range shown {
		last := i == len(shown)-1 && more == 0
		branch := "├─▶ "
		childPrefix := prefix + "│   "
		if last {
			branch = "└─▶ "
			childPrefix = prefix + "    "
		}
		if w.nodes >= maxPipelineNodes {
			w.truncated = true
			return
		}
		w.nodes++
		if contains(path, k.ID) {
			fmt.Fprintf(b, "%s%s%s ↺ (already shown above)\n", prefix, branch, k.Display())
			continue
		}
		doc := shortDoc(k, maxPipelineDocLen)
		if doc != "" {
			fmt.Fprintf(b, "%s%s%s — %s\n", prefix, branch, k.Display(), doc)
		} else {
			fmt.Fprintf(b, "%s%s%s\n", prefix, branch, k.Display())
		}
		w.walk(b, k, append(path, k.ID), depth+1, childPrefix)
	}
	if more > 0 {
		fmt.Fprintf(b, "%s└─▶ … (+%d more)\n", prefix, more)
	}
}

// topCallees resolves fn's callees to *Func, deduplicated in call-site
// order, skipping tests and init functions. The limit caps breadth.
func (p *Project) topCallees(fn *Func, limit int) []*Func {
	var out []*Func
	seen := map[string]bool{}
	for _, id := range fn.Calls {
		if seen[id] {
			continue
		}
		seen[id] = true
		c, ok := p.byID[id]
		if !ok || c.IsTest || c.IsInit {
			continue
		}
		out = append(out, c)
		if len(out) >= limit {
			break
		}
	}
	return out
}

// callRoots returns exported functions nothing calls, by importance —
// the API surface a library is used through.
func (p *Project) callRoots(n int) []*Func {
	var roots []*Func
	for _, fn := range p.byID {
		if fn.Exported && !fn.IsTest && !fn.IsInit && len(fn.Callers) == 0 {
			roots = append(roots, fn)
		}
	}
	sort.Slice(roots, func(i, j int) bool {
		if roots[i].Score != roots[j].Score {
			return roots[i].Score > roots[j].Score
		}
		return roots[i].ID < roots[j].ID
	})
	if len(roots) > n {
		roots = roots[:n]
	}
	return roots
}

// shortDoc returns the doc comment's first sentence, trimmed to max
// characters, without a trailing period.
func shortDoc(fn *Func, max int) string {
	d := firstSentence(fn.Doc)
	d = strings.TrimSuffix(d, ".")
	if len(d) > max {
		d = strings.TrimSpace(d[:max]) + "…"
	}
	return d
}

// joinWords joins names as "a, b and c".
func joinWords(names []string) string {
	switch len(names) {
	case 0:
		return ""
	case 1:
		return names[0]
	case 2:
		return names[0] + " and " + names[1]
	default:
		return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
	}
}
