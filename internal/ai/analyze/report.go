package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// Summary returns the one-screen project overview.
func (p *Project) Summary() string {
	name := p.Module
	if name == "" {
		name = filepathBase(p.Root)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Project: %s\n", name)
	fmt.Fprintf(&b, "%d packages, %d Go files, ~%d lines\n", len(p.Packages), p.Files, p.Lines)
	mains := p.EntryPoints()
	switch len(mains) {
	case 0:
		b.WriteString("No func main: this is a library, not a runnable program.\n")
	case 1:
		fmt.Fprintf(&b, "Entry point: %s\n", mains[0].File)
	default:
		b.WriteString("Entry points:\n")
		for _, m := range mains {
			fmt.Fprintf(&b, "  - %s\n", m.File)
		}
	}
	ext := 0
	for _, pkg := range p.Packages {
		ext += pkg.Tests
	}
	if ext > 0 {
		fmt.Fprintf(&b, "%d test files\n", ext)
	}
	if len(p.ParseErr) > 0 {
		fmt.Fprintf(&b, "%d files skipped (parse errors)\n", len(p.ParseErr))
	}
	b.WriteString("\nIN PLAIN ENGLISH\n")
	b.WriteString(p.Overview())
	b.WriteString("\n")
	return b.String()
}

// ReadingGuide returns the "where do I start" walkthrough: entry points,
// the engine room, key types, and a suggested package reading order.
func (p *Project) ReadingGuide() string {
	var b strings.Builder
	b.WriteString("WHERE TO START\n")
	mains := p.EntryPoints()
	if len(mains) == 0 {
		b.WriteString("No main package — start with the exported API of the root package.\n")
	} else {
		b.WriteString("1. Start at the entry point:\n")
		for _, m := range mains {
			fmt.Fprintf(&b, "   %s:%d — func main\n", m.File, m.Line)
		}
	}
	b.WriteString("2. Then read the engine room (most-called functions):\n")
	for _, fn := range p.TopFuncs(5) {
		fmt.Fprintf(&b, "   %s  (%d callers)  %s:%d\n", fn.Display(), len(fn.Callers), fn.File, fn.Line)
	}
	types := p.KeyTypes(5)
	if len(types) > 0 {
		b.WriteString("3. Learn the key types:\n")
		for _, t := range types {
			desc := t.Kind
			if t.Kind == "interface" && t.ImplCount > 0 {
				desc = fmt.Sprintf("interface, %d implementers", t.ImplCount)
			} else if t.Kind == "struct" && len(t.Methods) > 0 {
				desc = fmt.Sprintf("struct, %d methods", len(t.Methods))
			}
			fmt.Fprintf(&b, "   %s.%s — %s\n", t.Pkg, t.Name, desc)
		}
	}
	b.WriteString("4. Package reading order (dependencies first):\n")
	for i, dir := range p.readingOrderDirs() {
		label := dir
		if label == "" {
			label = ". (root)"
		}
		doc := ""
		if pkg := p.byPkgDir[dir]; pkg != nil {
			doc = firstSentence(pkg.Doc)
		}
		if doc != "" {
			fmt.Fprintf(&b, "   %d. %s\n      %s\n", i+1, label, doc)
		} else {
			fmt.Fprintf(&b, "   %d. %s\n", i+1, label)
		}
	}
	return b.String()
}

// readingOrderDirs returns package dirs sorted so dependencies come before the
// packages that import them (a topological-ish order by internal imports).
// "" means the root directory.
func (p *Project) readingOrderDirs() []string {
	depth := map[string]int{}
	var visit func(dir string, seen map[string]bool) int
	visit = func(dir string, seen map[string]bool) int {
		if d, ok := depth[dir]; ok {
			return d
		}
		if seen[dir] {
			return 0
		}
		seen[dir] = true
		pkg := p.byPkgDir[dir]
		max := 0
		if pkg != nil {
			for _, imp := range pkg.Internal {
				if d := visit(imp, seen); d+1 > max {
					max = d + 1
				}
			}
		}
		delete(seen, dir)
		depth[dir] = max
		return max
	}
	dirs := make([]string, 0, len(p.Packages))
	for _, pkg := range p.Packages {
		dirs = append(dirs, pkg.Dir)
		visit(pkg.Dir, map[string]bool{})
	}
	sort.Slice(dirs, func(i, j int) bool {
		if depth[dirs[i]] != depth[dirs[j]] {
			return depth[dirs[i]] < depth[dirs[j]]
		}
		return dirs[i] < dirs[j]
	})
	out := make([]string, len(dirs))
	copy(out, dirs)
	return out
}

// readingOrder is the display form of readingOrderDirs ("" -> ". (root)").
func (p *Project) readingOrder() []string {
	out := p.readingOrderDirs()
	for i, d := range out {
		if d == "" {
			out[i] = ". (root)"
		}
	}
	return out
}

// EngineRoom renders the top functions as an ASCII bar chart.
func (p *Project) EngineRoom(n int) string {
	top := p.TopFuncs(n)
	if len(top) == 0 {
		return "no functions found\n"
	}
	max := 0
	for _, fn := range top {
		if len(fn.Callers) > max {
			max = len(fn.Callers)
		}
	}
	var b strings.Builder
	b.WriteString("ENGINE ROOM (most-called functions)\n")
	for _, fn := range top {
		bars := 0
		if max > 0 {
			bars = 1 + 19*len(fn.Callers)/max
		}
		fmt.Fprintf(&b, "%s %s (%d callers)\n", strings.Repeat("█", bars), fn.Display(), len(fn.Callers))
	}
	return b.String()
}

// PackageTree renders the package directory tree.
func (p *Project) PackageTree() string {
	var b strings.Builder
	b.WriteString("PACKAGES\n")
	dirs := make([]string, 0, len(p.Packages))
	for _, pkg := range p.Packages {
		dirs = append(dirs, pkg.Dir)
	}
	sort.Strings(dirs)
	for _, d := range dirs {
		pkg := p.byPkgDir[d]
		label := d
		if label == "" {
			label = "."
		}
		main := ""
		if pkg.HasMain {
			main = " [main]"
		}
		fmt.Fprintf(&b, "  %s/  (package %s, %d funcs, %d types%s)\n",
			label, pkg.Name, len(pkg.Funcs), len(pkg.Types), main)
	}
	return b.String()
}

// ImportGraph renders internal package dependencies as text edges.
func (p *Project) ImportGraph() string {
	var b strings.Builder
	b.WriteString("PACKAGE DEPENDENCIES\n")
	any := false
	for _, pkg := range p.Packages {
		if len(pkg.Internal) == 0 {
			continue
		}
		any = true
		from := pkg.Dir
		if from == "" {
			from = "."
		}
		to := make([]string, len(pkg.Internal))
		copy(to, pkg.Internal)
		for i, d := range to {
			if d == "" {
				to[i] = "."
			}
		}
		sort.Strings(to)
		fmt.Fprintf(&b, "  %s → %s\n", from, strings.Join(to, ", "))
	}
	if !any {
		b.WriteString("  (no internal dependencies — packages are independent)\n")
	}
	return b.String()
}

// linkInternal fills each package's Internal list from its Imports.
func (p *Project) linkInternal() {
	for _, pkg := range p.Packages {
		var internal []string
		for _, imp := range pkg.Imports {
			if target := p.importTarget(imp); target != nil {
				dir := target.Dir
				if dir != pkg.Dir && !contains(internal, dir) {
					internal = append(internal, dir)
				}
			}
		}
		sort.Strings(internal)
		pkg.Internal = internal
	}
}

// ChangeHit is one "where to change" result.
type ChangeHit struct {
	What string // "func chat.routeDomain" / "type analyze.Project"
	File string
	Line int
	Why  string
}

// WhereToChange ranks the functions and types most relevant to a task
// description like "add a new chat command" or "fix the call graph".
// It matches task keywords against symbol names (camelCase-aware) and doc
// comments, boosted by structural importance, then pulls in the top
// callers so you see what touches the area too.
func (p *Project) WhereToChange(task string) []ChangeHit {
	want := wordSet(task)
	for w := range wordSet(task) {
		for _, alias := range taskAliases[w] {
			want[alias] = true
		}
	}
	if len(want) == 0 {
		return nil
	}
	type scored struct {
		hit   ChangeHit
		score int
		fn    *Func
	}
	var cands []scored
	addName := func(name, what, file string, line int, doc string, boost int, fn *Func) {
		nameWords := map[string]bool{}
		for _, w := range camelSplit(name) {
			lw := strings.ToLower(w)
			if len(lw) > 1 && !stopwords[lw] {
				nameWords[lw] = true
			}
		}
		score := 0
		var matched []string
		for w := range want {
			if nameWords[w] {
				score += 3
				matched = append(matched, w)
			}
		}
		if score == 0 && doc != "" {
			docWords := wordSet(doc)
			for w := range want {
				if docWords[w] {
					score += 1
					matched = append(matched, w)
				}
			}
		}
		if score == 0 {
			return
		}
		sort.Strings(matched)
		score += boost
		cands = append(cands, scored{
			hit: ChangeHit{
				What: what,
				File: file,
				Line: line,
				Why:  "matches " + strings.Join(matched, ", "),
			},
			score: score,
			fn:    fn,
		})
	}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			boost := fn.Score / 4
			if fn.Exported {
				boost++
			}
			addName(fn.Name, "func "+fn.Display(), fn.File, fn.Line, fn.Doc, boost, fn)
		}
		for _, t := range pkg.Types {
			addName(t.Name, t.Kind+" "+t.Pkg+"."+t.Name, t.File, t.Line, t.Doc, 1, nil)
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].score != cands[j].score {
			return cands[i].score > cands[j].score
		}
		return cands[i].hit.What < cands[j].hit.What
	})
	if len(cands) > 8 {
		cands = cands[:8]
	}
	hits := make([]ChangeHit, 0, len(cands))
	for _, c := range cands {
		h := c.hit
		if c.fn != nil && len(c.fn.Callers) > 0 {
			n := len(c.fn.Callers)
			callers := "callers"
			if n == 1 {
				callers = "caller"
			}
			h.Why += fmt.Sprintf("; %d %s", n, callers)
		}
		hits = append(hits, h)
	}
	return hits
}

// WhereToChangeText renders the hits for chat output.
func (p *Project) WhereToChangeText(task string) string {
	hits := p.WhereToChange(task)
	var b strings.Builder
	fmt.Fprintf(&b, "WHERE TO CHANGE for %q\n", task)
	if len(hits) == 0 {
		b.WriteString("No strong matches — try different keywords, or run a full analyze first.\n")
		return b.String()
	}
	for i, h := range hits {
		fmt.Fprintf(&b, "%d. %s  %s:%d\n   (%s)\n", i+1, h.What, h.File, h.Line, h.Why)
	}
	return b.String()
}

// Display renders "pkg.Name" or "pkg.Recv.Name".
func (fn *Func) Display() string {
	if fn.Receiver != "" {
		return fn.Pkg + "." + fn.Receiver + "." + fn.Name
	}
	return fn.Pkg + "." + fn.Name
}

func filepathBase(p string) string {
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[i+1:]
	}
	return p
}
