package analyze

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/golangast/gollemer/pkg/termui"
)

// Beginner-understanding reports (2026-10-20): five views built for
// someone meeting a project for the first time. Definitions rarely
// teach; examples, order, and watching the thing run do.

// sourceLines reads lines [from, to] (1-based, inclusive) of a project
// file. Returns nil when the file can't be read.
func (p *Project) sourceLines(file string, from, to int) []string {
	data, err := os.ReadFile(filepath.Join(p.Root, file))
	if err != nil {
		return nil
	}
	lines := strings.Split(string(data), "\n")
	if from < 1 {
		from = 1
	}
	if to > len(lines) {
		to = len(lines)
	}
	if from > to {
		return nil
	}
	return lines[from-1 : to]
}

// callWindow returns up to 3 source lines of caller centered on the
// first line that calls fnName, with the call line marked.
func (p *Project) callWindow(caller *Func, fnName string) (text string, line int) {
	lines := p.sourceLines(caller.File, caller.Line, caller.EndLine)
	needle := fnName + "("
	for i, l := range lines {
		if strings.Contains(l, needle) {
			lo := i - 1
			if lo < 0 {
				lo = 0
			}
			hi := i + 1
			if hi >= len(lines) {
				hi = len(lines) - 1
			}
			var b strings.Builder
			for j := lo; j <= hi; j++ {
				mark := "  "
				if j == i {
					mark = "→ "
				}
				fmt.Fprintf(&b, "    %s%s\n", mark, strings.TrimRight(lines[j], " \t"))
			}
			return b.String(), caller.Line + i
		}
	}
	return "", 0
}

// ExamplesText shows a function the way a beginner actually learns it:
// real call sites with the surrounding code, and a test that exercises
// it. When nothing calls it, it says why that might be.
func (p *Project) ExamplesText(fn *Func) string {
	var b strings.Builder
	b.WriteString(termui.SubHeader(fmt.Sprintf("How %s is really used", fn.Name)) + "\n")
	var realCallers, testCallers []*Func
	for _, id := range fn.Callers {
		c := p.FuncByID(id)
		if c == nil || c.ID == fn.ID {
			continue
		}
		if c.IsTest {
			testCallers = append(testCallers, c)
		} else {
			realCallers = append(realCallers, c)
		}
	}
	if len(realCallers) == 0 && len(testCallers) == 0 {
		switch {
		case fn.IsMain:
			b.WriteString("main is the entry point — nothing calls it; the operating system starts here.\n")
		case fn.Exported:
			b.WriteString("Nothing inside this project calls it. It's exported (capital letter), so other projects call it from outside — it's part of the public surface.\n")
		default:
			b.WriteString("Nothing in this project calls it. It may be dead code, or reached indirectly (through an interface, a callback, or reflection) in a way a static scan can't see.\n")
		}
		return b.String()
	}
	fmt.Fprintf(&b, "%s is called from %d place(s)", fn.Name, len(realCallers))
	if len(testCallers) > 0 {
		fmt.Fprintf(&b, ", and exercised by %d test(s)", len(testCallers))
	}
	b.WriteString(".\n")
	shown := 0
	for _, c := range realCallers {
		if shown >= 2 {
			fmt.Fprintf(&b, "  … and %d more caller(s) — try /impact %s for the full list.\n", len(realCallers)-shown, fn.Name)
			break
		}
		win, line := p.callWindow(c, fn.Name)
		if win == "" {
			continue
		}
		fmt.Fprintf(&b, "\nCalled by %s %s:\n%s", termui.Bright.Apply(shortID(c.ID)), termui.StyleFileLine(c.File, line), win)
		if c.Doc != "" {
			if s := PieceFirstSentence(c.Doc); s != "" {
				fmt.Fprintf(&b, "  That caller: %s\n", s)
			}
		}
		shown++
	}
	if len(testCallers) > 0 {
		t := testCallers[0]
		fmt.Fprintf(&b, "\nIn a test — %s %s:\n", termui.Bright.Apply(t.Name), termui.StyleFileLine(t.File, t.Line))
		for _, l := range p.sourceLines(t.File, t.Line, t.Line+11) {
			fmt.Fprintf(&b, "    %s\n", strings.TrimRight(l, " \t"))
		}
		b.WriteString("  Tests are the best examples: this is " + fn.Name + " being used for real, with real inputs.\n")
	}
	return b.String()
}

// typeOneLiner renders one glossary line for a type: its doc's
// first sentence when that sentence actually says something
// (see lowInformation in beginner_rank.go), otherwise a line
// derived from its shape.
func typeOneLiner(t *Type) string {
	desc, ok := docOneLiner(t, nil)
	if !ok {
		desc = shapeDesc(t)
	}
	if len(t.Methods) > 0 && t.Kind != "interface" {
		desc += fmt.Sprintf(" (%d methods)", len(t.Methods))
	}
	return desc
}

// GlossaryText builds the project's dictionary: the types a newcomer
// must know, one plain line each, most important first.
func (p *Project) GlossaryText() string {
	types := p.KeyTypes(18)
	var b strings.Builder
	b.WriteString(termui.SubHeader("The words this project uses — learn these and the rest starts to make sense:") + "\n\n")
	if len(types) == 0 {
		b.WriteString("No types stood out — this project is mostly functions.\n")
		return b.String()
	}
	nameCount := map[string]int{}
	for _, t := range types {
		nameCount[t.Name]++
	}
	for _, t := range types {
		name := t.Name
		if nameCount[t.Name] > 1 {
			dir := t.PkgDir
			if dir == "" {
				dir = "root"
			}
			name = fmt.Sprintf("%s (in %s)", t.Name, dir)
		}
		fmt.Fprintf(&b, "  %s%s\n", termui.Bright.Apply(name), termui.Muted.Apply(" — "+typeOneLiner(t)))
	}
	b.WriteString("\nAsk /whatis <word> about any of them for more.\n")
	return b.String()
}

var (
	reGoRoutine = regexp.MustCompile(`\bgo (func\b|[a-zA-Z_][\w.]*\()`)
	reGeneric   = regexp.MustCompile(`\[[A-Z]\w* (any|comparable)\]`)
)

// goFeatures names the Go ideas a chunk of code uses, each with the
// gobyexample page that teaches it.
func goFeatures(text string, isInterface bool) [][2]string {
	var out [][2]string
	seen := map[string]bool{}
	add := func(name, slug string) {
		if !seen[name] {
			seen[name] = true
			out = append(out, [2]string{name, "https://gobyexample.com/" + slug})
		}
	}
	if strings.Contains(text, "sync.Mutex") || strings.Contains(text, "sync.RWMutex") || strings.Contains(text, ".Lock()") {
		add("mutexes", "mutexes")
	}
	if strings.Contains(text, "chan ") || strings.Contains(text, "make(chan") {
		add("channels", "channels")
	}
	if reGoRoutine.MatchString(text) {
		add("goroutines", "goroutines")
	}
	if isInterface {
		add("interfaces", "interfaces")
	}
	if strings.Contains(text, " error") {
		add("errors", "errors")
	}
	if reGeneric.MatchString(text) {
		add("generics", "generics")
	}
	return out
}

// LearnConcept is one step of the project's learn path: an exported
// type, the concepts it builds on, and the Go features its methods
// use (each with the gobyexample page that teaches it). LearnPathText
// renders these; the chat's learn mode walks them one by one.
type LearnConcept struct {
	Type     *Type
	Name     string      // display name (package-disambiguated like the text path)
	OneLiner string      // the type in one plain line
	BuildsOn []string    // display names of the concepts this one uses, sorted
	Features [][2]string // Go ideas it uses: (name, gobyexample URL)
}

// LearnPath returns the project's concepts ordered so each one only
// uses ideas introduced earlier — the structured form of
// LearnPathText.
func (p *Project) LearnPath() []LearnConcept {
	// Concepts: exported types, most methods first, capped.
	var concepts []*Type
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Exported {
				concepts = append(concepts, t)
			}
		}
	}
	sort.Slice(concepts, func(i, j int) bool {
		if len(concepts[i].Methods) != len(concepts[j].Methods) {
			return len(concepts[i].Methods) > len(concepts[j].Methods)
		}
		return concepts[i].Name < concepts[j].Name
	})
	if len(concepts) > 15 {
		concepts = concepts[:15]
	}
	return p.buildConcepts(concepts)
}

// buildConcepts assembles LearnConcepts for a candidate set:
// dependency order (Kahn, alphabetical on ties), structural
// dependencies, Go features, and the non-circular one-liner
// each. LearnPath and BeginnerConcepts share it; they differ
// only in candidate selection and final ordering.
func (p *Project) buildConcepts(concepts []*Type) []LearnConcept {
	if len(concepts) == 0 {
		return nil
	}
	facts := p.typeFacts(concepts)
	// Concepts are keyed by package+name: two packages may both define
	// a "Container", and they are different concepts.
	keyOf := func(t *Type) string { return t.PkgDir + "\x00" + t.Name }
	nameCount := map[string]int{}
	for _, t := range concepts {
		nameCount[t.Name]++
	}
	display := func(t *Type) string {
		if nameCount[t.Name] > 1 {
			dir := t.PkgDir
			if dir == "" {
				dir = "root"
			}
			return fmt.Sprintf("%s (in %s)", t.Name, dir)
		}
		return t.Name
	}
	// Dependencies: a concept depends on another when its methods'
	// signatures or bodies mention the other type's name.
	deps := map[string]map[string]bool{}
	features := map[string][][2]string{}
	byKey := map[string]*Type{}
	for _, t := range concepts {
		byKey[keyOf(t)] = t
		deps[keyOf(t)] = map[string]bool{}
		var text strings.Builder
		for _, fn := range p.AllFuncs() {
			if strings.Trim(fn.Receiver, "*") == t.Name && fn.PkgDir == t.PkgDir {
				text.WriteString(fn.Sig + "\n" + fn.Src + "\n")
			}
		}
		body := text.String()
		for _, other := range concepts {
			if keyOf(other) == keyOf(t) {
				continue
			}
			if regexp.MustCompile(`\b` + regexp.QuoteMeta(other.Name) + `\b`).MatchString(body) {
				deps[keyOf(t)][keyOf(other)] = true
			}
		}
		features[keyOf(t)] = goFeatures(body, t.Kind == "interface")
	}
	// Kahn's algorithm, alphabetical on ties; cycles fall back to
	// alphabetical for whatever remains.
	var order []*Type
	done := map[string]bool{}
	for len(order) < len(concepts) {
		var ready []*Type
		for _, t := range concepts {
			if done[keyOf(t)] {
				continue
			}
			ok := true
			for d := range deps[keyOf(t)] {
				if !done[d] {
					ok = false
					break
				}
			}
			if ok {
				ready = append(ready, t)
			}
		}
		if len(ready) == 0 { // cycle: take the rest alphabetically
			for _, t := range concepts {
				if !done[keyOf(t)] {
					ready = append(ready, t)
				}
			}
		}
		sort.Slice(ready, func(i, j int) bool { return display(ready[i]) < display(ready[j]) })
		next := ready[0]
		done[keyOf(next)] = true
		order = append(order, next)
	}
	out := make([]LearnConcept, 0, len(order))
	for _, t := range order {
		var ds []string
		for d := range deps[keyOf(t)] {
			ds = append(ds, display(byKey[d]))
		}
		sort.Strings(ds)
		out = append(out, LearnConcept{
			Type:     t,
			Name:     display(t),
			OneLiner: p.conceptOneLiner(t, facts[keyOf(t)]),
			BuildsOn: ds,
			Features: features[keyOf(t)],
		})
	}
	return out
}

// LearnPathText renders LearnPath: the project's concepts ordered so
// each one only uses ideas introduced earlier, with the Go features
// each concept leans on linked to gobyexample.
func (p *Project) LearnPathText() string {
	order := p.LearnPath()
	if len(order) == 0 {
		return "No exported types to order — this project's concepts live in its functions. Start with /tour.\n"
	}
	var b strings.Builder
	b.WriteString(termui.SubHeader("Learn this project in this order — each step only uses ideas from the steps before it:") + "\n\n")
	for i, c := range order {
		fmt.Fprintf(&b, "%s %s%s\n", termui.Gold.Apply(fmt.Sprintf("%d.", i+1)),
			termui.Bright.Apply(c.Name), termui.Muted.Apply(" — "+c.OneLiner))
		if len(c.BuildsOn) > 0 {
			fmt.Fprintf(&b, "   %s %s.\n", termui.Muted.Apply("Builds on:"), strings.Join(c.BuildsOn, ", "))
		}
		if len(c.Features) > 0 {
			var parts []string
			for j, f := range c.Features {
				if j >= 2 {
					break
				}
				parts = append(parts, f[0]+" ("+f[1]+")")
			}
			fmt.Fprintf(&b, "   %s %s.\n", termui.Muted.Apply("Go ideas it uses:"), strings.Join(parts, ", "))
		}
	}
	return b.String()
}

// TourStops builds a paced walk through the project: where it starts,
// what to read first, the words it uses, its engine room, its shape,
// and its tests. The chat serves one stop at a time.
func (p *Project) TourStops() []string {
	var stops []string
	if eps := p.EntryPoints(); len(eps) > 0 {
		fn := eps[0]
		var s strings.Builder
		fmt.Fprintf(&s, "Start where the program starts: %s %s.", termui.Bright.Apply(fn.Name), termui.StyleFileLine(fn.File, fn.Line))
		if fn.Doc != "" {
			if d := PieceFirstSentence(fn.Doc); d != "" {
				fmt.Fprintf(&s, " %s", d)
			}
		}
		if len(fn.Calls) > 0 {
			var names []string
			for i, id := range fn.Calls {
				if i >= 4 {
					break
				}
				names = append(names, shortID(id))
			}
			fmt.Fprintf(&s, " First thing it does is call: %s.", strings.Join(names, ", "))
		}
		stops = append(stops, s.String())
	}
	if chain := p.readingChain(5); len(chain) > 0 {
		stops = append(stops, "Read the packages in this order: "+strings.Join(chain, " → ")+
			". Dependencies come first, so nothing you read uses words you haven't met yet.")
	}
	if types := p.KeyTypes(3); len(types) > 0 {
		var s strings.Builder
		s.WriteString(termui.SubHeader("The cast — the types everything revolves around:"))
		for _, t := range types {
			fmt.Fprintf(&s, "\n  %s%s", termui.Bright.Apply(t.Name), termui.Muted.Apply(" — "+typeOneLiner(t)))
		}
		stops = append(stops, s.String())
	}
	if tops := p.TopFuncs(3); len(tops) > 0 {
		var s strings.Builder
		s.WriteString(termui.SubHeader("The engine room — the most-called functions:"))
		for _, fn := range tops {
			fmt.Fprintf(&s, "\n  %s %s", termui.Bright.Apply(shortID(fn.ID)),
				termui.Muted.Apply("(called from ")+termui.Gold.Apply(fmt.Sprintf("%d", len(fn.Callers)))+termui.Muted.Apply(" places)"))
		}
		s.WriteString("\nUnderstand these and you understand what the program spends its time doing.")
		stops = append(stops, s.String())
	}
	{
		indeg := map[string]int{}
		for _, pkg := range p.Packages {
			for _, d := range pkg.Internal {
				indeg[d]++
			}
		}
		type hub struct {
			dir string
			n   int
		}
		var hubs []hub
		for d, n := range indeg {
			hubs = append(hubs, hub{d, n})
		}
		sort.Slice(hubs, func(i, j int) bool { return hubs[i].n > hubs[j].n })
		var s strings.Builder
		fmt.Fprintf(&s, "The shape: %d packages, %d files, %d lines.", len(p.Packages), p.Files, p.Lines)
		if len(hubs) > 0 {
			var names []string
			for i, h := range hubs {
				if i >= 3 {
					break
				}
				names = append(names, h.dir)
			}
			fmt.Fprintf(&s, " The hub packages (most imported) are %s — /arch draws the whole map.", strings.Join(names, ", "))
		}
		stops = append(stops, s.String())
	}
	{
		tests, files := 0, map[string]bool{}
		for _, fn := range p.AllFuncs() {
			if fn.IsTest {
				tests++
				files[fn.File] = true
			}
		}
		if tests > 0 {
			stops = append(stops, fmt.Sprintf(
				"There are %d tests in %d files. Tests are the best examples — pick any function and ask /example <name> to see it used for real, with real inputs.",
				tests, len(files)))
		}
	}
	stops = append(stops, "That's the tour. Good next steps: /glossary for the project's words, /learn-path for what to learn first, /watch to see it run.")
	// Welcome goes first, with the final stop count.
	module := p.Module
	if module == "" {
		module = "this project"
	}
	welcome := fmt.Sprintf("Welcome to %s — %s files, %s lines, %s packages. I'll walk you through it one stop at a time; say 'next' whenever you're ready.",
		termui.Bright.Apply(module),
		termui.Gold.Apply(fmt.Sprintf("%d", p.Files)),
		termui.Gold.Apply(fmt.Sprintf("%d", p.Lines)),
		termui.Gold.Apply(fmt.Sprintf("%d", len(p.Packages))))
	return append([]string{welcome}, stops...)
}
