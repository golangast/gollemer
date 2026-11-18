package analyze

// beginner_rank.go — the beginner-first ordering and the
// non-circular descriptions behind the tour's concept stop,
// /learn-path, and learn mode.
//
// LearnPath (beginner.go) orders types by structural centrality:
// the most-depended-on types first. In any framework those are
// plumbing — a builder, a logger seam, a platform adapter — and
// a beginner meets the wiring before the front door. This file
// adds a second ordering computed from the same analyzer facts:
//
//   - how close the program's entry points come to the type
//     (breadth-first over the call graph; a type is "touched" by
//     a function when its name, or its New<Name> constructor,
//     appears in that function's signature or source),
//   - whether an exported package-level function hands it out
//     (the project's public face),
//   - whether tests touch it,
//   - how many different packages reference it.
//
// and ranks DOWN, out of the first screen:
//
//   - types living under an internal/ package segment,
//   - types in deep packages (3+ path segments),
//   - islands: types referenced only from their own package (or
//     never referenced at all) and never near an entry point,
//   - adapter seams: interfaces whose implementations, computed
//     project-wide by method sets, sit in two or more
//     subpackages of the interface's own package (platform
//     adapters fan out exactly like this),
//   - tiny interfaces (≤2 methods) nobody in this project
//     implements and no entry point touches.
//
// Every signal is structural — package paths, callers, method
// sets. No name lists, no project-specific strings.
//
// The descriptions here replace doc-comment echoes that merely
// restate the type's name ("Builder defines a builder that can
// build framekit applications"). lowInformation detects them;
// when a doc line carries no new information, the description is
// built from facts instead — what a struct holds, what an
// interface's methods let you do, and the most front-door
// function that uses the type — or, when no facts support even
// that, the type's name, kind, and home package. A purpose is
// never invented.

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// typeRefFacts are the reference facts one type accumulates from
// the whole project: who mentions it, from how far away, and
// where its implementations live.
type typeRefFacts struct {
	refs      int             // functions mentioning the type
	refPkgs   map[string]bool // their package dirs
	minDepth  int             // BFS depth of the closest referencing func; -1 = none reachable / no refs
	ctorTouch bool            // an exported package-level func mentions it
	testTouch bool            // a _test.go func mentions it
	internal  bool            // its package dir has an "internal" segment
	pkgDepth  int             // segments in its package dir ("" = 0)
	island    bool            // referenced only from its own package, or never referenced
	impls     int             // project-wide implementers (interfaces)
	fanout    bool            // implementers in ≥2 distinct subpackages of its own package
	bestRef   *Func           // the most front-door referencing func
}

// typeFacts computes reference facts for the given candidate
// types in one pass over the project's functions.
func (p *Project) typeFacts(cands []*Type) map[string]*typeRefFacts {
	// Breadth-first depth from the entry points over Calls.
	depth := map[string]int{}
	var queue []*Func
	for _, e := range p.EntryPoints() {
		if _, ok := depth[e.ID]; !ok {
			depth[e.ID] = 0
			queue = append(queue, e)
		}
	}
	for len(queue) > 0 {
		fn := queue[0]
		queue = queue[1:]
		for _, calleeID := range fn.Calls {
			if _, ok := depth[calleeID]; ok {
				continue
			}
			callee := p.FuncByID(calleeID)
			if callee == nil {
				continue
			}
			depth[calleeID] = depth[fn.ID] + 1
			queue = append(queue, callee)
		}
	}

	facts := map[string]*typeRefFacts{}
	matchers := map[string][]*regexp.Regexp{}
	for _, t := range cands {
		key := t.PkgDir + "\x00" + t.Name
		f := &typeRefFacts{refPkgs: map[string]bool{}, minDepth: -1}
		for _, seg := range strings.Split(t.PkgDir, "/") {
			if seg == "internal" {
				f.internal = true
			}
		}
		if t.PkgDir != "" {
			f.pkgDepth = len(strings.Split(t.PkgDir, "/"))
		}
		facts[key] = f
		name := regexp.QuoteMeta(t.Name)
		matchers[key] = []*regexp.Regexp{
			regexp.MustCompile(`\b` + name + `\b`),
			regexp.MustCompile(`\bNew` + name + `\b`),
		}
	}

	for _, fn := range p.AllFuncs() {
		hay := fn.Sig + "\n" + fn.Src
		for _, t := range cands {
			key := t.PkgDir + "\x00" + t.Name
			hit := false
			for _, re := range matchers[key] {
				if re.MatchString(hay) {
					hit = true
					break
				}
			}
			if !hit {
				continue
			}
			f := facts[key]
			f.refs++
			f.refPkgs[fn.PkgDir] = true
			if fn.Receiver == "" && fn.Exported {
				f.ctorTouch = true
			}
			if fn.IsTest {
				f.testTouch = true
			}
			if d, ok := depth[fn.ID]; ok && (f.minDepth < 0 || d < f.minDepth) {
				f.minDepth = d
			}
			if betterRef(f.bestRef, fn, depth) {
				f.bestRef = fn
			}
		}
	}

	for _, t := range cands {
		f := facts[t.PkgDir+"\x00"+t.Name]
		// Island: nobody outside its own package machinery
		// references it (or nobody references it at all).
		f.island = true
		for dir := range f.refPkgs {
			if dir != t.PkgDir {
				f.island = false
				break
			}
		}
		if t.Kind != "interface" || len(t.Methods) == 0 {
			continue
		}
		// Implementers, project-wide, by method set.
		implDirs := map[string]bool{}
		for _, pkg := range p.Packages {
			for _, s := range pkg.Types {
				if s.Kind != "struct" || (s.Name == t.Name && s.PkgDir == t.PkgDir) {
					continue
				}
				have := map[string]bool{}
				for _, m := range p.methodsOf(s) {
					have[m] = true
				}
				ok := true
				for _, m := range t.Methods {
					if !have[m] {
						ok = false
						break
					}
				}
				if ok {
					f.impls++
					if strings.HasPrefix(s.PkgDir, t.PkgDir+"/") {
						implDirs[s.PkgDir] = true
					}
				}
			}
		}
		f.fanout = len(implDirs) >= 2
	}
	return facts
}

// betterRef reports whether candidate fn is a more front-door
// reference than the current best: closer to an entry point,
// then an exported package-level function, then a plain one.
func betterRef(best, fn *Func, depth map[string]int) bool {
	if best == nil {
		return true
	}
	bd, bok := depth[best.ID]
	fd, fok := depth[fn.ID]
	if fok != bok {
		return fok // reachable beats unreachable
	}
	if fok && fd != bd {
		return fd < bd
	}
	bScore := refKindScore(best)
	fScore := refKindScore(fn)
	if fScore != bScore {
		return fScore > bScore
	}
	return fn.Name < best.Name
}

func refKindScore(fn *Func) int {
	switch {
	case fn.IsMain:
		return 3
	case fn.Receiver == "" && fn.Exported && !fn.IsTest:
		return 2
	case !fn.IsTest:
		return 1
	default:
		return 0
	}
}

// score is the beginner-usefulness ranking: front-door facts
// add, plumbing signals subtract. See the file header for the
// signal list.
func (f *typeRefFacts) score(t *Type) int {
	s := 0
	switch f.minDepth {
	case 0:
		s += 4
	case 1:
		s += 3
	case 2:
		s += 1
	}
	if f.ctorTouch {
		s += 2
	}
	if f.testTouch {
		s += 1
	}
	if len(f.refPkgs) >= 2 {
		s += 1
	}
	if f.internal {
		s -= 4
	}
	if f.pkgDepth >= 3 {
		s -= 2
	}
	if f.island && f.minDepth != 0 {
		s -= 3
	}
	if f.fanout {
		s -= 2
	}
	if t.Kind == "interface" && len(t.Methods) <= 2 && f.impls == 0 && f.minDepth != 0 {
		s--
	}
	return s
}

// BeginnerConcepts returns the project's concepts in
// beginner-usefulness order: the types a newcomer should meet
// first. limit <= 0 returns them all; the full LearnPath still
// exists for the complete dependency-ordered walk. Candidates
// are the project's exported types (a wider pool than
// LearnPath's), scored by typeFacts; ties keep LearnPath's
// dependency order, then the name.
func (p *Project) BeginnerConcepts(limit int) []LearnConcept {
	var cands []*Type
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Exported {
				cands = append(cands, t)
			}
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if len(cands[i].Methods) != len(cands[j].Methods) {
			return len(cands[i].Methods) > len(cands[j].Methods)
		}
		return cands[i].Name < cands[j].Name
	})
	if len(cands) > 40 {
		cands = cands[:40]
	}
	concepts := p.buildConcepts(cands) // LearnPath (dependency) order
	facts := p.typeFacts(cands)
	rank := make([]int, len(concepts))
	for i := range concepts {
		rank[i] = i
	}
	sort.SliceStable(rank, func(a, b int) bool {
		ca, cb := concepts[rank[a]], concepts[rank[b]]
		sa := facts[ca.Type.PkgDir+"\x00"+ca.Type.Name].score(ca.Type)
		sb := facts[cb.Type.PkgDir+"\x00"+cb.Type.Name].score(cb.Type)
		if sa != sb {
			return sa > sb
		}
		return ca.Type.Name < cb.Type.Name
	})
	out := make([]LearnConcept, 0, len(rank))
	for _, i := range rank {
		out = append(out, concepts[i])
	}
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

// ---- non-circular descriptions ----

// docScaffold words carry sentence shape, not information:
// articles, copulas, and the verbs of "X defines an X…"
// doc-comment boilerplate.
var docScaffold = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true,
	"defines": true, "define": true, "defined": true,
	"represents": true, "represent": true,
	"provides": true, "provide": true, "provided": true,
	"type": true, "types": true, "struct": true,
	"interface": true, "kind": true,
	"that": true, "which": true, "can": true, "could": true,
	"do": true, "does": true, "did": true,
	"what": true, "with": true, "for": true, "of": true,
	"to": true, "in": true, "on": true, "by": true, "at": true,
	"from": true, "as": true, "and": true, "or": true,
	"it": true, "its": true, "this": true, "these": true,
	"those": true, "used": true, "using": true,
}

// camelParts splits a Go identifier into its humps:
// "CustomLogger" → [Custom Logger], "GtkMenu" → [Gtk Menu],
// "Dbusmenuer" → [Dbusmenuer].
func camelParts(name string) []string {
	var parts []string
	start := 0
	runes := []rune(name)
	for i := 1; i < len(runes); i++ {
		boundary := false
		switch {
		case isUpperRune(runes[i]) && !isUpperRune(runes[i-1]):
			boundary = true
		case isUpperRune(runes[i]) && i+1 < len(runes) && !isUpperRune(runes[i+1]):
			boundary = true // acronym end: "HTTPServer" → HTTP|Server
		}
		if boundary {
			parts = append(parts, string(runes[start:i]))
			start = i
		}
	}
	return append(parts, string(runes[start:]))
}

func isUpperRune(r rune) bool { return r >= 'A' && r <= 'Z' }

func commonPrefixLen(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// lowInformation reports whether a doc sentence says anything
// beyond the type's own name: after dropping scaffolding, the
// name's own words (and ≥4-letter stems of them — "build" for
// "Builder"), ambient project words, and opaque dotted tokens
// (com.canonical.dbusmenu), at most one content word may
// remain. "Container holds everything" keeps two (holds,
// everything) and passes; "Builder defines a builder that can
// build framekit applications" keeps none that are new.
func lowInformation(sentence, typeName string, ambient []string) bool {
	nameWords := map[string]bool{strings.ToLower(typeName): true}
	for _, part := range camelParts(typeName) {
		nameWords[strings.ToLower(part)] = true
	}
	amb := map[string]bool{}
	for _, w := range ambient {
		amb[strings.ToLower(w)] = true
	}
	fresh := 0
	for _, tok := range strings.FieldsFunc(sentence, func(r rune) bool {
		return r == ' ' || r == '\t' || r == ',' || r == ':' || r == ';' ||
			r == '(' || r == ')' || r == '"' || r == '“' || r == '”'
	}) {
		w := strings.ToLower(strings.TrimSuffix(tok, "."))
		if w == "" || docScaffold[w] || nameWords[w] || amb[w] {
			continue
		}
		if strings.Contains(w, ".") {
			continue // an opaque dotted identifier, not an explanation
		}
		stem := false
		for part := range nameWords {
			if len(part) >= 4 && commonPrefixLen(w, part) >= 4 {
				stem = true
				break
			}
		}
		if !stem {
			fresh++
		}
	}
	return fresh <= 1
}

// docOneLiner returns the doc's first sentence when it carries
// real information beyond the name; ambient lists extra
// project words (the module's own name) that don't count as
// information either.
func docOneLiner(t *Type, ambient []string) (string, bool) {
	if t.Doc == "" {
		return "", false
	}
	s := PieceFirstSentence(t.Doc)
	if s == "" || lowInformation(s, t.Name, ambient) {
		return "", false
	}
	return s, true
}

// projectAmbient is the project's own name, in words: the
// module path's last segment and its humps.
func (p *Project) projectAmbient() []string {
	if p.Module == "" {
		return nil
	}
	tail := p.Module
	if i := strings.LastIndex(tail, "/"); i >= 0 {
		tail = tail[i+1:]
	}
	out := []string{tail}
	out = append(out, camelParts(tail)...)
	return out
}

// shapeDesc is the no-doc fallback, built from the type's own
// declaration: what a struct holds, what an interface can do.
func shapeDesc(t *Type) string {
	switch t.Kind {
	case "struct":
		if len(t.Fields) > 0 {
			n := len(t.Fields)
			if n > 4 {
				n = 4
			}
			return "a struct holding " + strings.Join(t.Fields[:n], ", ")
		}
		return "a struct"
	case "interface":
		if len(t.Methods) > 0 {
			n := len(t.Methods)
			if n > 3 {
				n = 3
			}
			return "an interface — anything with " + strings.Join(t.Methods[:n], ", ") + " counts"
		}
		return "an interface"
	default:
		return "a " + t.Kind + " type"
	}
}

// pkgHome phrases where a type lives, for the no-facts case.
func pkgHome(t *Type) string {
	if t.Pkg != "" {
		return "package " + t.Pkg
	}
	if t.PkgDir != "" {
		return "package " + t.PkgDir
	}
	return "the root package"
}

// structuralDesc builds a concept description from facts only:
// the declaration's shape, plus — when a front-door function
// uses the type — who that is. When the shape itself has
// nothing to say (no fields, no methods), it falls back to
// name + kind + home rather than inventing a sentence.
func (p *Project) structuralDesc(t *Type, f *typeRefFacts) string {
	var d string
	switch t.Kind {
	case "struct":
		switch {
		case len(t.Fields) > 0:
			n := len(t.Fields)
			more := ""
			if n > 4 {
				n = 4
				more = ", …"
			}
			d = "a struct holding " + strings.Join(t.Fields[:n], ", ") + more
		case len(t.Methods) > 0:
			n := len(t.Methods)
			if n > 3 {
				n = 3
			}
			d = "a struct with methods " + strings.Join(t.Methods[:n], ", ")
		default:
			d = "a struct in " + pkgHome(t)
		}
	case "interface":
		if len(t.Methods) > 0 {
			n := len(t.Methods)
			more := ""
			if n > 3 {
				n = 3
				more = ", …"
			}
			d = "an interface — anything that can " + strings.Join(t.Methods[:n], ", ") + more + " counts"
		} else {
			d = "an interface in " + pkgHome(t)
		}
	default:
		d = "a " + t.Kind + " type in " + pkgHome(t)
	}
	if f != nil && f.bestRef != nil && f.minDepth >= 0 && f.minDepth <= 1 {
		if f.bestRef.IsMain {
			d += " — used where the program starts"
		} else {
			d += " — used by " + f.bestRef.Display()
		}
	}
	return d
}

// conceptOneLiner is the description LearnPath and
// BeginnerConcepts print: the doc's own sentence when it earns
// its place, the structural description when it doesn't.
func (p *Project) conceptOneLiner(t *Type, f *typeRefFacts) string {
	desc, ok := docOneLiner(t, p.projectAmbient())
	if !ok {
		desc = p.structuralDesc(t, f)
		return desc
	}
	if len(t.Methods) > 0 && t.Kind != "interface" {
		desc += " (" + strconv.Itoa(len(t.Methods)) + " methods)"
	}
	return desc
}
