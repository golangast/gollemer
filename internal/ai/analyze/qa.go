package analyze

// This file answers natural-language questions about an analyzed project:
// "what does routeDomain do", "how does NewTensor work", "what calls X",
// "show me X", "what's in package chat". It is the layer that lets a new
// programmer interrogate a codebase instead of just reading a static map.
//
// Answer only fires when the question names a symbol that actually exists
// in the project, so the chat layer can safely check it before routing:
// it can never steal a question meant for another brain.

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

var (
	qCallees  = regexp.MustCompile(`(?i)^\s*(?:so\s+)?what does (?:the )?(.+?) call\??$`)
	qWhatDoes = regexp.MustCompile(`(?i)^\s*(?:so\s+)?what does (?:the )?(.+?) do\??$`)
	qWhatIs   = regexp.MustCompile(`(?i)^\s*what(?:'s| is) (?:the )?(.+?)\??$`)
	qHowWorks = regexp.MustCompile(`(?i)^\s*how does (?:the )?(.+?) work\??$`)
	qWhereDef = regexp.MustCompile(`(?i)^\s*where is (?:the )?(.+?) (?:defined|declared|located)\??$`)
	qCallers  = regexp.MustCompile(`(?i)^\s*(?:what|who) (?:calls|uses) (?:the )?(.+?)\??$`)
	qShowMe   = regexp.MustCompile(`(?i)^\s*show me (?:the )?(?:code for |source of |source code for )?(.+?)\.?$`)
	qExplain  = regexp.MustCompile(`(?i)^\s*explain (?:the )?(.+?)\.?$`)
	qTellAbt  = regexp.MustCompile(`(?i)^\s*tell me about (?:the )?(.+?)\.?$`)
	qPkgWhats = regexp.MustCompile(`(?i)^\s*(?:what'?s|what is) in (?:the )?(?:package )?(.+?)\??$`)
	qPkgList  = regexp.MustCompile(`(?i)^\s*(?:list|show)(?: me)? (?:all )?(?:the )?functions in (?:the )?(?:package )?(.+?)\.?$`)
	qWhereHdl = regexp.MustCompile(`(?i)^\s*where is (.+?) (?:handled|done|implemented)\??$`)
	qWhereDo  = regexp.MustCompile(`(?i)^\s*where (?:do|would|could) i (.+?)\??$`)
	// Value-flow and effect questions, answered from the general index —
	// no per-feature logic: "what reads X", "where is the filesystem
	// written", "where does it use the network".
	qWhatReads = regexp.MustCompile(`(?i)^\s*(?:so\s+)?what reads (?:the )?(.+?)\??$`)
	qWhereUsed = regexp.MustCompile(`(?i)^\s*where is (?:the )?(.+?) (?:read|used)\??$`)
	qFsWrite   = regexp.MustCompile(`(?i)^\s*where\b(?:.*\b(?:files?|filesystem)\b.*\b(?:writ|touch|creat|delet|chang|modif)\w*|.*\b(?:writ|touch|creat|delet|chang|modif)\w*\b.*\b(?:files?|filesystem)\b)`)
	qNetUse    = regexp.MustCompile(`(?i)^\s*where\b.*?\b(?:network|internet)\b`)
	// Beginner walkthroughs, powered by the xray intelligence engine:
	// "walk me through routeDomain", "explain routeDomain for beginners".
	// Checked before qExplain/qWhatDoes so the plain "explain X" forms
	// keep their reference answers.
	qWalkThru = regexp.MustCompile(`(?i)^\s*walk(?: me)? through (?:the )?(.+?)\.?$`)
	qTraceSym = regexp.MustCompile(`(?i)^\s*trace (?:the )?(.+?)\.?$`)
	qBegFor   = regexp.MustCompile(`(?i)^\s*explain (?:the )?(.+?) for beginners?\.?$`)
	qBegLike  = regexp.MustCompile(`(?i)^\s*explain (?:the )?(.+?) like i['’]?m a beginner\.?$`)
)

// Answer answers a natural-language question about the project. It
// returns ok=false when the question doesn't reference anything in the
// project, so callers can fall through to normal routing.
func (p *Project) Answer(q string) (out string, ok bool) {
	// Beginner walkthroughs first: the xray engine's plain-words view.
	// On failure they fall back to the reference answer below.
	if m := qWalkThru.FindStringSubmatch(q); m != nil {
		return p.answerBeginner(cleanSymbol(m[1]))
	}
	if m := qTraceSym.FindStringSubmatch(q); m != nil {
		return p.answerBeginner(cleanSymbol(m[1]))
	}
	if m := qBegFor.FindStringSubmatch(q); m != nil {
		return p.answerBeginner(cleanSymbol(m[1]))
	}
	if m := qBegLike.FindStringSubmatch(q); m != nil {
		return p.answerBeginner(cleanSymbol(m[1]))
	}
	// Most-specific patterns first: "what does X call" before "what does X do".
	if m := qCallees.FindStringSubmatch(q); m != nil {
		return p.answerCallees(cleanSymbol(m[1]))
	}
	if m := qCallers.FindStringSubmatch(q); m != nil {
		return p.answerCallers(cleanSymbol(m[1]))
	}
	if m := qWhatReads.FindStringSubmatch(q); m != nil {
		return p.answerWhatReads(cleanSymbol(m[1]))
	}
	if m := qWhereUsed.FindStringSubmatch(q); m != nil {
		return p.answerWhatReads(cleanSymbol(m[1]))
	}
	if qFsWrite.MatchString(q) {
		return p.answerEffects(fxWrite, "write to the filesystem")
	}
	if qNetUse.MatchString(q) {
		return p.answerEffects(fxNet, "use the network")
	}
	if m := qWhatDoes.FindStringSubmatch(q); m != nil {
		return p.answerWhatDoes(cleanSymbol(m[1]))
	}
	if m := qHowWorks.FindStringSubmatch(q); m != nil {
		return p.answerHowWorks(cleanSymbol(m[1]))
	}
	if m := qWhereDef.FindStringSubmatch(q); m != nil {
		return p.answerWhereDefined(cleanSymbol(m[1]))
	}
	if m := qShowMe.FindStringSubmatch(q); m != nil {
		if sym := cleanSymbol(m[1]); !isVisualWord(sym) {
			return p.answerShowMe(sym)
		}
	}
	if m := qExplain.FindStringSubmatch(q); m != nil {
		return p.answerWhatDoes(cleanSymbol(m[1]))
	}
	if m := qTellAbt.FindStringSubmatch(q); m != nil {
		return p.answerWhatDoes(cleanSymbol(m[1]))
	}
	// Package questions before bare "what is": "what's in package store"
	// must not be swallowed by the what-is pattern.
	if m := qPkgWhats.FindStringSubmatch(q); m != nil {
		return p.answerPackage(cleanSymbol(m[1]))
	}
	if m := qPkgList.FindStringSubmatch(q); m != nil {
		return p.answerPackage(cleanSymbol(m[1]))
	}
	if m := qWhatIs.FindStringSubmatch(q); m != nil {
		sym := cleanSymbol(m[1])
		if !startsWithArticle(sym) {
			return p.answerWhatDoes(sym)
		}
	}
	// "where is X handled" / "where do I X" with no symbol: guided change
	// location — one anchor, the pattern to imitate, and you-are-here.
	if m := qWhereHdl.FindStringSubmatch(q); m != nil {
		return p.GuideChange(m[1]), true
	}
	if m := qWhereDo.FindStringSubmatch(q); m != nil {
		if sym := cleanSymbol(m[1]); sym == "" || !p.knowsSymbol(sym) {
			return p.GuideChange(m[1]), true
		}
	}
	return "", false
}

// cleanSymbol strips articles, quotes, and role words from a symbol name.
// It never invents a symbol: a multi-word phrase that isn't a known
// "X function" / "X method" form is returned as-is and won't resolve,
// so e.g. "what does make chat do" can't become "chat".
func cleanSymbol(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, `"'().,?!`)
	s = strings.TrimPrefix(s, "the ")
	for _, w := range []string{" function", " func", " method", " type", " struct", " package"} {
		s = strings.TrimSuffix(s, w)
	}
	return strings.TrimSpace(strings.Trim(s, `"'().,?!`))
}

func startsWithArticle(s string) bool {
	lower := strings.ToLower(s)
	return strings.HasPrefix(lower, "a ") || strings.HasPrefix(lower, "an ") ||
		strings.HasPrefix(lower, "the ")
}

// isVisualWord keeps "show me a visual" out of the symbol path.
func isVisualWord(s string) bool {
	switch strings.ToLower(s) {
	case "visual", "graph", "diagram", "picture", "map":
		return true
	}
	return false
}

// knowsSymbol reports whether name resolves to anything in the project.
func (p *Project) knowsSymbol(name string) bool {
	fn, _, typ, pkg := p.resolve(name)
	return fn != nil || typ != nil || pkg != nil
}

// resolve finds a function, type, or package by name. Name may be bare
// ("NewTensor"), dotted ("tensor.NewTensor"), or any case.
func (p *Project) resolve(name string) (fn *Func, cands []*Func, typ *Type, pkg *Package) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, nil, nil, nil
	}
	lower := strings.ToLower(name)
	dotted := strings.Contains(name, ".")
	// Functions.
	var matches []*Func
	for _, f := range p.byID {
		if dotted {
			if strings.EqualFold(f.Display(), name) {
				return f, nil, nil, nil
			}
			continue
		}
		if strings.EqualFold(f.Name, name) {
			matches = append(matches, f)
		}
	}
	if len(matches) == 1 {
		return matches[0], nil, nil, nil
	}
	if len(matches) > 1 {
		sort.Slice(matches, func(i, j int) bool {
			if matches[i].Exported != matches[j].Exported {
				return matches[i].Exported
			}
			return matches[i].Score > matches[j].Score
		})
		return nil, matches, nil, nil
	}
	// Types.
	for _, pk := range p.Packages {
		for _, t := range pk.Types {
			if strings.EqualFold(t.Name, name) ||
				(dotted && strings.EqualFold(t.Pkg+"."+t.Name, name)) {
				return nil, nil, t, nil
			}
		}
	}
	// Packages, by name or directory.
	for _, pk := range p.Packages {
		if strings.EqualFold(pk.Name, name) || strings.EqualFold(pk.Dir, name) ||
			strings.EqualFold(shortDir(pk.Dir), lower) {
			return nil, nil, nil, pk
		}
	}
	return nil, nil, nil, nil
}

func shortDir(dir string) string {
	if i := strings.LastIndex(dir, "/"); i >= 0 {
		return dir[i+1:]
	}
	return dir
}

// answerWhatDoes handles "what does X do" / "what is X" / "explain X".
// answerBeginner runs the symbol through the xray intelligence engine
// for the beginner walkthrough. If that fails (unknown symbol, unreadable
// file), it falls back to the reference answer so the user still gets
// something useful, including the ambiguous-name disambiguation.
func (p *Project) answerBeginner(name string) (string, bool) {
	if out, ok := p.ExplainBeginner(name); ok {
		return out, true
	}
	return p.answerWhatDoes(name)
}

func (p *Project) answerWhatDoes(name string) (string, bool) {
	fn, cands, typ, pkg := p.resolve(name)
	switch {
	case fn != nil:
		return fn.Detail(p), true
	case typ != nil:
		return typ.Detail(p), true
	case pkg != nil:
		return p.PackageDetail(pkg), true
	case len(cands) > 0:
		return p.ambiguous(name, cands), true
	}
	return "", false
}

// answerHowWorks handles "how does X work": the detail card plus what it
// calls, each with a one-line summary — the call chain a newcomer needs.
func (p *Project) answerHowWorks(name string) (string, bool) {
	fn, cands, typ, pkg := p.resolve(name)
	if typ != nil {
		return typ.Detail(p), true
	}
	if pkg != nil {
		return p.PackageDetail(pkg), true
	}
	if len(cands) > 0 {
		return p.ambiguous(name, cands), true
	}
	if fn == nil {
		return "", false
	}
	var b strings.Builder
	b.WriteString(fn.Detail(p))
	if len(fn.Calls) > 0 {
		b.WriteString("\nHow it works, step by step (what it calls):\n")
		seen := map[string]bool{}
		n := 0
		for _, id := range fn.Calls {
			callee := p.byID[id]
			if callee == nil || seen[callee.ID] {
				continue
			}
			seen[callee.ID] = true
			fmt.Fprintf(&b, "  %d. %s — %s\n", n+1, callee.Display(), callee.Summary())
			n++
			if n >= 8 {
				break
			}
		}
	}
	return b.String(), true
}

// answerCallers handles "what calls X" / "who uses X".
func (p *Project) answerCallers(name string) (string, bool) {
	fn, cands, typ, _ := p.resolve(name)
	if typ != nil {
		// "who uses this type": methods + constructors are the answer.
		var b strings.Builder
		fmt.Fprintf(&b, "type %s.%s is used by its %d method(s):\n", typ.Pkg, typ.Name, len(p.methodsOf(typ)))
		for _, m := range limitStr(p.methodsOf(typ), 12) {
			fmt.Fprintf(&b, "  - %s\n", m)
		}
		return b.String(), true
	}
	if len(cands) > 0 {
		return p.ambiguous(name, cands), true
	}
	if fn == nil {
		return "", false
	}
	var b strings.Builder
	if len(fn.Callers) == 0 {
		fmt.Fprintf(&b, "Nothing in this project calls %s.\n", fn.Display())
		if !fn.IsMain {
			b.WriteString("It's either an entry point, dead code, or only called from outside.\n")
		}
		return b.String(), true
	}
	callers := append([]string{}, fn.Callers...)
	sort.Strings(callers)
	fmt.Fprintf(&b, "%s is called from %d place(s):\n", fn.Display(), len(callers))
	for i, c := range callers {
		if i >= 15 {
			fmt.Fprintf(&b, "  ... and %d more\n", len(callers)-15)
			break
		}
		if f := p.byID[c]; f != nil {
			fmt.Fprintf(&b, "  - %s  (%s:%d)\n", f.Display(), p.LinkPath(f.File), f.Line)
		} else {
			fmt.Fprintf(&b, "  - %s\n", shortID(c))
		}
	}
	return b.String(), true
}

// answerWhatReads handles "what reads X" / "where is X read|used": the
// value-flow index — every function that can observe the value, directly
// or through threaded calls. Works for any struct field in the project.
func (p *Project) answerWhatReads(name string) (string, bool) {
	keys := p.resolveField(name)
	if len(keys) == 0 {
		return "", false
	}
	seen := map[string]bool{}
	var ids []string
	for _, k := range keys {
		for _, id := range p.FieldReaders(k) {
			if !seen[id] {
				seen[id] = true
				ids = append(ids, id)
			}
		}
	}
	sort.Strings(ids)
	if len(ids) == 0 {
		return "", false
	}
	var b strings.Builder
	fmt.Fprintf(&b, "`%s` is read by %d function(s):\n", name, len(ids))
	for i, id := range ids {
		if i >= 12 {
			fmt.Fprintf(&b, "  ... and %d more\n", len(ids)-12)
			break
		}
		if f := p.byID[id]; f != nil {
			fmt.Fprintf(&b, "  - %s  (%s:%d)\n", f.Display(), p.LinkPath(f.File), f.Line)
		} else {
			fmt.Fprintf(&b, "  - %s\n", shortID(id))
		}
	}
	return b.String(), true
}

// resolveField maps a field name to "pkg.Type.Field" keys across every
// struct type in the project. A "--flag" prefix is stripped so "what
// reads --verbose" works too.
func (p *Project) resolveField(name string) []string {
	name = strings.Trim(name, "`\"' ")
	name = strings.TrimPrefix(name, "--")
	var keys []string
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Kind != "struct" {
				continue
			}
			for _, f := range t.Fields {
				if strings.EqualFold(f, name) {
					keys = append(keys, pkg.Name+"."+t.Name+"."+f)
				}
			}
		}
	}
	sort.Strings(keys)
	return keys
}

// answerEffects handles "where is the filesystem written" / "where does
// it use the network": every function with the effect category in its
// transitive effects.
func (p *Project) answerEffects(cat, label string) (string, bool) {
	fns := p.effectFuncs(cat, 12)
	if len(fns) == 0 {
		return "", false
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Functions that %s:\n", label)
	for _, fn := range fns {
		fmt.Fprintf(&b, "  - %s  (%s:%d)\n", fn.Display(), p.LinkPath(fn.File), fn.Line)
	}
	return b.String(), true
}

// answerCallees handles "what does X call".
func (p *Project) answerCallees(name string) (string, bool) {
	fn, cands, _, _ := p.resolve(name)
	if len(cands) > 0 {
		return p.ambiguous(name, cands), true
	}
	if fn == nil {
		return "", false
	}
	var b strings.Builder
	if len(fn.Calls) == 0 {
		fmt.Fprintf(&b, "%s doesn't call anything else in this project — it's a leaf function.\n", fn.Display())
		return b.String(), true
	}
	fmt.Fprintf(&b, "%s calls %d function(s):\n", fn.Display(), len(fn.Calls))
	seen := map[string]bool{}
	n := 0
	for _, id := range fn.Calls {
		c := p.byID[id]
		if c == nil || seen[c.ID] {
			continue
		}
		seen[c.ID] = true
		fmt.Fprintf(&b, "  - %s — %s\n", c.Display(), c.Summary())
		n++
		if n >= 15 {
			break
		}
	}
	return b.String(), true
}

// answerWhereDefined handles "where is X defined".
func (p *Project) answerWhereDefined(name string) (string, bool) {
	fn, cands, typ, pkg := p.resolve(name)
	switch {
	case fn != nil:
		return fmt.Sprintf("%s is defined at %s:%d\n", fn.Display(), p.LinkPath(fn.File), fn.Line), true
	case typ != nil:
		return fmt.Sprintf("type %s.%s is defined at %s:%d\n", typ.Pkg, typ.Name, p.LinkPath(typ.File), typ.Line), true
	case pkg != nil:
		dir := pkg.Dir
		if dir == "" {
			dir = "."
		}
		return fmt.Sprintf("package %s lives in %s/\n", pkg.Name, dir), true
	case len(cands) > 0:
		return p.ambiguous(name, cands), true
	}
	return "", false
}

// answerShowMe handles "show me X": the actual source of the function.
func (p *Project) answerShowMe(name string) (string, bool) {
	fn, cands, typ, _ := p.resolve(name)
	if typ != nil {
		return typ.Detail(p), true
	}
	if len(cands) > 0 {
		return p.ambiguous(name, cands), true
	}
	if fn == nil {
		return "", false
	}
	src := p.SourceOf(fn, 60)
	if src == "" {
		return fmt.Sprintf("I couldn't read the source of %s.\n", fn.Display()), true
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s  (%s:%d)\n", fn.Display(), p.LinkPath(fn.File), fn.Line)
	b.WriteString("```go\n")
	b.WriteString(src)
	b.WriteString("\n```\n")
	return b.String(), true
}

// answerPackage handles "what's in <pkg>" / "list functions in <pkg>".
// It looks the package up directly: a type with the same name (e.g.
// type Store in package store) must not shadow the package here.
func (p *Project) answerPackage(name string) (string, bool) {
	pkg := p.findPackage(name)
	if pkg == nil {
		return "", false
	}
	return p.PackageDetail(pkg), true
}

// findPackage finds a package by name or directory, case-insensitively.
func (p *Project) findPackage(name string) *Package {
	lower := strings.ToLower(name)
	for _, pk := range p.Packages {
		if strings.EqualFold(pk.Name, name) || strings.EqualFold(pk.Dir, name) ||
			strings.EqualFold(shortDir(pk.Dir), lower) {
			return pk
		}
	}
	return nil
}

// ambiguous lists candidate functions when a bare name matches several.
func (p *Project) ambiguous(name string, cands []*Func) string {
	var b strings.Builder
	fmt.Fprintf(&b, "\"%s\" matches %d functions — which one did you mean?\n", name, len(cands))
	for i, fn := range cands {
		if i >= 8 {
			break
		}
		fmt.Fprintf(&b, "  - %s  (%s:%d)\n", fn.Display(), p.LinkPath(fn.File), fn.Line)
	}
	b.WriteString("Ask again with the full name, e.g. \"what does chat.routeDomain do\".\n")
	return b.String()
}
