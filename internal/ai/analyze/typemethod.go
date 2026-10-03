package analyze

import (
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// typeMethodPlan answers issues that ask for a new method on a named
// type: "Implement Info method on dataframe". The issue's nouns resolve
// against the index — the type is the target, its existing methods are
// the pattern to imitate, and the repo's own *Option types shape any
// options the issue asks for. No per-issue-type logic.
type typeMethodPlan struct {
	typ        *Type
	pkg        *Package
	methodName string           // "Info", from the issue
	receiver   string           // "df", the type's own convention
	siblings   []*Func          // methods on the type, keyword-ranked
	optTypes   []*Type          // *Option types in the package
	optUser    map[string]*Func // option type name -> example func using it
	wantsOpts  bool             // the issue asks for options
}

var methodRe = regexp.MustCompile(`(?i)\b([A-Za-z][A-Za-z0-9]*)\s+methods?\b`)

func (p *Project) buildTypeMethodPlan(c IssueConcepts) *typeMethodPlan {
	typ := p.targetType(c)
	if typ == nil {
		return nil
	}
	name := methodName(c)
	if name == "" {
		return nil
	}
	methods := p.typeMethods(typ)
	for _, m := range methods {
		if strings.EqualFold(m.Name, name) {
			return nil // the method already exists — not a new-method issue
		}
	}
	tp := &typeMethodPlan{typ: typ, methodName: name}
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t == typ {
				tp.pkg = pkg
			}
		}
	}
	tp.receiver = commonReceiver(methods)
	tp.siblings = rankSiblings(methods, c)
	tp.wantsOpts = strings.Contains(strings.ToLower(c.Title+" "+c.Body), "option")
	if tp.pkg != nil {
		tp.optUser = map[string]*Func{}
		for _, t := range tp.pkg.Types {
			if t.Kind == "struct" && strings.HasSuffix(t.Name, "Option") {
				tp.optTypes = append(tp.optTypes, t)
				if u := optionUser(tp.pkg, t.Name); u != nil {
					tp.optUser[t.Name] = u
				}
			}
		}
		sort.Slice(tp.optTypes, func(i, j int) bool { return tp.optTypes[i].Name < tp.optTypes[j].Name })
	}
	return tp
}

// targetType resolves the issue's nouns to a struct type: the type the
// new method belongs to. Candidates rank by method count — the type the
// issue is about is usually the one with the rich API.
func (p *Project) targetType(c IssueConcepts) *Type {
	words := map[string]bool{}
	for _, w := range strings.FieldsFunc(strings.ToLower(c.Title),
		func(r rune) bool { return r < 'a' || r > 'z' }) {
		words[w] = true
		words[stem(w)] = true
	}
	var best *Type
	bestN := -1
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Kind != "struct" {
				continue
			}
			if !words[strings.ToLower(t.Name)] {
				continue
			}
			if n := len(p.typeMethods(t)); n > bestN {
				best, bestN = t, n
			}
		}
	}
	return best
}

// typeMethods returns the *Func methods declared on a type.
func (p *Project) typeMethods(t *Type) []*Func {
	var out []*Func
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			if strings.TrimPrefix(fn.Receiver, "*") == t.Name {
				out = append(out, fn)
			}
		}
	}
	return out
}

// methodName extracts the new method's name: "Implement Info method"
// -> "Info". Generic verbs ("Add method", "new method") don't count.
func methodName(c IssueConcepts) string {
	for _, m := range methodRe.FindAllStringSubmatch(c.Title, -1) {
		n := m[1]
		if planStopwords[strings.ToLower(n)] {
			continue
		}
		return strings.ToUpper(n[:1]) + n[1:]
	}
	return ""
}

// commonReceiver returns the receiver name the type's methods use.
func commonReceiver(methods []*Func) string {
	counts := map[string]int{}
	best, bestN := "", 0
	for _, m := range methods {
		// Receiver field holds the type; recover the variable name from Sig.
		if i := strings.Index(m.Sig, "("); i >= 0 {
			rest := m.Sig[i+1:]
			if j := strings.IndexAny(rest, " )"); j > 0 {
				name := rest[:j]
				counts[name]++
				if counts[name] > bestN {
					best, bestN = name, counts[name]
				}
			}
		}
	}
	if best == "" {
		best = "x"
	}
	return best
}

// rankSiblings orders the type's methods by overlap with the issue's
// vocabulary: the closest relatives of the requested method first.
func rankSiblings(methods []*Func, c IssueConcepts) []*Func {
	kw := map[string]bool{}
	for _, s := range wordTokens(c.Title) {
		kw[s] = true
	}
	for _, s := range wordTokens(c.Body) {
		kw[s] = true
	}
	type scored struct {
		fn    *Func
		score int
	}
	var ss []scored
	for _, fn := range methods {
		score := 0
		for _, t := range tokenizeName(fn.Name) {
			if wordHit(t, kw) {
				score += 2
			}
		}
		ss = append(ss, scored{fn, score})
	}
	sort.Slice(ss, func(i, j int) bool {
		if ss[i].score != ss[j].score {
			return ss[i].score > ss[j].score
		}
		return ss[i].fn.Name < ss[j].fn.Name
	})
	var out []*Func
	for i, s := range ss {
		if i >= 4 {
			break
		}
		out = append(out, s.fn)
	}
	return out
}

// optionUser finds a non-test function whose signature mentions the
// option type: the repo's own example of the options pattern.
func optionUser(pkg *Package, optName string) *Func {
	for _, fn := range pkg.Funcs {
		if fn.IsTest {
			continue
		}
		if strings.Contains(fn.Sig, optName) {
			return fn
		}
	}
	return nil
}

func (p *Project) writeTypeMethodPlan(c IssueConcepts, tp *typeMethodPlan) string {
	var b strings.Builder
	repo := filepath.Base(p.Root)
	pkgDir := ""
	if tp.pkg != nil {
		pkgDir = tp.pkg.Dir
	} else if d := filepath.Dir(tp.typ.File); d != "." {
		pkgDir = d
	}
	fileBase := filepath.Base(tp.typ.File)

	methodSig := fmt.Sprintf("func (%s *%s) %s(...)", tp.receiver, tp.typ.Name, tp.methodName)
	if tp.wantsOpts && len(tp.optTypes) > 0 {
		methodSig = fmt.Sprintf("func (%s *%s) %s(verbose ...%sOption)",
			tp.receiver, tp.typ.Name, tp.methodName, tp.methodName)
	}

	b.WriteString(fmt.Sprintf("To add the %s method, edit one file:\n\n", tp.methodName))
	b.WriteString(fmt.Sprintf("  %s\n", repo))
	if pkgDir == "" {
		b.WriteString(fmt.Sprintf("  └── %s  ← OPEN THIS FILE\n", fileBase))
		b.WriteString(fmt.Sprintf("      └── type %s struct  (line %d)\n", tp.typ.Name, tp.typ.Line))
		b.WriteString("          └── ✚ ADD HERE\n")
		b.WriteString(fmt.Sprintf("              %s\n\n", methodSig))
	} else {
		b.WriteString(fmt.Sprintf("  └── %s/  ← package\n", pkgDir))
		b.WriteString(fmt.Sprintf("      └── %s  ← OPEN THIS FILE\n", fileBase))
		b.WriteString(fmt.Sprintf("          └── type %s struct  (line %d)\n", tp.typ.Name, tp.typ.Line))
		b.WriteString("              └── ✚ ADD HERE\n")
		b.WriteString(fmt.Sprintf("                  %s\n\n", methodSig))
	}

	b.WriteString("  Steps:\n")
	b.WriteString(fmt.Sprintf("  1. Open %s.\n", p.LinkPath(tp.typ.File)))
	b.WriteString(fmt.Sprintf("  2. Find `type %s struct` (line %d).\n", tp.typ.Name, tp.typ.Line))
	if ex := simplestFunc(tp.siblings); ex != nil {
		b.WriteString("  3. Add the new method after the type's other methods,\n")
		b.WriteString(fmt.Sprintf("     writing it like %s (%s:%d):\n", ex.Name, p.LinkPath(ex.File), ex.Line))
		b.WriteString(fmt.Sprintf("       %s\n", strings.TrimPrefix(ex.Sig, "func ")))
	} else {
		b.WriteString("  3. Add the new method right after the type declaration.\n")
	}
	if tp.wantsOpts && len(tp.optTypes) > 0 {
		ot := bestOption(tp)
		b.WriteString(fmt.Sprintf("  4. The issue asks for options: add `type %sOption struct`,\n", tp.methodName))
		b.WriteString(fmt.Sprintf("     copying %s (%s:%d).\n", ot.Name, p.LinkPath(ot.File), ot.Line))
	}
	return b.String()
}

// simplestFunc picks the shortest signature: the smallest complete
// method is the best template for a beginner to copy.
func simplestFunc(fns []*Func) *Func {
	var best *Func
	for _, fn := range fns {
		if best == nil || len(fn.Sig) < len(best.Sig) ||
			(len(fn.Sig) == len(best.Sig) && fn.Name < best.Name) {
			best = fn
		}
	}
	return best
}

// bestOption picks the *Option type whose example user is a method on
// the target type — the closest shape to the method being added.
func bestOption(tp *typeMethodPlan) *Type {
	for _, t := range tp.optTypes {
		if u := tp.optUser[t.Name]; u != nil &&
			strings.TrimPrefix(u.Receiver, "*") == tp.typ.Name {
			return t
		}
	}
	if len(tp.optTypes) > 0 {
		return tp.optTypes[0]
	}
	return nil
}
