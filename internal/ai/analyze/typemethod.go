package analyze

import (
	"fmt"
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
	fmt.Fprintf(&b, "TARGET — add the method to this type:\n")
	fmt.Fprintf(&b, "  %s:%d\n", p.LinkPath(tp.typ.File), tp.typ.Line)
	fmt.Fprintf(&b, "  type %s struct\n", tp.typ.Name)
	if tp.wantsOpts && len(tp.optTypes) > 0 {
		fmt.Fprintf(&b, "  New method, e.g.: func (%s *%s) %s(verbose ...%sOption)\n",
			tp.receiver, tp.typ.Name, tp.methodName, tp.methodName)
	} else {
		fmt.Fprintf(&b, "  New method: func (%s *%s) %s(...)\n",
			tp.receiver, tp.typ.Name, tp.methodName)
	}
	fmt.Fprintf(&b, "  Why here: the issue asks for %s %s method on %s.\n\n",
		an(tp.methodName), tp.methodName, strings.ToLower(tp.typ.Name))
	if len(tp.siblings) > 0 {
		fmt.Fprintf(&b, "SIBLINGS — methods on %s to imitate:\n", tp.typ.Name)
		for _, fn := range tp.siblings {
			fmt.Fprintf(&b, "  %s:%d — %s\n",
				p.LinkPath(fn.File), fn.Line, strings.TrimPrefix(fn.Sig, "func "))
		}
		b.WriteString("\n")
	}
	if tp.wantsOpts && len(tp.optTypes) > 0 {
		b.WriteString("OPTIONS — this repo's options pattern:\n")
		for _, t := range tp.optTypes {
			fmt.Fprintf(&b, "  type %s struct (%s:%d)\n", t.Name, p.LinkPath(t.File), t.Line)
			if u := tp.optUser[t.Name]; u != nil {
				fmt.Fprintf(&b, "  used by: %s\n", strings.TrimPrefix(u.Sig, "func "))
			}
		}
		fmt.Fprintf(&b, "  Why: the issue asks for options — add %s %sOption struct in the same shape.\n\n",
			an(tp.methodName), tp.methodName)
	}
	return b.String()
}

// an returns "a" or "an" for a word.
func an(w string) string {
	if w == "" {
		return "a"
	}
	switch strings.ToLower(w[:1]) {
	case "a", "e", "i", "o", "u":
		return "an"
	}
	return "a"
}
