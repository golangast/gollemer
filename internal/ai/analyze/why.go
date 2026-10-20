package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// dispName renders a function as Recv.Name or Name.
func dispName(fn *Func) string {
	if fn.Receiver != "" {
		return typeName(fn.Receiver) + "." + fn.Name
	}
	return fn.Name
}

// typeName renders a receiver type without its pointer star or package.
func typeName(recv string) string {
	return strings.TrimPrefix(shortType(recv), "*")
}

// siblingCount reports how many methods share fn's receiver (0 for plain funcs).
func (p *Project) siblingCount(fn *Func) int {
	if fn.Receiver == "" {
		return 0
	}
	n := 0
	for _, o := range p.AllFuncs() {
		if o.Receiver == fn.Receiver {
			n++
		}
	}
	return n
}

func behaviorSet(fn *Func) map[string]bool {
	out := map[string]bool{}
	for _, b := range FuncBehaviors(fn) {
		out[b] = true
	}
	return out
}

func callerNames(p *Project, fn *Func, limit int) []string {
	var names []string
	for _, id := range fn.Callers {
		if cf := p.FuncByID(id); cf != nil {
			names = append(names, dispName(cf))
		}
	}
	sort.Strings(names)
	if len(names) > limit {
		names = names[:limit]
	}
	return names
}

// WhyText is the WHY-BRAIN: it explains, from computed structure, WHY a
// function is shaped the way it is — distinguishing what the author STATED
// (doc comment) from what the structure IMPLIES. Where neither speaks, it
// says so instead of inventing intent.
func (p *Project) WhyText(fn *Func) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Why %s is the way it is:\n\n", dispName(fn))

	stated := PieceFirstSentence(fn.Doc)
	if stated != "" {
		fmt.Fprintf(&b, "Stated by the author: %q\n\n", stated)
	} else {
		b.WriteString("The author left no comment, so the reasons below are read from the structure alone.\n\n")
	}

	beh := behaviorSet(fn)
	var reasons []string

	if beh[behLocks] {
		sib := p.siblingCount(fn)
		reason := fmt.Sprintf("It LOCKS because %s is shared state", typeName(fn.Receiver))
		if sib > 1 {
			reason += fmt.Sprintf(" — %d methods work on the same %s", sib, typeName(fn.Receiver))
		}
		if len(fn.Callers) > 0 {
			reason += fmt.Sprintf(", and %d different places can call it", len(fn.Callers))
		}
		reason += ". Without the lock, two callers running at the same time could write over each other."
		reasons = append(reasons, reason)
	}
	if beh[behReadLocks] {
		reasons = append(reasons, fmt.Sprintf("It takes a READ lock because it only reads %s's state — the read lock lets many readers proceed at once while still keeping writers out until it finishes.", typeName(fn.Receiver)))
	}
	if beh[behIterates] {
		reason := "It LOOPS because its job covers a whole collection, not one value"
		if len(fn.LoopCalls) > 0 {
			var inside []string
			for id := range fn.LoopCalls {
				if cf := p.FuncByID(id); cf != nil {
					inside = append(inside, dispName(cf))
				}
			}
			sort.Strings(inside)
			if len(inside) > 0 {
				reason += " — for each entry it calls " + strings.Join(inside, ", ")
			}
		}
		reasons = append(reasons, reason+".")
	}
	if beh[behMayFail] {
		reason := "It RETURNS AN ERROR because the work can fail, and the caller has to decide what happens then"
		if names := callerNames(p, fn, 3); len(names) > 0 {
			reason += " — the error travels back to " + strings.Join(names, ", ")
		}
		reasons = append(reasons, reason+".")
	}
	if len(fn.Callers) > 0 && !beh[behMayFail] {
		reasons = append(reasons, fmt.Sprintf("It EXISTS FOR its callers: %s depend on it.", strings.Join(callerNames(p, fn, 3), ", ")))
	}
	if len(reasons) == 0 {
		reasons = append(reasons, fmt.Sprintf("Nothing special shows in the structure — no locks, no loop, no error. It does one plain job, and its name says it: %q.", SplitCamelWords(fn.Name)))
	}
	for _, r := range reasons {
		b.WriteString("• " + r + "\n")
	}
	b.WriteString("\nAnything deeper — what the author was thinking — is not in the code, and I won't guess it.")
	return strings.TrimRight(b.String(), "\n")
}

// CompareText is the COMPARE-BRAIN: two functions side by side, computed
// from the same facts the other brains use — purpose, shape, behavior,
// usage, relationship — ending with the single biggest difference.
func (p *Project) CompareText(a, b *Func) string {
	var out strings.Builder
	fmt.Fprintf(&out, "%s  vs  %s\n\n", dispName(a), dispName(b))

	shape := func(fn *Func) string {
		s := "a plain function"
		if fn.Receiver != "" {
			s = "a method on " + typeName(fn.Receiver)
		}
		return fmt.Sprintf("%s, %d lines, %d input(s)", s, fn.EndLine-fn.Line+1, fn.Params)
	}
	pa, pb := PieceFirstSentence(a.Doc), PieceFirstSentence(b.Doc)
	if pa == "" {
		pa = "(no comment)"
	}
	if pb == "" {
		pb = "(no comment)"
	}
	fmt.Fprintf(&out, "Purpose:  %s — %q\n          %s — %q\n", a.Name, pa, b.Name, pb)
	fmt.Fprintf(&out, "Shape:    %s is %s; %s is %s.\n", a.Name, shape(a), b.Name, shape(b))

	ba, bb := behaviorSet(a), behaviorSet(b)
	behLine := func(name string, set map[string]bool) string {
		var parts []string
		if set[behLocks] {
			parts = append(parts, "locks")
		}
		if set[behReadLocks] {
			parts = append(parts, "read-locks")
		}
		if set[behIterates] {
			parts = append(parts, "iterates")
		}
		if set[behMayFail] {
			parts = append(parts, "can fail")
		}
		if len(parts) == 0 {
			return name + ": plain, no locks/errors/loops"
		}
		return name + ": " + strings.Join(parts, ", ")
	}
	fmt.Fprintf(&out, "Behavior: %s\n          %s\n", behLine(a.Name, ba), behLine(b.Name, bb))
	fmt.Fprintf(&out, "Used by:  %s has %d caller(s); %s has %d.\n", a.Name, len(a.Callers), b.Name, len(b.Callers))

	// Relationship.
	rel := "They never call each other"
	for _, id := range a.Calls {
		if id == b.ID {
			rel = fmt.Sprintf("%s CALLS %s — %s builds on %s", a.Name, b.Name, a.Name, b.Name)
		}
	}
	for _, id := range b.Calls {
		if id == a.ID {
			rel = fmt.Sprintf("%s CALLS %s — %s builds on %s", b.Name, a.Name, b.Name, a.Name)
		}
	}
	shared := intersectStrings(a.Callers, b.Callers)
	if len(shared) > 0 && !strings.Contains(rel, "CALLS") {
		var names []string
		for _, id := range shared {
			if cf := p.FuncByID(id); cf != nil {
				names = append(names, dispName(cf))
			}
		}
		rel += "; they are both used by " + strings.Join(names, ", ")
	}
	fmt.Fprintf(&out, "Relation: %s.\n", rel)

	// Tests.
	tests := func(fn *Func) int {
		n := 0
		for _, id := range fn.Callers {
			if cf := p.FuncByID(id); cf != nil && cf.IsTest {
				n++
			}
		}
		return n
	}
	fmt.Fprintf(&out, "Tests:    %d test(s) exercise %s; %d exercise %s.\n", tests(a), a.Name, tests(b), b.Name)

	// The one biggest difference.
	big := ""
	switch {
	case ba[behMayFail] != bb[behMayFail]:
		failing, plain := a, b
		if !ba[behMayFail] {
			failing, plain = b, a
		}
		big = fmt.Sprintf("%s can fail and hands the caller an error; %s always succeeds", failing.Name, plain.Name)
	case ba[behLocks] != bb[behLocks]:
		lk, pl := a, b
		if !ba[behLocks] {
			lk, pl = b, a
		}
		big = fmt.Sprintf("%s locks shared state; %s doesn't need to", lk.Name, pl.Name)
	case len(a.Callers) != len(b.Callers):
		more, fewer := a, b
		if len(b.Callers) > len(a.Callers) {
			more, fewer = b, a
		}
		big = fmt.Sprintf("%s is the more depended-on of the two (%d callers vs %d)", more.Name, len(more.Callers), len(fewer.Callers))
	case pa != pb:
		big = "they do genuinely different jobs (see Purpose above)"
	default:
		big = "they are close cousins — same kind of shape, different names"
	}
	fmt.Fprintf(&out, "\nIn short: %s.", big)
	return out.String()
}

func intersectStrings(a, b []string) []string {
	inB := map[string]bool{}
	for _, s := range b {
		inB[s] = true
	}
	var out []string
	for _, s := range a {
		if inB[s] {
			out = append(out, s)
		}
	}
	return out
}
