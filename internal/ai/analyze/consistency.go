package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// consistencyPlan answers issues that ask to unify a behavior across
// components: "[FEAT] Consistent game win/fail behaviours". A single
// anchor is the wrong shape — the answer is the SET of sites doing the
// same job differently, grouped by component. The cross-cutting
// concepts come from the general index: the issue's keywords that match
// functions in multiple packages. No per-issue logic.
type consistencyPlan struct {
	concepts []string          // e.g. ["win", "fail"]
	sites    []consistencySite // grouped by package
	common   string            // deepest shared package dir, for the helper
}

type consistencySite struct {
	pkg   *Package
	funcs []*Func
}

// wantsConsistency reports whether the issue asks for uniformity across
// components. Thin request-language mapping, like flagsSignal.
func wantsConsistency(c IssueConcepts) bool {
	lower := strings.ToLower(c.Title + "\n" + c.Body)
	for _, w := range []string{"across", "consistent", "inconsistent", "unif"} {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}

func (p *Project) buildConsistencyPlan(c IssueConcepts) *consistencyPlan {
	if !wantsConsistency(c) {
		return nil
	}
	kw := map[string]bool{}
	addWords := func(s string) {
		for _, t := range wordTokens(s) {
			if !planStopwords[t] {
				kw[t] = true
			}
		}
	}
	addWords(c.Title)
	addWords(c.Body)
	// Cross-cutting concepts: keywords matching functions in 2+ packages
	// (capped so generic words don't qualify) whose matches look like
	// each other — a recurring non-concept token means the sites do the
	// same job. Ranked by package span.
	nPkgs := len(p.Packages)
	maxSpan := nPkgs / 2
	if maxSpan < 3 {
		maxSpan = 3
	}
	type ck struct {
		k     string
		n     int
		funcs []*Func
	}
	var cks []ck
	for k := range kw {
		if len(k) < 3 {
			continue
		}
		pkgs := map[string]bool{}
		var funcs []*Func
		for _, pkg := range p.Packages {
			for _, fn := range pkg.Funcs {
				if fn.IsTest {
					continue
				}
				hit := false
				for _, t := range tokenizeName(fn.Name) {
					if wordHit(t, map[string]bool{k: true}) {
						hit = true
						break
					}
				}
				if hit {
					pkgs[pkg.Dir] = true
					funcs = append(funcs, fn)
				}
			}
		}
		if len(pkgs) < 2 || len(pkgs) > maxSpan {
			continue
		}
		if !coherent(k, funcs) {
			continue
		}
		cks = append(cks, ck{k, len(pkgs), funcs})
	}
	sort.Slice(cks, func(i, j int) bool {
		if cks[i].n != cks[j].n {
			return cks[i].n > cks[j].n
		}
		return cks[i].k < cks[j].k
	})
	if len(cks) == 0 {
		return nil
	}
	// One concept: the widest-spanning. Family expansion below pulls in
	// relatives like CheckGameOver next to CheckWin.
	concept := cks[0].k
	concepts := []string{concept}
	cset := map[string]bool{}
	for _, k := range concepts {
		cset[k] = true
	}
	// Sites: functions matching the concepts, grouped by package.
	byPkg := map[string]*consistencySite{}
	var order []string
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			hit := false
			for _, t := range tokenizeName(fn.Name) {
				if wordHit(t, cset) {
					hit = true
					break
				}
			}
			if !hit {
				continue
			}
			s := byPkg[pkg.Dir]
			if s == nil {
				s = &consistencySite{pkg: pkg}
				byPkg[pkg.Dir] = s
				order = append(order, pkg.Dir)
			}
			s.funcs = append(s.funcs, fn)
		}
	}
	if len(order) < 2 {
		return nil
	}
	// Family expansion: in each site package, pull in functions that
	// share a name token with a site function and match an issue
	// keyword — CheckGameOver next to CheckWin. Same job, same family.
	for _, dir := range order {
		s := byPkg[dir]
		famToks := map[string]bool{}
		for _, fn := range s.funcs {
			for _, t := range tokenizeName(fn.Name) {
				if t != concept {
					famToks[t] = true
				}
			}
		}
		for _, fn := range s.pkg.Funcs {
			if fn.IsTest {
				continue
			}
			known := false
			for _, f := range s.funcs {
				if f.ID == fn.ID {
					known = true
					break
				}
			}
			if known {
				continue
			}
			shared, kwHit := false, false
			for _, t := range tokenizeName(fn.Name) {
				if famToks[t] {
					shared = true
				}
				if wordHit(t, kw) {
					kwHit = true
				}
			}
			if shared && kwHit {
				s.funcs = append(s.funcs, fn)
			}
		}
	}
	sort.Strings(order)
	cp := &consistencyPlan{concepts: concepts, common: commonDir(order)}
	for _, dir := range order {
		s := byPkg[dir]
		sort.Slice(s.funcs, func(i, j int) bool { return s.funcs[i].Name < s.funcs[j].Name })
		cp.sites = append(cp.sites, *s)
	}
	return cp
}

// coherent reports whether a concept's matches look like each other:
// the most common non-concept, non-stopword name token recurs in at
// least half the matches. CheckForWin/CheckWin cohere ("check");
// buildGameGrid/handleGameProgressTick don't.
func coherent(concept string, funcs []*Func) bool {
	if len(funcs) == 0 {
		return false
	}
	toks := map[string]int{}
	for _, fn := range funcs {
		for _, t := range tokenizeName(fn.Name) {
			if t == concept || planStopwords[t] {
				continue
			}
			toks[t]++
		}
	}
	for _, n := range toks {
		if n*2 >= len(funcs) {
			return true
		}
	}
	return false
}

// commonDir returns the deepest shared directory of the given dirs.
func commonDir(dirs []string) string {
	if len(dirs) == 0 {
		return ""
	}
	split := func(d string) []string { return strings.Split(d, "/") }
	parts := split(dirs[0])
	for _, d := range dirs[1:] {
		sp := split(d)
		i := 0
		for i < len(parts) && i < len(sp) && parts[i] == sp[i] {
			i++
		}
		parts = parts[:i]
	}
	return strings.Join(parts, "/")
}

func (p *Project) writeConsistencyPlan(c IssueConcepts, cp *consistencyPlan) string {
	var b strings.Builder
	fmt.Fprintf(&b, "SITES — %s handling by package:\n", strings.Join(cp.concepts, "/"))
	for _, s := range cp.sites {
		fmt.Fprintf(&b, "  %s — ", s.pkg.Name)
		var parts []string
		for _, fn := range s.funcs {
			parts = append(parts, fmt.Sprintf("%s (%s:%d)",
				strings.TrimPrefix(fn.Sig, "func "), p.LinkPath(fn.File), fn.Line))
		}
		fmt.Fprintf(&b, "%s\n", strings.Join(parts, ", "))
	}
	fmt.Fprintf(&b, "\n  Why: the issue asks for consistent %s behavior across packages — "+
		"these sites do the same job differently. Unify them behind one shared helper",
		strings.Join(cp.concepts, "/"))
	if cp.common != "" {
		fmt.Fprintf(&b, ", e.g. in %s", cp.common)
	}
	b.WriteString(".\n\n")
	return b.String()
}
