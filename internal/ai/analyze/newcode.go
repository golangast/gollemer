package analyze

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// newCodePlan answers issues that ask for new code in a named location
// against a named package/module: benchmark suites, new commands, new
// adapters. Fully general — the issue's own backticked paths name the
// new locations, the issue's nouns resolve the target package, and the
// package's exported API is the work surface. No per-issue-type logic.
type newCodePlan struct {
	newPaths []string // issue-named paths that don't exist yet
	target   *Package // the module/package the issue is about
	targets  []*Func  // exported API, keyword-ranked
	patterns []string // existing test files to imitate, rel to Root
	replace  []replaceSite // functions using the calls the issue wants replaced
	replaceCalls []string   // the named calls, e.g. ["os.ReadFile", "io.ReadAll"]
}

// replaceSite is one function using an external call the issue wants
// replaced — the modification site for a "replace X with Y" change.
type replaceSite struct {
	fn  *Func
	via []string // the named calls it uses
}

func (p *Project) buildNewCodePlan(c IssueConcepts) *newCodePlan {
	paths := missingPaths(p, c)
	if len(paths) == 0 {
		return nil
	}
	pkg := p.targetPackage(c)
	if pkg == nil {
		return nil
	}
	np := &newCodePlan{newPaths: paths, target: pkg}
	np.targets = p.packageTargets(pkg, c)
	np.patterns = p.testPatterns(pkg)
	// "Replace X with Y": the issue names existing files and the calls
	// to swap out — the model resolves both against the index.
	if wantsReplace(c) {
		if files := existingPaths(p, c); len(files) > 0 {
			if calls := namedCalls(c); len(calls) > 0 {
				np.replaceCalls = calls
				np.replace = p.replaceSites(files, calls)
			}
		}
	}
	return np
}

// existingPaths returns the issue's backticked paths that exist on disk:
// the files (or directories) the change touches.
func existingPaths(p *Project, c IssueConcepts) []string {
	var out []string
	seen := map[string]bool{}
	for _, w := range c.CodeWords {
		w = strings.Trim(w, "`\"' ")
		if !isPathWord(w) || seen[w] {
			continue
		}
		seen[w] = true
		full := filepath.Join(p.Root, filepath.FromSlash(strings.TrimSuffix(w, "/")))
		if _, err := os.Stat(full); err == nil {
			out = append(out, strings.TrimSuffix(w, "/"))
		}
	}
	return out
}

// namedCalls returns backticked pkg.Func words: external calls the issue
// talks about, e.g. os.ReadFile.
func namedCalls(c IssueConcepts) []string {
	var out []string
	seen := map[string]bool{}
	for _, w := range c.CodeWords {
		w = strings.Trim(w, "`\"' ")
		if seen[w] || strings.Contains(w, "/") {
			continue
		}
		parts := strings.SplitN(w, ".", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			continue
		}
		c0, c1 := rune(parts[0][0]), rune(parts[1][0])
		if !(c0 >= 'a' && c0 <= 'z') || !(c1 >= 'A' && c1 <= 'Z') {
			continue
		}
		seen[w] = true
		out = append(out, w)
	}
	return out
}

// wantsReplace reports whether the issue asks to swap one approach for
// another. Thin request-language mapping.
func wantsReplace(c IssueConcepts) bool {
	lower := strings.ToLower(c.Title + "\n" + c.Body)
	for _, w := range []string{"replac", "instead of", "swap", "migrat"} {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}

// replaceSites finds the non-test functions in the named files (or
// directories) that call the named external calls: the modification
// sites for a replacement.
func (p *Project) replaceSites(files []string, calls []string) []replaceSite {
	want := map[string]bool{}
	for _, c := range calls {
		want[c] = true
	}
	matchFile := func(rel string) bool {
		for _, f := range files {
			if rel == f || strings.HasPrefix(rel, f+"/") {
				return true
			}
		}
		return false
	}
	var out []replaceSite
	for _, fn := range p.byID {
		if fn.IsTest || !matchFile(fn.File) {
			continue
		}
		var via []string
		for _, e := range fn.ExtCalls {
			if want[e] {
				via = append(via, e)
			}
		}
		if len(via) > 0 {
			out = append(out, replaceSite{fn: fn, via: via})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].fn.File != out[j].fn.File {
			return out[i].fn.File < out[j].fn.File
		}
		return out[i].fn.Line < out[j].fn.Line
	})
	if len(out) > 6 {
		out = out[:6]
	}
	return out
}

// isPathWord reports whether a backticked word looks like a repo path:
// has a slash, no whitespace, not a URL or module path (first segment
// has no dot, so golang.org/x/sys is out), and looks like a file or
// directory (trailing slash, .go suffix, or at least two slashes — so
// allocs/op is out).
func isPathWord(w string) bool {
	if !strings.Contains(w, "/") || strings.ContainsAny(w, " \t\n") ||
		strings.Contains(w, "://") {
		return false
	}
	if strings.Contains(strings.Split(w, "/")[0], ".") {
		return false
	}
	if strings.HasSuffix(w, "/") || strings.HasSuffix(w, ".go") {
		return true
	}
	return strings.Count(w, "/") >= 2
}

// missingPaths returns the issue's backticked paths that don't exist in
// the repo: the issue telling us where to create.
func missingPaths(p *Project, c IssueConcepts) []string {
	var out []string
	seen := map[string]bool{}
	for _, w := range c.CodeWords {
		w = strings.Trim(w, "`\"' ")
		if !isPathWord(w) || seen[w] {
			continue
		}
		seen[w] = true
		full := filepath.Join(p.Root, filepath.FromSlash(strings.TrimSuffix(w, "/")))
		if _, err := os.Stat(full); os.IsNotExist(err) {
			out = append(out, w)
		}
	}
	return out
}

// targetPackage resolves the module the new code is about. The
// new-code path's last segment names it: in
// internal/tests/benchmark/filemanager/ the module is "filemanager".
// Exact package-name match wins over partial ("filemanager" beats
// "filemanager_test").
func (p *Project) targetPackage(c IssueConcepts) *Package {
	norm := func(s string) string {
		var b strings.Builder
		for _, r := range strings.ToLower(s) {
			if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
				b.WriteRune(r)
			}
		}
		return b.String()
	}
	for _, path := range missingPaths(p, c) {
		segs := strings.Split(strings.TrimSuffix(path, "/"), "/")
		last := segs[len(segs)-1]
		// internal/util/mmap.go names a file: the module is the parent
		// directory (util), not the file.
		if strings.HasSuffix(last, ".go") && len(segs) >= 2 {
			last = segs[len(segs)-2]
		}
		want := norm(last)
		var partial *Package
		for _, pkg := range p.Packages {
			name := norm(pkg.Name)
			if name == want {
				return pkg
			}
			if partial == nil && strings.Contains(name, want) {
				partial = pkg
			}
		}
		if partial != nil {
			return partial
		}
	}
	return nil
}

// packageTargets ranks the package's exported functions by keyword
// overlap with the issue's own vocabulary (title + body): the API
// surface the new code works against. Ties break toward hot paths
// (more callers), then name.
func (p *Project) packageTargets(pkg *Package, c IssueConcepts) []*Func {
	kw := map[string]bool{}
	for _, s := range wordTokens(c.Title) {
		kw[s] = true
	}
	for _, s := range wordTokens(c.Body) {
		kw[s] = true
	}
	for _, m := range c.Mechanisms {
		kw[stem(m)] = true
	}
	type scored struct {
		fn    *Func
		score int
	}
	var ss []scored
	for _, fn := range pkg.Funcs {
		if fn.IsTest || !fn.Exported {
			continue
		}
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
		if len(ss[i].fn.Callers) != len(ss[j].fn.Callers) {
			return len(ss[i].fn.Callers) > len(ss[j].fn.Callers)
		}
		return ss[i].fn.Name < ss[j].fn.Name
	})
	var out []*Func
	for i, s := range ss {
		if i >= 5 {
			break
		}
		out = append(out, s.fn)
	}
	return out
}

// wordHit reports whether a name token matches an issue keyword,
// tolerating stemmer wobble ("scanning" stems to "scann" but "Scan"
// stems to "scan") via prefix match. Only used for ranking — never
// for a semantic claim.
func wordHit(tok string, kw map[string]bool) bool {
	if kw[tok] {
		return true
	}
	for k := range kw {
		if len(tok) >= 4 && len(k) >= 4 &&
			(strings.HasPrefix(tok, k) || strings.HasPrefix(k, tok)) {
			return true
		}
	}
	return false
}

// testPatterns finds existing test files to imitate: same package
// first, files with benchmarks preferred. Purely structural — it works
// for any kind of new test code.
func (p *Project) testPatterns(pkg *Package) []string {
	type scored struct {
		rel   string
		score int
	}
	var ss []scored
	seen := map[string]bool{}
	for _, q := range p.Packages {
		for _, f := range q.Files {
			if !strings.HasSuffix(f, "_test.go") || seen[f] {
				continue
			}
			seen[f] = true
			score := 0
			if q == pkg {
				score += 2
			}
			if hasBenchmark(filepath.Join(p.Root, f)) {
				score += 3
			}
			ss = append(ss, scored{f, score})
		}
	}
	sort.Slice(ss, func(i, j int) bool {
		if ss[i].score != ss[j].score {
			return ss[i].score > ss[j].score
		}
		return ss[i].rel < ss[j].rel
	})
	var out []string
	for i, s := range ss {
		if i >= 3 {
			break
		}
		out = append(out, s.rel)
	}
	return out
}

// hasBenchmark reports whether a test file defines a Benchmark function.
func hasBenchmark(full string) bool {
	src, err := os.ReadFile(full)
	if err != nil {
		return false
	}
	return strings.Contains(string(src), "func Benchmark")
}

func (p *Project) writeNewCodePlan(c IssueConcepts, np *newCodePlan) string {
	var b strings.Builder
	b.WriteString("NEW CODE — create this:\n")
	for _, path := range np.newPaths {
		fmt.Fprintf(&b, "  %s\n", path)
	}
	b.WriteString("  Why here: the issue names this location and it doesn't exist yet.\n\n")
	if len(np.targets) > 0 {
		fmt.Fprintf(&b, "TARGETS — the %s package's API:\n", np.target.Name)
		for _, fn := range np.targets {
			fmt.Fprintf(&b, "  %s:%d — %s\n",
				p.LinkPath(fn.File), fn.Line, strings.TrimPrefix(fn.Sig, "func "))
		}
		fmt.Fprintf(&b, "  Why these: the issue is about the %s module — these are its exported functions.\n\n",
			np.target.Name)
	}
	if len(np.patterns) > 0 {
		b.WriteString("IMITATE — existing tests to follow:\n")
		for _, rel := range np.patterns {
			note := ""
			if hasBenchmark(filepath.Join(p.Root, rel)) {
				note = " (has benchmarks)"
			}
			fmt.Fprintf(&b, "  %s%s\n", rel, note)
		}
		b.WriteString("\n")
	}
	if len(np.replace) > 0 {
		fmt.Fprintf(&b, "REPLACE — swap %s for the new approach here:\n",
			strings.Join(np.replaceCalls, " / "))
		for _, r := range np.replace {
			fmt.Fprintf(&b, "  %s:%d — %s calls %s\n",
				p.LinkPath(r.fn.File), r.fn.Line,
				strings.TrimPrefix(r.fn.Sig, "func "), strings.Join(r.via, ", "))
		}
		b.WriteString("  Why: the issue names these calls and files — swap them where they're used.\n\n")
	}
	return b.String()
}
