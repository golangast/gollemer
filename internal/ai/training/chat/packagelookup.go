package chat

// packagelookup.go — package lookup: an explicit "which package
// …" / "what package is X in" question is answered from the
// standard-library catalog snapshot (data/stdpackages.json,
// built by scripts/import_stdpackages from pkg.go.dev/std),
// never from a brain's guess. The claim is gated twice: the
// sentence must wear the package-lookup shape (a question about
// the library, not about this project — "what package is my
// server in" keeps falling through), and the catalog match must
// be strong (a package whose name or synopsis the question's
// own words hit hard). Anything weaker falls through untouched,
// and an absent snapshot turns the feature off entirely: absent
// file, byte-identical behaviour.
//
// The matching is token-exact over package names and synopses —
// the same vocabulary discipline as the resources matcher — and
// internal packages are in the snapshot (the index lists them)
// but never offered as answers: user code cannot import them.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// --- catalog loading (mirrors advocates()) ---

type stdPackageEntry struct {
	Path     string `json:"path"`
	Synopsis string `json:"synopsis"`
}

type stdPackagesDoc struct {
	Source   string            `json:"source"`
	Packages []stdPackageEntry `json:"packages"`
}

var (
	stdPackagesOnce  = &sync.Once{} // a pointer so tests can swap in a fresh one
	stdPackagesCache []stdPackageEntry
)

// stdPackages lazily loads data/stdpackages.json from the chat's
// project root (or the working directory). Absent or empty =
// nil, and the lookup feature is off.
func stdPackages() []stdPackageEntry {
	stdPackagesOnce.Do(func() {
		for _, base := range []string{unifiedProjectRoot, startFolder()} {
			if base == "" {
				continue
			}
			blob, err := os.ReadFile(filepath.Join(base, "data", "stdpackages.json"))
			if err != nil {
				continue
			}
			var d stdPackagesDoc
			if json.Unmarshal(blob, &d) == nil && len(d.Packages) > 0 {
				stdPackagesCache = d.Packages
				return
			}
		}
	})
	return stdPackagesCache
}

// --- the question shape ---

// packageLookupPrefixes open a package-lookup question; the
// rest of the line is the thing being looked up.
var packageLookupPrefixes = []string{
	"which go package ",
	"what go package ",
	"which package ",
	"what package ",
	"what's the package for ",
	"what is the package for ",
}

// packageLookupHelpers are filler openers stripped from the
// rest ("which package should i use for X", "which package is
// X in"): they carry no topic, and leaving them in would only
// add noise terms to the match.
var packageLookupHelpers = []string{
	"should i use for ", "do i use for ", "can i use for ",
	"is used for ", "to use for ", "is for ", "does ", "is ",
	"has ", "can ", "should ",
}

// parsePackageLookup reads one line as a package-lookup
// question and returns what is being looked up, as typed
// (minus the shape's own furniture). False for every other
// line — including project questions wearing similar words.
func parsePackageLookup(line string) (string, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSpace(s)
	low := strings.ToLower(s)
	rest := ""
	for _, p := range packageLookupPrefixes {
		if strings.HasPrefix(low, p) {
			rest = s[len(p):]
			break
		}
	}
	if rest == "" {
		return "", false
	}
	// Helper openers, longest first by construction of the list.
	rlow := strings.ToLower(rest)
	for _, h := range packageLookupHelpers {
		if strings.HasPrefix(rlow, h) {
			rest = rest[len(h):]
			break
		}
	}
	// The "… in" tail of the is-X-in shape.
	rlow = strings.ToLower(rest)
	for _, suf := range []string{" in", " from", " part of", " for"} {
		if strings.HasSuffix(rlow, suf) {
			rest = rest[:len(rest)-len(suf)]
			break
		}
	}
	rest = strings.TrimSpace(rest)
	if rest == "" || len(strings.Fields(rest)) > 6 {
		return "", false
	}
	// A question about the user's own code is a project
	// question, not a library lookup.
	for _, f := range strings.Fields(strings.ToLower(rest)) {
		if f == "my" || f == "our" || f == "this" {
			return "", false
		}
	}
	return rest, true
}

// --- matching ---

// packageBase is a package path's last segment — the name code
// actually writes after the dot.
func packageBase(path string) string {
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[i+1:]
	}
	return path
}

// matchStdPackages answers a lookup from the catalog: a direct
// hit (the rest IS a package path, or a unique package name),
// else the token match — name hits weigh most, synopsis hits
// corroborate, and only a strong best is an answer. Ties at the
// top are returned together rather than guessed between.
// Internal packages are never answers.
func matchStdPackages(pkgs []stdPackageEntry, rest string) []stdPackageEntry {
	var visible []stdPackageEntry
	for _, p := range pkgs {
		if strings.Contains("/"+p.Path+"/", "/internal/") {
			continue
		}
		visible = append(visible, p)
	}
	// Direct hits first: the user named the package, they want
	// its synopsis and page.
	rlow := strings.ToLower(strings.TrimSpace(rest))
	if !strings.Contains(rlow, " ") {
		for _, p := range visible {
			if p.Path == rlow {
				return []stdPackageEntry{p}
			}
		}
		var byBase []stdPackageEntry
		for _, p := range visible {
			if packageBase(p.Path) == rlow {
				byBase = append(byBase, p)
			}
		}
		if len(byBase) == 1 {
			return byBase
		}
		if len(byBase) > 1 {
			return byBase // an ambiguous name: name them all
		}
	}
	// A path or a file name that isn't a package is not a
	// lookup the catalog can answer.
	if strings.ContainsAny(rlow, "/") || strings.Contains(rlow, ".go") || strings.Contains(rlow, "://") {
		return nil
	}
	terms := termSet(rest)
	if len(terms) == 0 {
		return nil
	}
	type scored struct {
		p     stdPackageEntry
		score int
		idx   int
	}
	var hits []scored
	for i, p := range visible {
		if p.Synopsis == "" {
			continue
		}
		score := 0
		base := packageBase(p.Path)
		pathTerms := termSet(p.Path)
		syn := termSet(p.Synopsis)
		for t := range terms {
			switch {
			case t == base:
				score += 5
			case pathTerms[t]:
				score += 3
			case syn[t]:
				score += 2
			}
		}
		if score > 0 {
			hits = append(hits, scored{p, score, i})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].idx < hits[j].idx
	})
	if len(hits) == 0 || hits[0].score < 4 {
		return nil
	}
	out := []stdPackageEntry{hits[0].p}
	for _, h := range hits[1:] {
		if h.score != hits[0].score || len(out) >= 3 {
			break
		}
		out = append(out, h.p)
	}
	return out
}

// --- the answer ---

// tryPackageLookup claims an explicit package-lookup question
// and answers it from the catalog snapshot. See the file header
// for the ownership rules.
func tryPackageLookup(line string) bool {
	rest, ok := parsePackageLookup(line)
	if !ok {
		return false
	}
	pkgs := stdPackages()
	if len(pkgs) == 0 {
		return false
	}
	hits := matchStdPackages(pkgs, rest)
	if len(hits) == 0 {
		return false
	}
	var b strings.Builder
	if len(hits) == 1 {
		p := hits[0]
		fmt.Fprintf(&b, "That's package %s — %s\n", p.Path, p.Synopsis)
		fmt.Fprintf(&b, "Docs: https://pkg.go.dev/%s\n", p.Path)
	} else {
		b.WriteString("More than one package matches that:\n")
		for _, p := range hits {
			fmt.Fprintf(&b, "%s — %s (https://pkg.go.dev/%s)\n", p.Path, p.Synopsis, p.Path)
		}
	}
	printPaced(b.String())
	recordBrain("analyst", true, "package lookup: "+rest)
	return true
}
