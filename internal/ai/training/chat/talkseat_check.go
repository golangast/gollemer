package chat

// The talk seat's claim-checker: the verifier for generated
// talk, in the idiom of the uexplain checker
// (modelExplanationValid) — generate, then verify against the
// index before anything prints as fact.
//
// It applies to every seat reply that is not purely
// deterministic-fact output (above all, social model output).
// The check is deliberately shallow, and stated as such: it
// verifies EXISTENCE only — that a project name a reply
// presents is really in the project — never logic, exactly as
// build verification checks that code compiles, not that it
// is right.
//
// Extraction covers the code-ish reference shapes: backticked
// tokens, Name( call shapes, pkg.Name shapes, and file paths
// ending in .go. A project name that does not exist must
// NEVER print as bare fact: the sentence carrying it is
// marked with talkInventedMarker appended to that sentence;
// when marking is impossible without mangling — the sentence
// is a bare fragment consisting of little more than the
// invented reference itself — the sentence is suppressed
// instead. Sentences with no checkable reference, and
// references that are not project-shaped (stdlib pkg.Name,
// Go builtins, plain lowercase words), pass untouched.

import (
	"regexp"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// talkInventedMarker is the exact marking suffix appended to
// a sentence whose claim names something the index does not
// hold. Pinned by tests.
const talkInventedMarker = " (I don't see that in your project)"

// talkIndex is the checker's existence view of one project:
// lowercased name sets, per-package name sets, and the file
// list. Nil project -> nil index -> checking is skipped (no
// index, no verdict; the seat's fact arms already declined).
type talkIndex struct {
	funcs     map[string]bool // lowercased function names
	types     map[string]bool // lowercased type names
	pkgs      map[string]bool // lowercased package names + dir short names
	pkgNames  map[string]map[string]bool
	files     map[string]bool // slash-relative .go paths
	basenames map[string]bool // lowercased .go basenames
}

func buildTalkIndex(p *analyze.Project) *talkIndex {
	if p == nil {
		return nil
	}
	ix := &talkIndex{
		funcs:     map[string]bool{},
		types:     map[string]bool{},
		pkgs:      map[string]bool{},
		pkgNames:  map[string]map[string]bool{},
		files:     map[string]bool{},
		basenames: map[string]bool{},
	}
	for _, pkg := range p.Packages {
		set := map[string]bool{}
		ix.pkgs[strings.ToLower(pkg.Name)] = true
		short := pkg.Dir
		if i := strings.LastIndex(short, "/"); i >= 0 {
			short = short[i+1:]
		}
		if short != "" {
			ix.pkgs[strings.ToLower(short)] = true
		}
		for _, fn := range pkg.Funcs {
			ix.funcs[strings.ToLower(fn.Name)] = true
			set[strings.ToLower(fn.Name)] = true
		}
		for _, t := range pkg.Types {
			ix.types[strings.ToLower(t.Name)] = true
			set[strings.ToLower(t.Name)] = true
		}
		ix.pkgNames[strings.ToLower(pkg.Name)] = set
		if short != "" {
			ix.pkgNames[strings.ToLower(short)] = set
		}
		for _, f := range pkg.Files {
			ix.files[f] = true
			base := f
			if i := strings.LastIndex(base, "/"); i >= 0 {
				base = base[i+1:]
			}
			ix.basenames[strings.ToLower(base)] = true
		}
	}
	return ix
}

func (ix *talkIndex) hasName(name string) bool {
	lower := strings.ToLower(name)
	return ix.funcs[lower] || ix.types[lower]
}

var (
	talkBacktickRe = regexp.MustCompile("`([^`]+)`")
	talkCallRe     = regexp.MustCompile(`\b([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	talkDottedRe   = regexp.MustCompile(`\b([a-z][A-Za-z0-9_]*)\.([A-Z][A-Za-z0-9_]*)`)
	talkGoPathRe   = regexp.MustCompile(`([A-Za-z0-9_./-]+\.go)\b`)
	talkIdentRe    = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.]*$`)
)

// talkBuiltins are Go's predeclared functions and language
// keywords: a Name( shape naming one of these is a language
// fact, never a project claim.
var talkBuiltins = map[string]bool{
	"append": true, "cap": true, "clear": true, "close": true,
	"complex": true, "copy": true, "delete": true, "imag": true,
	"len": true, "make": true, "max": true, "min": true,
	"new": true, "panic": true, "print": true, "println": true,
	"real": true, "recover": true,
	"if": true, "for": true, "func": true, "return": true,
	"switch": true, "select": true, "case": true, "go": true,
	"defer": true, "range": true, "type": true, "var": true,
}

// talkCodeyName reports whether an identifier is code-shaped
// enough to be a checkable claim: it starts with a capital,
// or carries a capital or an underscore. A plain lowercase
// English word is prose, not a project reference.
func talkCodeyName(name string) bool {
	if name == "" {
		return false
	}
	if name[0] >= 'A' && name[0] <= 'Z' {
		return true
	}
	return strings.ContainsAny(name, "_") || strings.ToLower(name) != name
}

// talkInventedRefs returns the project-shaped references in
// one sentence that the index does not hold.
func talkInventedRefs(sentence string, ix *talkIndex) []string {
	if ix == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	add := func(ref string, invented bool) {
		if invented && !seen[ref] {
			seen[ref] = true
			out = append(out, ref)
		}
	}
	// Backticked tokens: the strongest claim shape — the
	// writer set the name apart as code.
	for _, m := range talkBacktickRe.FindAllStringSubmatch(sentence, -1) {
		tok := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(m[1]), "()"))
		switch {
		case strings.HasSuffix(tok, ".go"):
			add(tok, !talkPathExists(ix, tok))
		case strings.Contains(tok, ".") && talkIdentRe.MatchString(tok):
			parts := strings.SplitN(tok, ".", 2)
			add(tok, !talkDottedExists(ix, parts[0], parts[1]))
		case talkIdentRe.MatchString(tok) && talkCodeyName(tok):
			add(tok, !ix.hasName(tok))
		}
	}
	// Name( call shapes. A name immediately preceded by a
	// dot is a selector call (mu.Lock(), http.HandleFunc):
	// its existence claim attaches to the receiver, which
	// the pkg.Name rule below adjudicates for project
	// packages and correctly leaves alone for external ones —
	// so the bare-call rule skips it.
	for _, m := range talkCallRe.FindAllStringSubmatchIndex(sentence, -1) {
		name := sentence[m[2]:m[3]]
		if m[0] > 0 && sentence[m[0]-1] == '.' {
			continue
		}
		// A paren that opens with a digit is a count in a
		// label ("Fields (1): Prefix"), not a call.
		rest := strings.TrimLeft(sentence[m[1]:], " ")
		if len(rest) > 0 && rest[0] >= '0' && rest[0] <= '9' {
			continue
		}
		if talkBuiltins[strings.ToLower(name)] || !talkCodeyName(name) {
			continue
		}
		add(name, !ix.hasName(name))
	}
	// pkg.Name shapes: checked only when pkg is one of this
	// project's own packages — anything else (fmt, http,
	// sync) is general knowledge, not a project claim.
	for _, m := range talkDottedRe.FindAllStringSubmatch(sentence, -1) {
		pkg, name := m[1], m[2]
		if !ix.pkgs[strings.ToLower(pkg)] {
			continue
		}
		add(m[0], !talkDottedExists(ix, pkg, name))
	}
	// File paths ending in .go.
	for _, m := range talkGoPathRe.FindAllStringSubmatch(sentence, -1) {
		add(m[1], !talkPathExists(ix, m[1]))
	}
	return out
}

// talkDottedExists reports whether pkg.Name resolves: for a
// project package, Name must be one of its functions or
// types; for any other package the shape is not a project
// claim and counts as existing (unchecked).
func talkDottedExists(ix *talkIndex, pkg, name string) bool {
	set, ok := ix.pkgNames[strings.ToLower(pkg)]
	if !ok {
		return true
	}
	return set[strings.ToLower(name)]
}

// talkPathExists reports whether a .go path resolves against
// the project's file list: a path with a folder must match
// exactly — or be the project's own LinkPath rendering of a
// real file (a cwd-relative path whose tail is the project
// path, "../../../../tmp/x/store/store.go" for
// "store/store.go"); a bare file name must exist somewhere
// in the project.
func talkPathExists(ix *talkIndex, rel string) bool {
	rel = strings.TrimPrefix(rel, "./")
	if strings.Contains(rel, "/") {
		if ix.files[rel] {
			return true
		}
		for f := range ix.files {
			if strings.HasSuffix(rel, "/"+f) {
				return true
			}
		}
		return false
	}
	return ix.basenames[strings.ToLower(rel)]
}

// splitTalkSentences splits a reply into sentences, keeping
// each sentence's terminator. A boundary is a terminator
// followed by real whitespace (or the end of the text): the
// dots inside code references (greet.Hello, cache/magic.go)
// are never followed by whitespace, so they can never split
// a sentence — or be mangled by the rejoin.
func splitTalkSentences(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] != '.' && s[i] != '!' && s[i] != '?' {
			continue
		}
		j := i + 1
		for j < len(s) && (s[j] == ' ' || s[j] == '\n' || s[j] == '\t') {
			j++
		}
		if j > i+1 && j < len(s) {
			if t := strings.TrimSpace(s[start : i+1]); t != "" {
				out = append(out, t)
			}
			start = j
			i = j - 1
		}
	}
	if t := strings.TrimSpace(s[start:]); t != "" {
		out = append(out, t)
	}
	return out
}

// markTalkSentence appends the marker to one sentence,
// inside its terminator when it has one.
func markTalkSentence(sentence string) string {
	t := strings.TrimSpace(sentence)
	if t == "" {
		return t
	}
	last := t[len(t)-1]
	if last == '.' || last == '!' || last == '?' {
		return t[:len(t)-1] + talkInventedMarker + string(last)
	}
	return t + talkInventedMarker
}

// talkSentenceIsDenial reports whether a sentence mentions
// its names under an explicit denial ("I don't see a file
// called cache/magic.go in this project") — a mention being
// denied is not a claim printed as fact, in the checker and
// in the probe bar alike.
func talkSentenceIsDenial(s string) bool {
	lower := strings.ToLower(s)
	return strings.Contains(lower, "don't see") || strings.Contains(lower, "do not see") ||
		strings.Contains(lower, "doesn't exist") || strings.Contains(lower, "does not exist") ||
		strings.Contains(lower, "isn't in") || strings.Contains(lower, "is not in")
}

// talkCheckReply verifies one generated reply against the
// project index and returns the text that may print: clean
// sentences untouched, a sentence naming an invented project
// thing marked, and a bare fragment sentence (two words or
// fewer — nothing left to mark without mangling) suppressed.
// With no index there is nothing to verify against and the
// reply passes as-is; the seat's fact arms need an index too,
// so this only ever covers free social prose in a project the
// analyzer could not read.
func talkCheckReply(reply string, p *analyze.Project) string {
	ix := buildTalkIndex(p)
	if ix == nil {
		return reply
	}
	sentences := splitTalkSentences(reply)
	kept := make([]string, 0, len(sentences))
	for _, s := range sentences {
		if talkSentenceIsDenial(s) || len(talkInventedRefs(s, ix)) == 0 {
			kept = append(kept, s)
			continue
		}
		if len(strings.Fields(s)) <= 2 {
			continue // suppress: marking a bare fragment would mangle it
		}
		kept = append(kept, markTalkSentence(s))
	}
	return strings.Join(kept, " ")
}

// talkUnmarkedInventions is the probe harness's mechanical
// bar: the invented references in a PRINTED reply whose
// sentences do not carry the marker. Zero is the ship bar.
func talkUnmarkedInventions(reply string, p *analyze.Project) []string {
	ix := buildTalkIndex(p)
	if ix == nil {
		return nil
	}
	var out []string
	for _, s := range splitTalkSentences(reply) {
		if strings.Contains(s, talkInventedMarker) || talkSentenceIsDenial(s) {
			continue
		}
		out = append(out, talkInventedRefs(s, ix)...)
	}
	return out
}
