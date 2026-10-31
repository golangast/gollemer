package chat

// The concept book + composer: the write family's answer to
// grammar-per-kind sprawl. Concepts — the basic pieces of Go a
// beginner asks to add, by NAME — live as trained DATA
// (data/concepts.json in the repo, loaded like the Advocates
// snapshot; the planner and row format are
// internal/ai/analyze/editconcept.go), and ONE composed grammar
// here matches a sentence against the loaded book, binds its
// slots, and hands the result to the PlanEdit spine. Adding a
// concept is adding a row: the day a "for loop" row lands in the
// book, "add a for loop to main.go" works with no new Go code
// (proven by the fixture-book test). The bespoke shapes keep
// their own sentences — this claim sits after them and before
// the write-sentence boundary, so known concepts are served and
// unknown ones still land on the boundary's honest answer.
// Neural generation stays out of it: the book is recall data,
// assembled deterministically, like the makefile brain's pairs.
//
// Grammar (slots may flex in the family's order):
//
//	add|put [a|an|the] <concept name from the book, longest
//	  match> [named <Name> | name is <Name>] [saying "<text>"]
//	  [like <Type>] [in <Func>] to|in|into [the] [file|folder]
//	  <target>
//
// "saying" may also be spelled "that says". The "in <Func>"
// qualifier is read only when a separate target clause follows
// it — a lone "in <word>" is the target, exactly as the
// add-server grammar reads it. A concept matched with no target
// (or a declaration with no name) is still claimed: the composer
// answers with the guidance naming the working sentence, in the
// nameless-designator tradition. Unknown concept names are NOT
// claimed here at all — they fall to the boundary.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// defaultConceptBook mirrors data/concepts.json: the snapshot in
// the repo is the editable source, and this compiled copy is
// what a chat started outside the repo (where no data/ file
// exists) falls back to. A test pins the two to the same rows so
// they cannot drift.
const defaultConceptBook = `{
  "concepts": [
    {
      "id": "print-line",
      "names": ["print line", "print statement", "println", "print"],
      "kind": "statement",
      "template": "fmt.Println(\"{text}\")",
      "imports": ["fmt"]
    },
    {
      "id": "interface",
      "names": ["interface"],
      "kind": "declaration",
      "template": "type {name} interface {\n{methods}\n}"
    },
    {
      "id": "struct",
      "names": ["struct", "struct type"],
      "kind": "declaration",
      "template": "type {name} struct {\n\t// TODO: add fields — or say \"add a <Name> field of type <Type> to {name}\"\n}"
    },
    {
      "id": "function",
      "names": ["function", "func"],
      "kind": "declaration",
      "template": "func {name}() {\n\t// TODO: write the body\n}"
    }
  ]
}`

var (
	conceptOnce  = &sync.Once{} // a pointer so tests can swap in a fresh one
	conceptCache []analyze.ConceptEntry
)

// conceptEntries lazily loads the book: data/concepts.json from
// the chat's project root (or the working directory) first —
// the advocates precedent — else the compiled default. A book
// whose rows are all malformed degrades to the default too:
// ParseConceptBook skips bad rows, never the whole book.
func conceptEntries() []analyze.ConceptEntry {
	conceptOnce.Do(func() {
		for _, base := range []string{unifiedProjectRoot, startFolder()} {
			if base == "" {
				continue
			}
			blob, err := os.ReadFile(filepath.Join(base, "data", "concepts.json"))
			if err != nil {
				continue
			}
			if entries, _ := analyze.ParseConceptBook(blob); len(entries) > 0 {
				conceptCache = entries
				return
			}
		}
		conceptCache, _ = analyze.ParseConceptBook([]byte(defaultConceptBook))
	})
	return conceptCache
}

// conceptName is one phrasing of one entry, word-split for the
// longest-match walk.
type conceptName struct {
	words []string
	entry int // index into the loaded entries
}

// conceptNames flattens the book's phrasings, longest first (so
// "print line" beats its own prefix "print"), ties broken by
// phrase for determinism.
func conceptNames(entries []analyze.ConceptEntry) []conceptName {
	var out []conceptName
	for i, e := range entries {
		for _, n := range e.Names {
			out = append(out, conceptName{words: strings.Fields(n), entry: i})
		}
	}
	sort.Slice(out, func(a, b int) bool {
		if len(out[a].words) != len(out[b].words) {
			return len(out[a].words) > len(out[b].words)
		}
		return strings.Join(out[a].words, " ") < strings.Join(out[b].words, " ")
	})
	return out
}

// conceptReq is one parsed composer sentence: the matched book
// entry and every bound slot.
type conceptReq struct {
	entry   analyze.ConceptEntry
	name    string // {name}: "named <Name>" or "name is <Name>"
	text    string // {text}: the saying slot's string value
	textSet bool
	inFunc  string // statement target function: "in <Func>"
	like    string // {methods} source type: "like <Type>"
	target  string // the file/folder as typed; "" = unnamed
}

var reConceptSaying = regexp.MustCompile(`(?i)\b(?:saying|that says|says)\s+"((?:[^"\\]|\\.)*)"`)

// parseConcept reads one line as a concept-book ask against the
// loaded book. The second return is false for anything outside
// the composed grammar — including every bespoke shape's
// sentence (no book name matches their object words) and any
// line whose slot zone carries a word the grammar doesn't know
// (unknown qualifiers are never silently dropped).
func parseConcept(line string) (conceptReq, bool) {
	return parseConceptFrom(line, conceptEntries())
}

// parseConceptFrom is parseConcept against a given book — the
// seam the data-driven tests drive a fixture book through.
func parseConceptFrom(line string, entries []analyze.ConceptEntry) (conceptReq, bool) {
	s := strings.TrimSpace(line)

	// The saying slot comes out first, quoted text intact: its
	// value is a Go string's content, unescaped here and
	// re-escaped by the planner's renderer.
	var req conceptReq
	if m := reConceptSaying.FindStringSubmatchIndex(s); m != nil {
		inner := s[m[2]:m[3]]
		req.text = inner
		if uq, err := strconv.Unquote(`"` + inner + `"`); err == nil {
			req.text = uq
		}
		req.textSet = true
		s = strings.TrimSpace(s[:m[0]] + " " + s[m[1]:])
	}
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return conceptReq{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	if low[0] != "add" && low[0] != "put" {
		return conceptReq{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	if i >= len(fields) {
		return conceptReq{}, false
	}

	// The concept name: the book's longest phrasing that matches
	// at this position. No match, no claim — an unknown concept
	// belongs to the boundary.
	matched := false
	for _, cn := range conceptNames(entries) {
		if i+len(cn.words) > len(fields) {
			continue
		}
		ok := true
		for j, w := range cn.words {
			if low[i+j] != w {
				ok = false
				break
			}
		}
		if ok {
			req.entry = entries[cn.entry]
			i += len(cn.words)
			matched = true
			break
		}
	}
	if !matched {
		return conceptReq{}, false
	}

	// Split what remains into the slot zone and the target
	// clause. The target starts at its preposition; an "in"
	// counts only when what follows it reads as a target (a
	// .go file, a path, or — when no later target clause
	// exists — the line's tail), so "in main to jim/main.go"
	// leaves "in main" in the slot zone.
	tIdx := -1
	for k := i; k < len(fields); k++ {
		switch low[k] {
		case "to", "into":
			tIdx = k
		case "in":
			j := k + 1
			if j < len(fields) && low[j] == "the" {
				j++
			}
			if j < len(fields) && conceptDesignator(low[j]) {
				j++
			}
			switch {
			case j >= len(fields):
				tIdx = k // "… in folder" — claimed, nameless
			case strings.HasSuffix(low[j], ".go") || strings.Contains(fields[j], "/"):
				tIdx = k
			case !conceptHasLaterTarget(low, k+1):
				tIdx = k // lone "in <word>": the target (add-server rule)
			}
		}
		if tIdx >= 0 {
			break
		}
	}
	slotEnd := len(fields)
	if tIdx >= 0 {
		slotEnd = tIdx
	}

	// The slot zone: named <Name> | name is <Name> | like
	// <Type> | in <Func>, in any order the family allows.
	// Anything else refuses the claim.
	for j := i; j < slotEnd; {
		switch low[j] {
		case "named":
			if j+1 >= slotEnd {
				return conceptReq{}, false
			}
			req.name = fields[j+1]
			j += 2
		case "name":
			// "name is <Name>": the name slot spelled out,
			// the same binding as "named", one word longer.
			if j+2 >= slotEnd || low[j+1] != "is" {
				return conceptReq{}, false
			}
			req.name = fields[j+2]
			j += 3
		case "like":
			if j+1 >= slotEnd {
				return conceptReq{}, false
			}
			req.like = fields[j+1]
			j += 2
		case "in":
			if j+1 >= slotEnd {
				return conceptReq{}, false
			}
			req.inFunc = fields[j+1]
			j += 2
		default:
			return conceptReq{}, false
		}
	}

	// The target clause: article and designator are slots, never
	// part of the path (the add-server rule).
	if tIdx >= 0 {
		j := tIdx + 1
		if j < len(fields) && low[j] == "the" {
			j++
		}
		if j < len(fields) && conceptDesignator(low[j]) {
			j++
		}
		if j < len(fields) {
			req.target = strings.Join(fields[j:], " ")
		}
	}

	// Two bound slots can still make the line another shape's:
	// a "like" the entry's template can't spend (it has no
	// {methods} slot) would be silently dropped — and "add a
	// function like Greet named GreetLoudly" is the CLONE's
	// sentence, not an empty scaffold's. Likewise a "that" in
	// the target tail is a behaviour clause, not a path: "add
	// a function in jim/main.go that calls Load" is the
	// DELEGATE's sentence, and its tail is not a file. Both
	// refuse the claim; qualifiers are never silently dropped
	// and behaviour clauses are never read as paths.
	if req.like != "" && !strings.Contains(req.entry.Template, "{methods}") {
		return conceptReq{}, false
	}
	for _, w := range strings.Fields(strings.ToLower(req.target)) {
		if w == "that" {
			return conceptReq{}, false
		}
	}
	return req, true
}

// conceptHasLaterTarget reports whether a target clause starts
// somewhere at or after index from: a to|into, or an "in" whose
// following word is a .go file or a path.
func conceptHasLaterTarget(low []string, from int) bool {
	for k := from; k < len(low); k++ {
		if low[k] == "to" || low[k] == "into" {
			return true
		}
		if low[k] == "in" && k+1 < len(low) &&
			(strings.HasSuffix(low[k+1], ".go") || strings.Contains(low[k+1], "/")) {
			return true
		}
	}
	return false
}

// conceptArticle picks a/an for a concept's canonical name.
func conceptArticle(name string) string {
	if name != "" && strings.ContainsRune("aeiou", rune(name[0])) {
		return "an"
	}
	return "a"
}

// conceptDesignator reports whether w is a target kind word
// (file|folder|directory, a trailing slash tolerated).
func conceptDesignator(w string) bool {
	switch strings.TrimSuffix(w, "/") {
	case "file", "folder", "directory":
		return true
	}
	return false
}

// tryConceptBook is the chat-loop entry: it claims concept-book
// sentences and runs the flow on the chat's line reader.
func tryConceptBook(line string, r *LineReader) bool {
	return runConcept(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runConcept carries one concept sentence from the book match
// to the shared edit spine (or to composer's guidance when the
// sentence is missing a piece no plan can supply).
func runConcept(line string, ask editAsk) bool {
	req, ok := parseConcept(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	canon := req.entry.Canonical()

	// Guidance before any project work: a declaration with no
	// name, and any concept with no target, are answered in
	// composer terms — the sentence was this claim's.
	if req.entry.Kind == analyze.ConceptDeclaration && req.name == "" {
		if req.entry.ID == "function" {
			// A nameless function ask is a near-miss of THREE
			// shapes, not one: this claim's scaffold, the
			// delegate, and the clone. The guidance names all
			// three, with the user's own target filled in.
			file := req.target
			if file == "" {
				file = "main.go"
			}
			fmt.Printf("%s Not adding that: a function needs a name — say \"add a function named <Name> to %s\" for an empty scaffold you fill in. If you want it doing real work from the start, say \"add a function in %s that calls <something that exists>\" or \"copy <existing function> as <new name>\".\n", tag, file, file)
			recordBrain("composer", false, "concept refused: function unnamed")
			return true
		}
		fmt.Printf("%s Not adding that: %s %s needs a name — say \"add a %s named <Name> to <file>\", like \"add a %s named Reader to main.go\".\n", tag, conceptArticle(canon), canon, canon, canon)
		recordBrain("composer", false, "concept refused: declaration unnamed")
		return true
	}
	if req.target == "" {
		fmt.Printf("%s Not adding that: the sentence doesn't name the file or folder — say it like \"add a %s to jim/main.go\" (a file) or \"add a %s to jim\" (its folder).\n", tag, canon, canon)
		recordBrain("composer", false, "concept refused: no target named")
		return true
	}

	// Target resolution and project acquisition: the
	// add-server flow's own rules, verbatim in spirit — the
	// creations fence's root, its folder-to-file resolver, its
	// quiet analyze / freshness guard.
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "concept refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "concept refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "concept refused: root missing")
		return true
	}
	rel, refusal := resolveServerFile(rootAbs, req.target)
	if refusal != "" {
		fmt.Printf("%s Not adding that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "concept refused: target")
		return true
	}
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		if !analyzeQuiet(rootAbs) {
			recordBrain("composer", false, "concept refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "concept refused: stale project unreadable")
		return true
	}
	entry := req.entry
	return executeEditPlan(analyze.EditRequest{
		Kind:        analyze.EditConcept,
		File:        rel,
		Concept:     &entry,
		ConceptName: req.name,
		ConceptText: req.text,
		ConceptFunc: req.inFunc,
		ConceptLike: req.like,
	}, ask)
}
