package chat

// Function-edit sentences about an EXISTING project function —
// "remove the function named jake", "have the function jake
// return a string" — claimed deterministically in the unified
// loop directly beside the goroutine-result piece, whose
// sibling this grammar is ("have the goroutine return a
// string" is a landing; "have the function jake return a
// string" is the same sentence aimed at a named function).
//
// Why this claim exists: John's live failure, 2026-11-21. In
// a session where he had just created func jake in
// jake/main.go, "remove the function named jake" fell through
// every claim to the talk brain, which answered a project
// sentence with a social greeting ("good to see you. how's
// the house going?"), and "have the function jake return a
// string" reached the intent layer, whose [go] brain answered
// an edit ask with a lecture on string comparison. Neither
// operation has a landing in the composer — there is no
// remove-function piece and no change-the-return-type piece
// (the signature machinery covers parameters only) — so this
// claim answers both in the composer's own voice: an honest
// decline that names the resolved function and its file, in
// the file-delete refusal's style (create.go's boundary:
// limit first, the fact of where the thing lives, then the
// nearest real help). Nothing here writes, plans, or gates.
//
// The grammar is resolution-gated, which is what makes the
// seat safe: a shape claims only when the named function
// actually resolves in the current project's index. A name
// that resolves nowhere declines SILENTLY and routing
// proceeds exactly as today, so a social line that happens
// to carry the word "function" is never stolen. Questions
// are vetoed outright — "what does jake return", "how does
// the function jake work", "does jake return a string?" keep
// their go/analyze owners — and a name that resolves to
// several functions gets the house clarifying question
// (clarify.go), never a guess.
//
// Grammar (trailing "." or "?" optional, one leading
// "can you"/"could you"/"please" allowed, as the neighbours
// allow):
//
//	remove|delete [the] function|func [named] <Name> [from|in <file>]
//	have|make [the] [function|func] <Name> return [a|an|the] <type>
//	change [the] [function|func] <Name> to return [a|an|the] <type>
//	<Name> should return [a|an|the] <type>
//
// <Name> is one identifier token (a dotted Type.Method name
// is allowed, as the rename grammar allows); <type> is one
// type-shaped token — a builtin, a dotted package type, a
// capitalized name, or a * / [] form — so "make jake return
// the favor" is not an edit shape. Rename sentences are NOT
// here: "rename the function jake to john" already lands in
// the rename claim above, verified against its grammar.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// funcEditKind names which unserved operation a claimed
// line asked for.
type funcEditKind string

const (
	funcEditRemove funcEditKind = "remove"
	funcEditReturn funcEditKind = "return"
)

// funcEditRequest is one parsed function-edit ask.
type funcEditRequest struct {
	kind funcEditKind
	name string // the function name as typed
	typ  string // the return type as typed (return kind only)
	file string // the "from/in <file>" as typed (remove kind only; "" = none)
}

var reFuncEditName = regexp.MustCompile(`^[A-Za-z_][\w.]*$`)

// funcEditQuestionWords veto a parse when one leads the
// line (after any politeness): a question about a function
// belongs to the go/analyze brains, never to this claim.
// The shapes themselves already exclude these leads; the
// veto is the belt to their braces, named so the rule is
// visible.
var funcEditQuestionWords = map[string]bool{
	"what": true, "how": true, "why": true, "when": true,
	"where": true, "which": true, "who": true, "does": true,
	"do": true, "did": true, "is": true, "are": true,
	"was": true, "were": true,
}

// funcEditBuiltinTypes are the predeclared type words the
// return grammar accepts outright; any other type token
// must be shaped like one (capitalized, dotted, or a
// pointer/slice form) — see funcEditTypeShape.
var funcEditBuiltinTypes = map[string]bool{
	"string": true, "bool": true, "error": true, "any": true,
	"byte": true, "rune": true,
	"int": true, "int8": true, "int16": true, "int32": true, "int64": true,
	"uint": true, "uint8": true, "uint16": true, "uint32": true, "uint64": true,
	"uintptr": true, "float32": true, "float64": true,
	"complex64": true, "complex128": true,
}

// funcEditTypeShape reports whether tok can be the type in
// a "return a <type>" tail: a builtin, a dotted package
// type (time.Duration), a capitalized type name (Point),
// or a pointer/slice spelling (*Point, []string). A plain
// lowercase word that is not a builtin — "favor", " favor" —
// is not a type, and the sentence is not this grammar's.
func funcEditTypeShape(tok string) bool {
	if funcEditBuiltinTypes[strings.ToLower(tok)] {
		return true
	}
	if strings.HasPrefix(tok, "*") {
		return funcEditTypeShape(strings.TrimPrefix(tok, "*"))
	}
	if strings.HasPrefix(tok, "[]") {
		rest := strings.TrimPrefix(tok, "[]")
		return funcEditBuiltinTypes[strings.ToLower(rest)] || reFuncEditName.MatchString(rest)
	}
	if !reFuncEditName.MatchString(tok) {
		return false
	}
	if strings.Contains(tok, ".") {
		return true
	}
	return tok[0] >= 'A' && tok[0] <= 'Z'
}

// parseFuncEdit reads one chat line as a function-edit ask.
// The second return is false for anything outside the
// grammar (see the file header) — questions included.
func parseFuncEdit(line string) (funcEditRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return funcEditRequest{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	i := 0
	// One leading politeness shape, as the neighbours allow.
	if i+1 < len(fields) && (low[i] == "can" || low[i] == "could") && low[i+1] == "you" {
		i += 2
	} else if low[i] == "please" {
		i++
	}
	if i >= len(fields) || funcEditQuestionWords[low[i]] {
		return funcEditRequest{}, false
	}
	switch low[i] {
	case "remove", "delete":
		return parseFuncEditRemove(fields, low, i+1)
	case "have", "make", "change":
		return parseFuncEditReturn(fields, low, i, false)
	default:
		// "<Name> should return a <type>" — the name leads.
		return parseFuncEditReturn(fields, low, i, true)
	}
}

// parseFuncEditRemove parses from just after the verb:
// [the] function|func [named] <Name> [from|in <file>].
func parseFuncEditRemove(fields, low []string, i int) (funcEditRequest, bool) {
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i >= len(fields) || (low[i] != "function" && low[i] != "func") {
		return funcEditRequest{}, false
	}
	i++
	if i < len(fields) && low[i] == "named" {
		i++
	}
	if i >= len(fields) || !reFuncEditName.MatchString(fields[i]) {
		return funcEditRequest{}, false
	}
	req := funcEditRequest{kind: funcEditRemove, name: fields[i]}
	i++
	if i < len(fields) && (low[i] == "from" || low[i] == "in") {
		if i+1 >= len(fields) {
			return funcEditRequest{}, false
		}
		req.file = fields[i+1]
		i += 2
	}
	if i != len(fields) {
		return funcEditRequest{}, false
	}
	return req, true
}

// parseFuncEditReturn parses the return-type shapes. With
// nameFirst set, fields[i] is the name and "should return"
// follows; otherwise fields[i] is have|make|change and the
// name follows an optional [the] [function|func] (with
// "to" before "return" for change).
func parseFuncEditReturn(fields, low []string, i int, nameFirst bool) (funcEditRequest, bool) {
	var name string
	if nameFirst {
		if !reFuncEditName.MatchString(fields[i]) || low[i] == "the" {
			return funcEditRequest{}, false
		}
		name = fields[i]
		i++
		if i+1 >= len(fields) || low[i] != "should" || low[i+1] != "return" {
			return funcEditRequest{}, false
		}
		i += 2
	} else {
		verb := low[i]
		i++
		if i < len(fields) && low[i] == "the" {
			i++
		}
		if i < len(fields) && (low[i] == "function" || low[i] == "func") {
			i++
		}
		if i >= len(fields) || !reFuncEditName.MatchString(fields[i]) {
			return funcEditRequest{}, false
		}
		name = fields[i]
		i++
		if verb == "change" {
			if i >= len(fields) || low[i] != "to" {
				return funcEditRequest{}, false
			}
			i++
		}
		if i >= len(fields) || low[i] != "return" {
			return funcEditRequest{}, false
		}
		i++
	}
	if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
		i++
	}
	if i >= len(fields) || !funcEditTypeShape(fields[i]) {
		return funcEditRequest{}, false
	}
	typ := fields[i]
	i++
	if i != len(fields) {
		return funcEditRequest{}, false
	}
	return funcEditRequest{kind: funcEditReturn, name: name, typ: typ}, true
}

// tryFuncEdit is the chat-loop entry: it claims
// function-edit sentences and answers them on the chat's
// line reader (the clarifying question's answers arrive
// through the clarify seat, as the neighbours').
func tryFuncEdit(line string, r *LineReader) bool {
	return runFuncEdit(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runFuncEdit carries one line from sentence to resolution
// to the composer's answer. It returns true only when the
// line was this claim's: a parsed shape whose named
// function resolves in the current project. Every other
// outcome — no shape, no project, an unresolvable name —
// returns false and the line keeps the fate it has today.
func runFuncEdit(line string, ask editAsk) bool {
	req, ok := parseFuncEdit(line)
	if !ok {
		return false
	}
	cands := resolveFuncEditCandidates(req)
	switch {
	case cands == nil:
		// No project index to resolve against: the gate
		// cannot verify the function, so the catch stays
		// silent and routing proceeds as today.
		return false
	case len(cands) == 0:
		return false
	case len(cands) == 1:
		printFuncEditDecline(cTag("gollemer [composer]>"), req, cands[0])
		return true
	default:
		tag := cTag("gollemer [composer]>")
		cc := make([]clarifyCandidate, len(cands))
		for i, fn := range cands {
			cc[i] = clarifyCandidate{
				Label: fn.File,
				Desc:  fmt.Sprintf("line %d", fn.Line),
				Words: []string{funcEditFolderWord(fn.File)},
			}
		}
		askPickQuestion(tag,
			fmt.Sprintf("There's more than one function named %s — which one do you mean? Nothing's been written.", req.name),
			"Answer with the number or the file name",
			cc,
			func(idx int, ask2 editAsk) bool {
				printFuncEditDecline(tag, req, cands[idx])
				return true
			})
		recordBrain("composer", false, "func-edit question: which function")
		return true
	}
}

// resolveFuncEditCandidates loads (or refreshes) the
// current project's index the neighbours' way and returns
// every function the request names — honouring the
// sentence's own file when it gave one. A nil result means
// there is no index to resolve against (no project folder,
// or the project cannot be read); an empty result means
// the name resolves nowhere in it.
func resolveFuncEditCandidates(req funcEditRequest) []*analyze.Func {
	rootRaw := creationRoot()
	if rootRaw == "" {
		return nil
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			return nil
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		return nil
	}
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		p, err := analyze.Analyze(rootAbs)
		switch {
		case err == nil:
			setCurrentProject(rootRaw, p)
		case strings.Contains(err.Error(), "no Go packages found"):
			setCurrentProject(rootRaw, &analyze.Project{Root: rootAbs})
		default:
			return nil
		}
	} else if !ensureFreshProject() {
		return nil
	}
	var out []*analyze.Func
	for _, pkg := range lastAnalyzeProject.Packages {
		for _, fn := range pkg.Funcs {
			if !funcEditNameMatches(fn, req.name) {
				continue
			}
			if req.file != "" && fn.File != filepath.ToSlash(req.file) {
				continue
			}
			out = append(out, fn)
		}
	}
	return out
}

// funcEditNameMatches reports whether fn is the function
// the sentence named: a plain name matches case-insensitively
// (the rename resolution's rule), and a dotted name matches
// the receiver-qualified form (Store.Save).
func funcEditNameMatches(fn *analyze.Func, name string) bool {
	if !strings.Contains(name, ".") {
		return strings.EqualFold(fn.Name, name)
	}
	if fn.Receiver == "" {
		return false
	}
	recv := strings.TrimPrefix(fn.Receiver, "*")
	if i := strings.LastIndex(recv, "."); i >= 0 {
		recv = recv[i+1:]
	}
	return strings.EqualFold(recv+"."+fn.Name, name)
}

// funcEditFolderWord is a candidate's distinguishing word
// for the clarifying question: its file's first folder, or
// the file's base name at the project root.
func funcEditFolderWord(file string) string {
	if i := strings.Index(file, "/"); i >= 0 {
		return file[:i]
	}
	return strings.TrimSuffix(file, ".go")
}

// printFuncEditDecline renders the composer's honest answer
// for one resolved function: the limit first, the fact of
// where the function lives (and, for a return ask, its
// signature today), then the nearest real help — the
// file-delete refusal's shape.
func printFuncEditDecline(tag string, req funcEditRequest, fn *analyze.Func) {
	kindWord := "function"
	if fn.Receiver != "" {
		kindWord = "method"
	}
	where := fmt.Sprintf("%s is in %s (line %d)", fn.Name, fn.File, fn.Line)
	switch req.kind {
	case funcEditRemove:
		fmt.Printf("%s I can't remove a %s from your code — taking code out is yours alone, like deleting a file. %s: delete its declaration there in your editor, along with any lines that call it. What I can do: rename %s everywhere it appears, or explain what it does and what calls it.\n",
			tag, kindWord, where, fn.Name)
		recordBrain("composer", false, "func-edit refused: remove")
	case funcEditReturn:
		sig := ""
		if fn.Sig != "" {
			sig = fmt.Sprintf(", and its signature today is %s", fn.Sig)
		}
		fmt.Printf("%s I can't change what a %s returns yet — that edit isn't one of my pieces. You asked for %s to return a %s; %s%s. Making it return one means editing that declaration in your editor and returning a value wherever it finishes. What I can do: rename %s or one of its parameters, add a parameter to it, or explain how it's called today.\n",
			tag, kindWord, fn.Name, req.typ, where, sig, fn.Name)
		recordBrain("composer", false, "func-edit refused: return type")
	}
}
