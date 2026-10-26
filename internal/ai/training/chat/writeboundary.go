package chat

// The write-sentence boundary: the last deterministic claim in
// the unified loop, after every servable write shape (assembled
// edits, rename, creations, add-server, delegate) and before the
// intent layer. It exists because of a live failure on the
// shipped stack (2026-10-25 20:11): in a fresh chat,
//
//	add print line to jim/main.go   → gollemer [gocode>social]>
//	                                  "salutations! even a way,
//	                                  dusty shelf kind."
//	add a function to jim/main.go   → gollemer [gocode>social]>
//	                                  "time, don'T mood book, and
//	                                  an extra early!"
//
// Both are sentences asking to WRITE code that no shipped shape
// can serve, and left unclaimed they fell through every parser
// to the generative brains, which can only improvise. The
// settled principle (the nameless-designator claim of 9c369b0,
// generalized): a write-family sentence that can't be served is
// claimed here and answered in composer terms — near-miss
// guidance naming the exact sentences that DO work, or an honest
// boundary naming the closest real alternatives, in the
// creations boundary voice (create.go's parseBoundary). Nothing
// here writes, plans, or gates: it only talks.
//
// The trigger is calibrated against the parsers' own corpora
// (probe-verified at the base: every guidance case below parses
// false in parseAssembledEdit, parseRenameSentence,
// parseBuildGoal, parseCreate, parseBoundary, parseAddServer,
// and parseDelegate alike):
//
//   - led by a write verb (add|put|insert, plus create|write
//     when a file target is named — those two verbs also lead
//     brain-owned generation asks, which keep their brains), AND
//   - naming a .go file or a path-like target, or naming a code
//     object from the set below — unless the line carries a
//     "that …" behaviour clause and no file/path target, which
//     is the generation family's shape ("write a function that
//     reverses a string" is the gocode brain's line, pinned in
//     intent_test.go and real_seq2seq_chat_test.go).
//
// Questions never start with the bare verb ("how do i add…"
// keeps its owner), and the object set deliberately omits the
// words other layers own without a file target: getter/test/
// comment sentences belong to the intent layer and /contribute
// (they are pinned as other brains' sentences in edits_test.go
// and rename_test.go), and "add it to the list of things to do"
// names no code object at all.

import (
	"fmt"
	"strings"
)

// writeBoundaryKind names which answer a claimed line gets.
type writeBoundaryKind string

const (
	wbFunction  writeBoundaryKind = "function near-miss"
	wbField     writeBoundaryKind = "field near-miss"
	wbParameter writeBoundaryKind = "parameter near-miss"
	wbServer    writeBoundaryKind = "webserver near-miss"
	wbPrintLine writeBoundaryKind = "print line boundary"
	wbGeneric   writeBoundaryKind = "no-shape boundary"
)

// writeBoundaryReq is one claimed line, parsed only far enough to
// answer it: the answer kind, the code object as typed (generic
// answers name it back), the target as typed ("" when none), and
// whether that target is a .go file (the file-shaped answers and
// alternatives key off it).
type writeBoundaryReq struct {
	kind      writeBoundaryKind
	object    string // the object phrase as typed, e.g. "an interface"
	target    string // the target as typed, e.g. "jim/main.go"
	fileTgt   bool   // target is (or ends in) a .go file
	hasTarget bool
}

var writeVerbs = map[string]bool{
	"add": true, "put": true, "insert": true, "create": true, "write": true,
}

// writeObjectWords are the code objects whose naming makes an
// imperative line a write ask aimed at code. See the header for
// the words deliberately left out.
var writeObjectWords = map[string]bool{
	"function": true, "func": true, "method": true,
	"line": true, "statement": true, "print": true, "println": true,
	"field": true, "case": true, "import": true, "handler": true,
	"interface": true, "struct": true, "type": true,
	"variable": true, "parameter": true,
	"server": true, "webserver": true,
	"switch": true, "loop": true, "map": true, "slice": true,
	"array": true, "channel": true, "goroutine": true,
	"const": true, "var": true,
}

// parseWriteBoundary reads one line as an unservable write ask.
// The second return is false for anything the boundary must not
// claim: questions, ordinary chat, brain-owned generation asks,
// and — by loop position — every sentence a real shape serves
// (those never reach this parser in the loop; the parse-level
// non-claims below are the belt to that ordering's braces).
func parseWriteBoundary(line string) (writeBoundaryReq, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return writeBoundaryReq{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(strings.Trim(f, ",;:\"'()"))
	}
	if !writeVerbs[low[0]] {
		return writeBoundaryReq{}, false
	}

	// Evidence: a .go file target, a path-like target, or a code
	// object. A "that …" clause with no file/path target is a
	// generation ask — the brains' line, not an edit's.
	fileTgt, pathTgt, objAt := false, false, -1
	for i, w := range low {
		if strings.HasSuffix(w, ".go") {
			fileTgt = true
		}
		if strings.Contains(fields[i], "/") {
			pathTgt = true
		}
		if objAt < 0 && writeObjectWords[w] {
			objAt = i
		}
	}
	hasThat := false
	for _, w := range low[1:] {
		if w == "that" {
			hasThat = true
		}
	}
	switch {
	case fileTgt || pathTgt:
		// A file was named: an edit ask whatever else it says.
	case objAt >= 0 && !hasThat:
		// A code object with a destination or standing alone.
	default:
		return writeBoundaryReq{}, false
	}
	// create|write also lead pure generation asks ("write a
	// function that reverses a string", "create a function for
	// my app"): those verbs claim only when a file is named.
	if (low[0] == "create" || low[0] == "write") && !fileTgt && !pathTgt {
		return writeBoundaryReq{}, false
	}

	req := writeBoundaryReq{fileTgt: fileTgt}
	req.target, req.hasTarget = writeBoundaryTarget(fields, low)

	// The object phrase as typed, with its article when it has
	// one ("an interface"); for the print line, both words.
	if objAt >= 0 {
		start := objAt
		if objAt > 0 && (low[objAt-1] == "a" || low[objAt-1] == "an" || low[objAt-1] == "the") {
			start = objAt - 1
		}
		end := objAt + 1
		if low[objAt] == "print" && objAt+1 < len(low) && low[objAt+1] == "line" {
			end = objAt + 2
		}
		req.object = strings.Join(fields[start:end], " ")
	}

	// Answer kind, near-misses first. Function, field, and
	// parameter are the shapes a sentence can almost spell;
	// print is the approved concept-book boundary; server is
	// the webserver component misnamed. (Case is NOT here: the
	// assembled parser already claims its under-specified forms
	// — probe-verified, "add a case to Describe" parses as an
	// edit — so its refusals are the planner's own.)
	has := func(words ...string) bool {
		for _, w := range low {
			for _, want := range words {
				if w == want {
					return true
				}
			}
		}
		return false
	}
	switch {
	case has("function", "func"):
		req.kind = wbFunction
	case has("field"):
		req.kind = wbField
	case has("parameter"):
		req.kind = wbParameter
	case has("print", "println"):
		req.kind = wbPrintLine
	case has("server", "webserver"):
		req.kind = wbServer
	default:
		req.kind = wbGeneric
	}
	return req, true
}

// writeBoundaryTarget pulls the target as typed out of a claimed
// line: a .go token wherever it sits, else the words after the
// last to|in|into with any file/folder designator stripped (the
// add-server rule: designators name the kind, never the path).
func writeBoundaryTarget(fields, low []string) (string, bool) {
	for i, w := range low {
		if strings.HasSuffix(w, ".go") {
			return strings.Trim(fields[i], ",;:\"'()"), true
		}
	}
	at := -1
	for i := 1; i < len(low); i++ {
		if low[i] == "to" || low[i] == "in" || low[i] == "into" {
			at = i
		}
	}
	if at < 0 || at+1 >= len(fields) {
		return "", false
	}
	rest := fields[at+1:]
	restLow := low[at+1:]
	j := 0
	for j < len(rest) && (restLow[j] == "the" || restLow[j] == "file" || restLow[j] == "folder" || restLow[j] == "directory") {
		j++
	}
	if j >= len(rest) {
		return "", false
	}
	return strings.Join(rest[j:], " "), true
}

// exampleFile is the target the guidance sentences are spelled
// with: the user's own when they named a file, else the family's
// standing example.
func (r writeBoundaryReq) exampleFile() string {
	if r.fileTgt && r.hasTarget {
		return r.target
	}
	return "main.go"
}

// closestAlternatives names the two real moves nearest a
// file-targeted ask: the webserver into that file, or a function
// copied from a real one there. Without a file target it names
// the two name-targeted moves instead: the clone and the field.
func (r writeBoundaryReq) closestAlternatives() string {
	if r.fileTgt {
		f := r.exampleFile()
		return fmt.Sprintf("What I can add to %s today: a webserver — \"add a webserver to %s\" — or a function copied from a real one — \"add a function in %s that calls <Name>\"", f, f, f)
	}
	return "What I can add today: a function copied from a real one — \"copy <Name> as <NewName>\" — or a field on a struct — \"add a <Name> field of type <type> to <Struct>\""
}

// answer renders the boundary reply for one claimed line.
func (r writeBoundaryReq) answer() string {
	switch r.kind {
	case wbFunction:
		return fmt.Sprintf("A function needs a source I can copy from — I can't write one from nothing. The two that work: \"add a function in %s that calls <something that exists>\" (a new function wrapping a real one) or \"copy <existing function> as <new name>\" (a clone you then edit)", r.exampleFile())
	case wbField:
		head := "A field goes into a named struct, not straight into a file"
		if r.fileTgt && r.hasTarget {
			head = fmt.Sprintf("A field goes into a struct — %s alone doesn't tell me which struct should carry it", r.target)
		}
		return head + ". Say it like \"add a <Name> field of type <type> to <Struct>\", e.g. \"add a Timeout field of type int to Config\""
	case wbParameter:
		return "A parameter needs more than its place: a type, a name, and the value every caller will pass. The forms that work: \"add a context parameter to <Func>\", or \"add a <type> parameter named <name> to <Func> using <expression>\""
	case wbServer:
		f := r.exampleFile()
		return fmt.Sprintf("The piece's name here is webserver: \"add a webserver to %s\" puts one in an existing file, or \"create a file %s with a webserver\" writes a new one", f, f)
	case wbPrintLine:
		return fmt.Sprintf("I can't add a print line yet — a loose statement isn't one of the pieces I can assemble today (it's first in line for the concept book). %s", r.closestAlternatives())
	default:
		obj := r.object
		if obj == "" {
			obj = "that"
		}
		return fmt.Sprintf("I can't add %s yet — that's not one of the pieces I can assemble today. %s", obj, r.closestAlternatives())
	}
}

// tryWriteBoundary is the chat-loop entry: it claims unservable
// write sentences and answers them in composer terms.
func tryWriteBoundary(line string) bool {
	return runWriteBoundary(line)
}

// runWriteBoundary answers one line when the boundary claims it.
// It never asks anything and never touches a file.
func runWriteBoundary(line string) bool {
	req, ok := parseWriteBoundary(line)
	if !ok {
		return false
	}
	fmt.Printf("%s %s.\n", cTag("gollemer [composer]>"), strings.TrimSuffix(req.answer(), "."))
	recordBrain("composer", true, "write-boundary answer: "+string(req.kind)+" (not performed)")
	return true
}
