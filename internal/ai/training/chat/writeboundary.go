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
//     intent_test.go and real_seq2seq_chat_test.go). One
//     exception, from the 2026-10-30 live miss ("add a function
//     named jake that returns a string" → social salad): a name
//     slot — "named <Name>" or "name is <Name>", the concept
//     grammar's own two spellings — BEFORE the clause makes the
//     line a declaration being specified, not behaviour being
//     requested, and the boundary claims it; a name inside the
//     clause ("…that returns a struct named Point") is the
//     clause's own business and still refuses. The name rides
//     into the answer with a file target too (2026-10-31:
//     the same line WITH "to jim/main.go" was claimed on its
//     target and answered generically — and misrouted by the
//     phrasing layer before it ever got here).
//
// The destination arm (2026-11-10): an add|put|insert line
// naming something and landing it in ONE bare destination
// word claims too, even when the thing is no code object and
// the destination carries no slash. "add templates in jim"
// reached the Uses brain ("How Add is really used") and "add
// header and footer templates to jim" the Comparer on
// 2026-11-10, live — both plainly unservable add asks. The
// arm's discipline: no "that …" clause (a behaviour clause
// keeps the line a generation ask), a destination of exactly
// one token, and an object zone that is a plain noun phrase
// — any pronoun or preposition inside it empties it, which
// is what keeps the header's own example unclaimed: "add it
// to the list of things to do" does have a one-word
// destination ("do", after its last "to"), but its zone is
// "it to the list of things", a pronoun and a prepositional
// chain, not a thing being added (writeBoundaryObjectZone);
// and the zone's head word is not one of the kinds the
// intent layer owns — comment, test, getter — the same
// exclusions the object set below keeps, so "add a comment
// to Describe" passes this arm by as it passes the set.
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
	objWord   string // the object word itself, lowered, e.g. "interface"
	target    string // the target as typed, e.g. "jim/main.go"
	fileTgt   bool   // target is (or ends in) a .go file
	hasTarget bool

	// fileItself marks a claimed line whose object is the
	// file itself: the token after the verb (and any article)
	// is "file", and the line carries a file/path target.
	// Such a line reached the boundary only because its tail
	// kept the create grammar from claiming it (a trailing
	// "please", a stray "to"); the generic answer for it
	// leads its suggestions with the create sentence for
	// that target — the file is made, not added to.
	fileItself bool

	// Set only for a line claimed with a name slot and a
	// "that …" clause — with or without a file/path target:
	// the declaration's name as typed, and whether the clause
	// is a returns clause. Those lines' answers spell the
	// working scaffold sentence with the user's own name (and
	// their own target, when they named one) instead of a
	// placeholder.
	declName      string
	returnsClause bool
}

var writeVerbs = map[string]bool{
	"add": true, "put": true, "insert": true, "create": true, "write": true,
}

// writeBoundaryOwnedHeads are the zone head words the
// destination arm never claims: the kinds other layers own
// without a file target — comment, test, and getter sentences
// belong to the intent layer and /contribute, the same
// exclusions writeObjectWords keeps (see the file header).
var writeBoundaryOwnedHeads = map[string]bool{
	"comment": true, "comments": true,
	"test": true, "tests": true,
	"getter": true, "getters": true, "setter": true, "setters": true,
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
	// generation ask — the brains' line, not an edit's — unless
	// a name slot before the clause marks the line as a
	// declaration being specified (see the header).
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
	thatAt := -1
	for i := 1; i < len(low); i++ {
		if low[i] == "that" {
			thatAt = i
			break
		}
	}
	hasThat := thatAt >= 0

	// The name slot, in the concept grammar's two spellings,
	// read only before the behaviour clause: the name token
	// itself must sit in that region (a dangling "named" right
	// before "that" names nothing).
	declName, hasDeclName := "", false
	limit := len(low)
	if hasThat {
		limit = thatAt
	}
	for i := 1; i < limit && !hasDeclName; i++ {
		switch {
		case low[i] == "named" && i+1 < limit:
			declName, hasDeclName = fields[i+1], true
		case low[i] == "name" && i+2 < limit && low[i+1] == "is":
			declName, hasDeclName = fields[i+2], true
		}
	}
	// The destination arm (see the header): computed here so
	// the evidence switch stays the one place claims are made.
	// The zone's head word keeps the object set's own
	// exclusions: comment, test, and getter sentences belong
	// to the intent layer and /contribute (the header's
	// standing note), so "add a comment to Describe" passes
	// this arm by exactly as it passes the object set.
	destObject := ""
	destClaim := false
	if (low[0] == "add" || low[0] == "put" || low[0] == "insert") && !hasThat {
		if tgt, ok2 := writeBoundaryTarget(fields, low); ok2 && len(strings.Fields(tgt)) == 1 {
			destObject = writeBoundaryObjectZone(fields, low)
			destClaim = destObject != ""
			if destClaim {
				head := strings.ToLower(strings.Fields(destObject)[0])
				if writeBoundaryOwnedHeads[head] {
					destClaim = false
				}
			}
		}
	}

	switch {
	case fileTgt || pathTgt:
		// A file was named: an edit ask whatever else it says.
	case destClaim:
		// An unservable add ask aimed at a bare destination:
		// claimed here, never at an analysis brain.
	case objAt >= 0 && !hasThat:
		// A code object with a destination or standing alone.
	case objAt >= 0 && hasDeclName:
		// A declaration being specified: claimed even with the
		// clause riding along. The clause stays unservable —
		// the answer names what is.
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

	// A line claimed BY the destination arm names its object
	// back in the generic answer: the zone as typed
	// ("templates", "header and footer templates"). A line
	// that claimed on file evidence keeps its standing
	// phrasing — the file-itself lines ("add a file to
	// x.go") reach the dest test too, and their answer's
	// object stays the generic "that" they have always used.
	if destClaim && objAt < 0 && !fileTgt && !pathTgt {
		req.object = destObject
	}

	// The file itself as the object: the token right after
	// the verb and any article is "file", with a target
	// riding along. ("file" is deliberately not a
	// writeObjectWord — these lines claim on their target
	// alone, and land in wbGeneric.)
	objIdx := 1
	if objIdx < len(low) && (low[objIdx] == "a" || low[objIdx] == "an" || low[objIdx] == "the") {
		objIdx++
	}
	req.fileItself = objIdx < len(low) && low[objIdx] == "file" && req.hasTarget

	// The object phrase as typed, with its article when it has
	// one ("an interface"); for the print line, both words.
	if objAt >= 0 {
		req.objWord = low[objAt]
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
	// print is the fallback for a print ask the concept book
	// doesn't carry (the concept claim, one step earlier in
	// the chain, serves the book's own print line); server is
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

	// A line claimed with a name slot and a "that …" clause
	// carries the name into its answer — whether the claim
	// rode the name slot alone (no target) or a file/path
	// target brought it here. (A named line with NO clause
	// keeps the standing answers, placeholders and all.) The
	// target case is the 2026-10-31 live miss's second half:
	// "add function named jake that returns a string to
	// jim/main.go" claimed on its target but answered with the
	// generic placeholder forms while the phrasing layer —
	// seated before this boundary — misrouted it entirely;
	// the scaffold sentence now spells the user's own target
	// through exampleFile, like every other targeted answer.
	if hasThat && hasDeclName {
		req.declName = declName
		req.returnsClause = thatAt+1 < len(low) && low[thatAt+1] == "returns"
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

// writeBoundaryObjectZone returns the object zone of an
// imperative line, as typed: the words between the verb (and
// one article after it) and the LAST destination preposition
// (to|in|into) — or the line's end when there is none — with a
// leading article trimmed. The zone must be a plain noun
// phrase: any pronoun or preposition standing inside it
// empties it. That is what keeps the header's own example
// unclaimed — "add it to the list of things to do" has the
// one-word destination "do" (the words after its last "to"),
// but its zone, "it to the list of things", is a pronoun and
// a prepositional chain, not a thing being added. Shared by
// the boundary's destination arm and the phrasing verdict's
// fallback answer (phrasingverdict.go), which names the zone
// back in the same voice.
func writeBoundaryObjectZone(fields, low []string) string {
	at := -1
	for i := 1; i < len(low); i++ {
		if low[i] == "to" || low[i] == "in" || low[i] == "into" {
			at = i
		}
	}
	end := len(fields)
	if at >= 0 {
		end = at
	}
	start := 1
	if start < end && (low[start] == "a" || low[start] == "an" || low[start] == "the") {
		start++
	}
	if start >= end {
		return ""
	}
	for i := start; i < end; i++ {
		switch low[i] {
		case "it", "this", "that", "them", "to", "in", "into", "on", "for", "with", "from", "at", "of", "by":
			return ""
		}
	}
	return strings.Join(fields[start:end], " ")
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
		// Three forms work, not two: the concept book's named
		// scaffold (served one claim earlier for add|put lines;
		// a line lands HERE only when the composer didn't claim
		// it — insert|write verbs, an unparseable slot zone),
		// the delegate, and the clone.
		f := r.exampleFile()
		if r.declName != "" {
			return r.namedFunctionAnswer(f)
		}
		return fmt.Sprintf("A function needs a shape I can assemble — I can't invent one from an idea alone. The forms that work: \"add a function named <Name> to %s\" (an empty scaffold you fill in), \"add a function in %s that calls <something that exists>\" (a new function wrapping a real one), or \"copy <existing function> as <new name>\" (a clone you then edit)", f, f)
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
		// The concept claim serves the book's print line before
		// this net runs; a line lands here only when the book in
		// play doesn't carry the entry (a hand-trimmed book).
		return fmt.Sprintf("I can't add a print line yet — a loose statement isn't one of the pieces I can assemble today. %s", r.closestAlternatives())
	default:
		obj := r.object
		if obj == "" {
			obj = "that"
		}
		if r.declName != "" && (r.objWord == "struct" || r.objWord == "interface") {
			// The book's named declaration rows: the working
			// sentence is the scaffold with the user's own
			// name — the clause is the only unservable part.
			return fmt.Sprintf("I can't add %s as asked — the \"that …\" clause isn't one of the pieces I can assemble. The form that works: \"add %s named %s to %s\" (an empty scaffold you fill in). %s", obj, obj, r.declName, r.exampleFile(), r.closestAlternatives())
		}
		if r.fileItself {
			// The ask was for the file itself: lead with the
			// one sentence that makes it, then the standing
			// alternatives for adding INTO that file.
			return fmt.Sprintf("I can't add %s yet — that's not one of the pieces I can assemble today. To make the file itself, say \"create a file %s\". %s", obj, r.target, r.closestAlternatives())
		}
		return fmt.Sprintf("I can't add %s yet — that's not one of the pieces I can assemble today. %s", obj, r.closestAlternatives())
	}
}

// namedFunctionAnswer answers a function line claimed on its
// name slot with a "that …" clause riding along: the clause is
// not spendable, so the answer says so plainly — a returns
// clause gets the returns truth (return types aren't assembled
// yet; the scaffold comes with no return and the body is the
// user's to fill in) — and spells the working forms with the
// user's own name in the scaffold sentence.
func (r writeBoundaryReq) namedFunctionAnswer(f string) string {
	forms := fmt.Sprintf("The forms that work: \"add a function named %s to %s\" (an empty scaffold you fill in), \"add a function in %s that calls <something that exists>\" (a new function wrapping a real one), or \"copy <existing function> as <new name>\" (a clone you then edit)", r.declName, f, f)
	if r.returnsClause {
		return "I can't add a function with a return type yet — return types aren't one of the pieces I can assemble, so the scaffold comes with no return and the body is yours to fill in. " + forms
	}
	return "I can't add a function with that extra clause — the \"that …\" part isn't one of the pieces I can assemble; I can only place the function itself. " + forms
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
	return answerWriteBoundary(req, "write-boundary answer: "+string(req.kind)+" (not performed)")
}

// runWriteBoundaryVerdict answers a line the phrasing
// recognizer judged a command (phrasingverdict.go) with the
// boundary's generic answer: an add-family ask no shape can
// serve, named back from its object zone — the same words the
// boundary itself would give the line, reached by the
// recognizer's verdict instead of the boundary's own shape
// test, and recorded as such.
func runWriteBoundaryVerdict(line string) bool {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(strings.Trim(f, ",;:\"'()"))
	}
	req := writeBoundaryReq{kind: wbGeneric}
	req.target, req.hasTarget = writeBoundaryTarget(fields, low)
	req.fileTgt = req.hasTarget && strings.HasSuffix(strings.ToLower(req.target), ".go")
	req.object = writeBoundaryObjectZone(fields, low)
	return answerWriteBoundary(req, "write-boundary answer: phrasing-verdict command (not performed)")
}

// answerWriteBoundary prints one boundary answer and records
// it — the one tail both boundary routes share.
func answerWriteBoundary(req writeBoundaryReq, note string) bool {
	fmt.Printf("%s %s.\n", cTag("gollemer [composer]>"), strings.TrimSuffix(req.answer(), "."))
	recordBrain("composer", true, note)
	return true
}
