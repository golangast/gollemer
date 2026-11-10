package chat

// "Have the goroutine return a string" — the goroutine-result
// piece, claimed deterministically in the unified loop beside
// the other write-side families, ahead of the phrasing layer.
//
// Why this claim exists: John's live failure, 2026-11-09. In
// jim/main.go he typed "can you update the goroutine to
// return a string in jim/main.go" and the phrasing layer read
// it as "add a string to jim/main.go" — a stray string
// statement landed where a goroutine edit was asked for. The
// ask is a real one, and it has a real answer: a goroutine
// cannot return a value in Go (a go statement throws results
// away), so the piece assembles the language's own idiom —
// a channel declared before the go statement, a send at the
// end of the goroutine's body, the receive right after
// (analyze's EditGoroutineResult planner). His words now
// claim this piece, in this grammar, and never reach the
// phrasing layer at all.
//
// Grammar (exactly this; trailing "." or "?" optional, one
// leading "can you"/"could you"/"please" allowed):
//
//	have|make [the] [ordinal] goroutine [in <Func>] [in <file>] return [a|an] <type>
//	update [the] [ordinal] goroutine [in <Func>] [in <file>] to return [a|an] <type>
//	return [a|an] <type> from [the] [ordinal] goroutine [in <Func>] [in <file>]
//	ordinal: first|second|third|fourth|fifth
//	<file>: one token ending in .go; <Func>: one identifier
//
// The two "in" clauses may come in either order and either
// may be absent: a clause whose token ends in .go is the
// file, any other is the function. No file named reads as
// the session's last worked-on file (printed, per the
// session-context rule), else the project's main file by
// the folder rule. No function named scopes the whole file.
//
// The type slot is one type-shaped token. A type outside
// the planner's closed set (string, int, bool, float64)
// still CLAIMS — and refuses in the piece's own words —
// because the sentence is unmistakably this piece's shape;
// only a shape outside the grammar falls through.
//
// Non-claims are pinned as carefully as claims: "have the
// goroutine print a string" (a different ask — printing is
// not returning), "add a goroutine …" (the concept book's
// own sentence), statements about goroutines ("the
// goroutine returns a string"), and every other family's
// lines keep their claims or their falls.

import (
	"fmt"
	"go/token"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// goroutineResultRequest is one parsed goroutine-result ask.
type goroutineResultRequest struct {
	typ     string // the result type as typed
	fn      string // the function as typed ("" = the whole file)
	file    string // the file as typed ("" = session/project default)
	ordinal int    // 1-based position named in the sentence (0 = none)
}

// goroutineOrdinalWords are the positions the grammar
// names. The planner counts the same positions.
var goroutineOrdinalWords = map[string]int{
	"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}

// goroutineOrdinalName renders a position for the
// clarifying question's canonical completion.
func goroutineOrdinalName(n int) string {
	for w, v := range goroutineOrdinalWords {
		if v == n {
			return w
		}
	}
	return ""
}

// parseGoroutineResult reads one chat line as a
// goroutine-result ask. The second return is false for
// anything outside the grammar (see the file header).
func parseGoroutineResult(line string) (goroutineResultRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 4 {
		return goroutineResultRequest{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	i := 0
	// One leading politeness shape.
	if i+1 < len(fields) && (low[i] == "can" || low[i] == "could") && low[i+1] == "you" {
		i += 2
	} else if low[i] == "please" {
		i++
	}
	if i >= len(fields) {
		return goroutineResultRequest{}, false
	}
	var req goroutineResultRequest
	var ok bool
	switch low[i] {
	case "have", "make":
		req, i, ok = goroutineHeadAt(fields, low, i+1)
		if !ok {
			return goroutineResultRequest{}, false
		}
		if i >= len(fields) || low[i] != "return" {
			return goroutineResultRequest{}, false
		}
		req.typ, i, ok = goroutineTypeAt(fields, low, i+1)
		if !ok {
			return goroutineResultRequest{}, false
		}
	case "update":
		req, i, ok = goroutineHeadAt(fields, low, i+1)
		if !ok {
			return goroutineResultRequest{}, false
		}
		if i+1 >= len(fields) || low[i] != "to" || low[i+1] != "return" {
			return goroutineResultRequest{}, false
		}
		req.typ, i, ok = goroutineTypeAt(fields, low, i+2)
		if !ok {
			return goroutineResultRequest{}, false
		}
	case "return":
		req.typ, i, ok = goroutineTypeAt(fields, low, i+1)
		if !ok {
			return goroutineResultRequest{}, false
		}
		if i >= len(fields) || low[i] != "from" {
			return goroutineResultRequest{}, false
		}
		// The head parser expects the goroutine word
		// next; hand it the position after "from".
		req2, ni, ok2 := goroutineHeadAt(fields, low, i+1)
		if !ok2 {
			return goroutineResultRequest{}, false
		}
		req.fn, req.file, req.ordinal = req2.fn, req2.file, req2.ordinal
		i = ni
	default:
		return goroutineResultRequest{}, false
	}
	// The tail: only "in" clauses may follow (the head
	// already consumed the ones before the type).
	req, i, ok = goroutineInClauses(fields, low, i, req)
	if !ok || i != len(fields) {
		return goroutineResultRequest{}, false
	}
	if req.typ == "" {
		return goroutineResultRequest{}, false
	}
	return req, true
}

// goroutineHeadAt reads "[the] [ordinal] goroutine" plus
// any "in" clauses that follow it (the pre-type clauses).
// It returns the request slots the clauses filled and the
// position of the next word.
func goroutineHeadAt(fields, low []string, i int) (goroutineResultRequest, int, bool) {
	var req goroutineResultRequest
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i < len(fields) {
		if ord, ok := goroutineOrdinalWords[low[i]]; ok {
			req.ordinal = ord
			i++
		}
	}
	// "goroutine", or the book's two-word spelling.
	switch {
	case i < len(fields) && low[i] == "goroutine":
		i++
	case i+1 < len(fields) && low[i] == "go" && low[i+1] == "routine":
		i += 2
	default:
		return req, i, false
	}
	req, i, ok := goroutineInClauses(fields, low, i, req)
	return req, i, ok
}

// goroutineInClauses consumes a run of "in <token>"
// clauses: a token ending in .go fills the file slot, any
// other identifier fills the function slot. A clause of a
// kind already filled, or a token of neither shape, ends
// the claim.
func goroutineInClauses(fields, low []string, i int, req goroutineResultRequest) (goroutineResultRequest, int, bool) {
	for i+1 < len(fields) && low[i] == "in" {
		tok, tokLow := fields[i+1], low[i+1]
		switch {
		case strings.HasSuffix(tokLow, ".go"):
			if req.file != "" {
				return req, i, false
			}
			req.file = tok
		case token.IsIdentifier(tok):
			if req.fn != "" {
				return req, i, false
			}
			req.fn = tok
		default:
			return req, i, false
		}
		i += 2
	}
	// A dangling "in" (nothing after it) is a started
	// clause that named nothing: outside the grammar.
	if i < len(fields) && low[i] == "in" {
		return req, i, false
	}
	return req, i, true
}

// goroutineTypeAt reads the type slot: one optional
// article ("a", "an", or "the" — "return the string" is
// the same ask as "return a string"), then one
// type-shaped token (a letter or underscore, then
// letters, digits, underscores, dots). The word is
// returned as typed; set membership is the flow's check,
// so an unknown type can refuse in the piece's own words
// instead of falling through.
func goroutineTypeAt(fields, low []string, i int) (string, int, bool) {
	if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
		i++
	}
	if i >= len(fields) {
		return "", i, false
	}
	tok := fields[i]
	for j, r := range tok {
		switch {
		case r == '_' || ('a' <= r && r <= 'z') || ('A' <= r && r <= 'Z'):
		case j > 0 && (('0' <= r && r <= '9') || r == '.'):
		default:
			return "", i, false
		}
	}
	return tok, i + 1, true
}

// tryGoroutineResult is the chat-loop entry: it claims
// goroutine-result sentences and runs the flow on the
// chat's line reader.
func tryGoroutineResult(line string, r *LineReader) bool {
	return runGoroutineResult(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runGoroutineResult carries one line from sentence to
// resolved file to the shared edit spine. It returns true
// when the line belonged to this claim at all.
func runGoroutineResult(line string, ask editAsk) bool {
	q, ok := parseGoroutineResult(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	typ := strings.ToLower(q.typ)
	if _, known := analyze.GoroutineResultZeroValue(typ); !known {
		fmt.Printf("%s Not doing that: %q isn't a result type I can build this with yet — %s are the ones this piece knows. (A channel carrying one of your own types is a bigger piece.)\n", tag, q.typ, analyze.GoroutineResultTypes())
		recordBrain("composer", false, "goroutine-result refused: type")
		return true
	}
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "goroutine-result refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "goroutine-result refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "goroutine-result refused: root missing")
		return true
	}
	// The file the goroutine lives in: the sentence's, the
	// session's last worked-on file (printed when used —
	// the session-context rule), or the project's main file
	// by the folder rule.
	rel, refusal := resolveGoroutineFile(rootAbs, q, tag)
	if refusal != "" {
		fmt.Printf("%s Not doing that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "goroutine-result refused: file")
		return true
	}
	// The spine plans against the analyzed project; make
	// the target's root the current one, quietly (the
	// neighbours' reasoning). A project with no Go packages
	// yet cannot be analyzed — and needs no analysis for
	// the planner to answer honestly (it finds no goroutine
	// and says so): an empty index rooted there is enough.
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		p, err := analyze.Analyze(rootAbs)
		switch {
		case err == nil:
			setCurrentProject(rootRaw, p)
		case strings.Contains(err.Error(), "no Go packages found"):
			setCurrentProject(rootRaw, &analyze.Project{Root: rootAbs})
		default:
			fmt.Printf("I couldn't analyze %s: %v\n", rootAbs, err)
			recordBrain("composer", false, "goroutine-result refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "goroutine-result refused: stale project unreadable")
		return true
	}
	// Several goroutines and the sentence pinned none of
	// them: a question, not a dead end (clarify.go). The
	// answer — a number, or the candidate's words —
	// completes the original ask as its canonical sentence,
	// through the same plan, gate, and build check. Every
	// other state (none, one, a shape the piece declines)
	// keeps the planner's own answers via the spine below.
	if q.ordinal == 0 {
		if sites := lastAnalyzeProject.GoroutineResultSites(rel, q.fn); len(sites) >= 2 {
			cands := make([]clarifyCandidate, len(sites))
			for i, s := range sites {
				ord := goroutineOrdinalName(i + 1)
				cands[i] = clarifyCandidate{
					Label: fmt.Sprintf("the %s goroutine in %s", ord, s.Func),
					Desc:  fmt.Sprintf("%s:%d", rel, s.Line),
					Words: []string{ord},
				}
			}
			fn, file := q.fn, q.file
			askPickQuestion(tag,
				"There's more than one goroutine — which one should get the result channel? Nothing's been written.",
				"Answer with the number or the position word",
				cands,
				func(idx int, ask2 editAsk) bool {
					var b strings.Builder
					b.WriteString("have the ")
					b.WriteString(goroutineOrdinalName(idx + 1))
					b.WriteString(" goroutine")
					if fn != "" {
						b.WriteString(" in " + fn)
					}
					b.WriteString(" return a " + typ)
					if file != "" {
						b.WriteString(" in " + file)
					}
					return runGoroutineResult(b.String(), ask2)
				})
			recordBrain("composer", false, "goroutine-result question: which goroutine")
			return true
		}
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:             analyze.EditGoroutineResult,
		File:             rel,
		FuncName:         q.fn,
		GoroutineType:    typ,
		GoroutineOrdinal: q.ordinal,
	}, ask)
}

// resolveGoroutineFile settles the ask's file inside
// rootAbs, or returns an honest refusal. A named file must
// exist and be a .go file — this piece edits a goroutine
// that is already there; it never creates the file.
func resolveGoroutineFile(rootAbs string, q goroutineResultRequest, tag string) (string, string) {
	if q.file != "" {
		abs, rel, refusal := fenceResolve(rootAbs, q.file)
		if refusal != "" {
			return "", refusal
		}
		st, err := os.Stat(abs)
		switch {
		case err != nil:
			return "", fmt.Sprintf("there's no file %s in this project — the goroutine has to already exist in one of your files", q.file)
		case st.IsDir():
			return "", fmt.Sprintf("%s is a folder — name the .go file in it that holds the goroutine", q.file)
		case !strings.HasSuffix(strings.ToLower(q.file), ".go"):
			return "", fmt.Sprintf("%s isn't a Go file — a goroutine lives in a .go file", q.file)
		}
		return rel, ""
	}
	if workContextLive(rootAbs) && workCtx.lastFile != "" {
		fmt.Printf("%s No file named — reading that as %s, the file we worked on last in this session.\n", tag, workCtx.lastFile)
		return workCtx.lastFile, ""
	}
	if fi, err := os.Stat(filepath.Join(rootAbs, "main.go")); err == nil && !fi.IsDir() {
		return "main.go", ""
	}
	cands := packageMainFiles(rootAbs)
	switch len(cands) {
	case 1:
		return cands[0], ""
	case 0:
		return "", "I don't know which file's goroutine you mean — this project has no main program at its root to read it as. Name the file, like \"have the goroutine return a string in jim/main.go\""
	default:
		return "", fmt.Sprintf("I don't know which file's goroutine you mean — the project root has more than one main-package file (%s). Name the file, like \"have the goroutine return a string in %s\"", strings.Join(cands, ", "), cands[0])
	}
}
