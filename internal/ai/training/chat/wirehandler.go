package chat

// "Connect the handler to the webserver" — the wiring recipe,
// claimed deterministically in the unified loop beside
// add-a-webserver and serve-an-html-file, whose precedent it
// follows. It exists for the live dead end of 2026-11-03: a
// form handler landed in jim/main.go (composer confirmed,
// build-checked), and the natural next sentence — "connect
// the handler to the webserver" — fell through every claim to
// the intent layer, because placement existed and wiring did
// not. Now the deterministic side owns it: the analyzer finds
// the handler and the serving function in the project index
// and assembles ONE registration line behind the shared spine
// (analyze.EditWireHandler) — plan, [y/n] gate, transaction,
// whole-project build, verify, restore.
//
// Grammar (exactly this):
//
//	connect|wire|register|attach|link|plug [up] [the|a|an] <handler ref> [up]
//	    to|with|into [the|a|an] webserver|web server|server [at <path>]
//	hook up [the|a|an] <handler ref> … | hook [the|a|an] <handler ref> up …
//	handler ref: handler | handler <Name> | <Name> | <Name> handler
//
// ("hook" is a phrasal verb here — it claims only with its
// "up", in front of the reference or split behind it; a bare
// "hook the handler to…" is a pinned non-claim.)
//
// The verb family (link, plug), the "into" preposition, and
// the split phrasal ("wire Jake up to the webserver", "hook
// Jake up…") were widened 2026-11-05: the phrasal verb's
// particle rides behind the handler's name as naturally as
// in front of it, and the grammar read it only in front.
// "point" was probed and deliberately NOT added — "at" is
// this grammar's route slot, so "point Jake at the webserver"
// cannot be read without breaking "…at /path".
//
// The claim requires BOTH a handler reference and a server
// word, so it can never steal the neighbours' sentences: the
// database family's connects carry no handler word ("add a
// database connection…", "connect to the database"), add-a-
// webserver starts with its own verb, and serve-an-html-file
// with "serve". The direction is fixed — handler INTO server;
// "connect the webserver to the handler" is not this grammar.
// A sentence that starts this grammar and stops early (names
// the handler but trails off before the webserver) is still
// claimed and answered in wiring terms, the add-server
// ruling: it must never fall through to the intent layer.
//
// Coverage is the deterministic grammar only: messy phrasings
// ("could you wire Jake up for me?") belong to the phrasing
// layer's world and are NOT claimed here — the same accepted
// scope as the neighbouring recipes' first builds.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// wireHandlerRequest is one parsed wiring ask.
type wireHandlerRequest struct {
	name    string // the handler's name as typed ("" = the bare "the handler")
	route   string // the path as typed ("" = the default: "/" + lowercased name)
	partial bool   // the grammar started but stopped before the webserver was named
}

// wireNameStopwords are the words that can never be a handler
// name in the name slot: the grammar's own furniture. A line
// like "connect to the database" dies here — "to" is not a
// name, and no handler reference was made.
var wireNameStopwords = map[string]bool{
	"the": true, "a": true, "an": true, "to": true, "with": true,
	"at": true, "in": true, "into": true, "from": true, "up": true,
	"handler": true, "webserver": true, "server": true, "web": true,
}

// isWireNameToken reports whether a token can stand as a
// handler name: identifier-shaped and not the grammar's own
// furniture.
func isWireNameToken(tok, low string) bool {
	if tok == "" || wireNameStopwords[low] {
		return false
	}
	for i, r := range tok {
		switch {
		case r == '_' || ('a' <= r && r <= 'z') || ('A' <= r && r <= 'Z'):
		case '0' <= r && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

// parseWireHandler reads one chat line as a wiring ask. The
// second return is false for anything outside the grammar —
// including every neighbour's sentence (see the file header).
// A line that matches the grammar's opening and then stops
// (it named the handler and trailed off before the webserver)
// returns partial=true: claimed, so the flow can answer in
// wiring terms.
func parseWireHandler(line string) (wireHandlerRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return wireHandlerRequest{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	i := 0
	switch low[0] {
	case "connect", "wire", "register", "attach", "link", "plug", "hook":
		i = 1
	default:
		return wireHandlerRequest{}, false
	}
	// "wire up the handler…" — the phrasal particle.
	upSeen := false
	if i < len(fields) && low[i] == "up" {
		upSeen = true
		i++
	}
	if i < len(fields) && (low[i] == "the" || low[i] == "a" || low[i] == "an") {
		i++
	}
	// The handler reference.
	req := wireHandlerRequest{}
	switch {
	case i < len(fields) && low[i] == "handler":
		i++
		// "the handler Jake" — the name after the kind word.
		if i < len(fields) && isWireNameToken(fields[i], low[i]) {
			req.name = fields[i]
			i++
		}
	case i < len(fields) && isWireNameToken(fields[i], low[i]):
		req.name = fields[i]
		i++
		if i < len(fields) && low[i] == "handler" {
			i++
		}
	default:
		return wireHandlerRequest{}, false
	}
	// The phrasal "up" can also ride behind the handler
	// reference: "wire Jake up to the webserver", "hook Jake
	// up to the webserver". (A handler can't be named "Up" —
	// the word is grammar furniture, in wireNameStopwords.)
	if i < len(fields) && low[i] == "up" {
		upSeen = true
		i++
	}
	// "hook" is this grammar's phrasal verb: it claims only
	// WITH its particle — in front ("hook up Jake…") or split
	// behind the reference ("hook Jake up…"). A bare "hook
	// the handler to…" was pinned a non-claim when the family
	// shipped and stays one.
	if low[0] == "hook" && !upSeen {
		return wireHandlerRequest{}, false
	}
	// The line named the handler and stopped ("connect the
	// handler", "wire Jake"): claimed, answered in wiring
	// terms — a sentence about wiring a handler must never
	// reach the intent layer.
	if i >= len(fields) {
		req.partial = true
		return req, true
	}
	if low[i] != "to" && low[i] != "with" && low[i] != "into" {
		return wireHandlerRequest{}, false
	}
	i++
	if i >= len(fields) {
		req.partial = true
		return req, true
	}
	if low[i] == "the" || low[i] == "a" || low[i] == "an" {
		i++
	}
	if i >= len(fields) {
		req.partial = true
		return req, true
	}
	// The server word. Anything else here ("the database") is
	// another family's sentence, unclaimed.
	switch {
	case low[i] == "webserver" || low[i] == "server":
		i++
	case low[i] == "web" && i+1 < len(fields) && low[i+1] == "server":
		i += 2
	default:
		return wireHandlerRequest{}, false
	}
	if i == len(fields) {
		return req, true
	}
	// The optional route: "at <path>", one token, then the
	// end of the line.
	if low[i] == "at" && i+2 == len(fields) {
		req.route = fields[i+1]
		return req, true
	}
	// A longer tail is another ask glued behind this one;
	// multi-piece lines are not this claim's.
	return wireHandlerRequest{}, false
}

// tryWireHandler is the chat-loop entry: it claims
// wire-a-handler sentences and runs the flow on the chat's
// line reader.
func tryWireHandler(line string, r *LineReader) bool {
	return runWireHandler(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runWireHandler carries one line from sentence to the shared
// edit spine. It returns true when the line belonged to this
// claim at all.
func runWireHandler(line string, ask editAsk) bool {
	q, ok := parseWireHandler(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if q.partial {
		// The grammar matched and named the handler but the
		// sentence stopped before the webserver: claimed, and
		// answered in wiring terms.
		fmt.Printf("%s Not wiring that yet: the sentence names the handler but stops before the webserver — say it like \"connect the Jake handler to the webserver\" and I'll show you the one-line plan.\n", tag)
		recordBrain("composer", false, "wire-handler refused: sentence unfinished")
		return true
	}
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "wire-handler refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "wire-handler refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "wire-handler refused: root missing")
		return true
	}
	// The spine plans against the analyzed project; make the
	// target's root the current one, quietly (the neighbours'
	// reasoning). A project with no Go packages yet cannot be
	// analyzed — and needs no analysis for the planner to
	// answer honestly (it finds no handler and no server, and
	// says so): an empty index rooted there is enough.
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		p, err := analyze.Analyze(rootAbs)
		switch {
		case err == nil:
			setCurrentProject(rootRaw, p)
		case strings.Contains(err.Error(), "no Go packages found"):
			setCurrentProject(rootRaw, &analyze.Project{Root: rootAbs})
		default:
			fmt.Printf("I couldn't analyze %s: %v\n", rootAbs, err)
			recordBrain("composer", false, "wire-handler refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "wire-handler refused: stale project unreadable")
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:      analyze.EditWireHandler,
		WireName:  q.name,
		WireRoute: q.route,
	}, ask)
}
