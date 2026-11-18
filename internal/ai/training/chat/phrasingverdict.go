package chat

// The phrasing verdict: the recognizer consulted FIRST on
// lines every deterministic claim declined — the second half
// of the 2026-11-10 dispatch fix. The phrasing layer
// (phrasing2.go) already sits late in the unified loop, but a
// line whose family is confidently recognized while its
// slots fail extraction or validation declines there
// SILENTLY — and the line then travels on to the analysis
// brains, which answer an unserved command as if it were a
// question about code. The recorded thefts: "add templates
// in jim" answered by the Uses brain, "add header and footer
// templates to jim" by the Comparer — both imperative add
// asks no composer shape owns. The universal boundary
// (writeboundary.go's destination arm) closes the
// shape-recognizable half of that hole; this claim closes
// the phrasing half: when every deterministic claim has
// declined and the boundary's own shape test declined too,
// the recognizer's verdict is consulted BEFORE the intent
// layer, and a line it judges a command gets the boundary's
// answer — an honest "not one of the pieces I can assemble
// today", never an analysis answer.
//
// The verdict is deliberately NOT the recognizer's raw
// output. Measured against the shipped model (2026-11-10),
// the gated classifier alone is far too hot to seat here:
// "my dog is bella" gates print at 0.99, "show me Add" gates
// concept-time-now at 0.80, "walk me through Add" gates
// concept-walk-directory at 0.95, "the server returns an
// html file" (a statement) gates serve-html at 0.99. What
// makes a verdict trustworthy is the conjunction, mirroring
// how the route itself reads a line:
//
//   - the line segments like a command (segmentLine, the
//     route's own segmentation — wrappers stripped, so
//     "could you stick a for loop in main for me?" reads as
//     the ask it is);
//   - NO fragment opens with a question word — the route's
//     own question veto (phrasingRouteB's set), so a question
//     always proceeds to analysis untouched;
//   - the FIRST fragment — the line's own ask; a compound's
//     later fragments are separate asks the route handles
//     piece by piece, and a bare trailing fragment is never
//     a command by itself ("compare Add and Get" segments
//     to ["compare Add", "Get"], and the bare "Get" fragment
//     gates a family on its own — reading fragments past
//     the first would steal the Comparer line this claim
//     exists to protect) — OPENS with one of the write
//     families' own lead verbs (the extractor's verb
//     vocabulary: add, put, insert, place, stick, make,
//     create, set, get, register, record, have, serve, copy,
//     clone, write) and carries at least one more word — an
//     imperative lead, which no measured false positive
//     carries ("build" is deliberately NOT in the set: the
//     intent layer owns build-led lines — "build an api
//     with /users and /health" is its build_api shape, and
//     the recognizer gates a concept family for it);
//   - and the gated classifier names a write family for that
//     fragment — never "none", never ungated.
//
// A gated "none", a failed gate, a question fragment, or a
// non-imperative lead all decline the verdict, and the line
// falls through exactly as before: the boundary's rule,
// then analysis. The concept and social seats are untouched;
// the model file is not retrained or edited by this piece.

// phrasingLeadVerbs are the write families' own lead verbs —
// the words an imperative command fragment can open with,
// taken from the extractor's verb vocabulary (its per-family
// switches and the gerund table). "build" is excluded on
// purpose: the intent layer's build_api owns build-led
// lines, and the recognizer gates concept families for them.
var phrasingLeadVerbs = map[string]bool{
	"add": true, "put": true, "insert": true, "place": true,
	"stick": true, "make": true, "create": true, "set": true,
	"get": true, "register": true, "record": true, "have": true,
	"serve": true, "serves": true, "serving": true,
	"copy": true, "clone": true, "write": true,
}

// phrasingQuestionHeads is the route's own question veto set
// (phrasingRouteB): a fragment opening with one of these is
// a question, never a command.
var phrasingQuestionHeads = map[string]bool{
	"how": true, "what": true, "why": true, "when": true,
	"where": true, "who": true, "is": true, "are": true,
	"was": true, "were": true, "do": true, "does": true,
	"did": true,
}

// phrasingCommandVerdict reports whether the recognizer
// judges a line a command: segmented, question-free,
// imperative-led, and gated to a write family. See the file
// header for why all four conditions are required.
func phrasingCommandVerdict(line string) bool {
	frags, _, ok := segmentLine(line)
	if !ok {
		frags = []frag{newFrag(PhrasingStripWrapper(line))}
	}
	for _, f := range frags {
		if len(f.low) > 0 && phrasingQuestionHeads[f.low[0]] {
			return false
		}
	}
	if len(frags) == 0 {
		return false
	}
	f := frags[0]
	if len(f.low) < 2 || !phrasingLeadVerbs[f.low[0]] {
		return false
	}
	family, gated := phrasingClassifier(f.text())
	return gated && family != "" && family != "none"
}

// tryPhrasingVerdict is the chat-loop entry, seated in the
// unified loop directly before the intent dispatch: every
// deterministic claim — the boundary among them — has
// declined by then, and the intent layer is the first
// analysis brain.
func tryPhrasingVerdict(line string) bool {
	return runPhrasingVerdict(line)
}

// runPhrasingVerdict answers a verdict-command line with the
// write boundary's generic answer. A declined verdict
// declines whole: the line continues to analysis as today.
func runPhrasingVerdict(line string) bool {
	if !phrasingCommandVerdict(line) {
		return false
	}
	return runWriteBoundaryVerdict(line)
}
