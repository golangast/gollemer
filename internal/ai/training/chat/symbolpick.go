package chat

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// symbolpick.go — the two deterministic symbol-reference paths that
// run inside the intent layer's front door, before the classifier:
//
//   - A pending function pick: when a codebase answer was a numbered
//     disambiguation list, the user's reply ("2") resolves against
//     the very candidates that were printed.
//   - An exact dotted symbol reference: the name the pick-list (or
//     the code) prints — "codec.decoderBincBytes.mapEnd" — pasted
//     back as a bare line, or named in a "what does … do" question.
//
// Before these existed, the classifier claimed the bare dotted line
// as a find search: slotFindConcept kept the (lowercased) dotted
// token whole as the search "concept", and the find matcher — which
// substring-matches names and docs — could never match it, so the
// system answered `No matches for "codec.decoderbincbytes.mapend"`
// to the exact string it had itself suggested. A dotted reference is
// unambiguous code, never small talk or a search phrase, so it is
// resolved deterministically here, the same reason beginner commands
// and assembled edits are claimed before the classifier.

// pendingFuncPick is the one outstanding disambiguation list: the
// candidates, in printed order. Package state, like the pending
// clarification's — exactly one at a time, and any later line that
// reaches the intent layer either answers it or clears it.
type pendingFuncPick struct {
	cands []*analyze.Func
}

var pendingPick *pendingFuncPick

// setFuncPick records (or clears) the outstanding pick-list. Called
// wherever a codebase Q&A answer is produced: a fresh list replaces
// any older one, and a non-list answer clears it.
func setFuncPick(cands []*analyze.Func) {
	if len(cands) == 0 {
		pendingPick = nil
		return
	}
	pendingPick = &pendingFuncPick{cands: cands}
}

// resolveFuncPick gives a pending pick-list first claim on a numeric
// reply. done=false means the line is no answer (any pending list is
// forgotten) and must be processed normally — the same contract as
// resolveClarification.
func resolveFuncPick(line string) (done, handled bool) {
	pk := pendingPick
	if pk == nil {
		return false, false
	}
	n, err := strconv.Atoi(strings.TrimSpace(line))
	if err != nil || n < 1 || n > len(pk.cands) {
		pendingPick = nil
		return false, false
	}
	pendingPick = nil
	fn := pk.cands[n-1]
	// A pick recorded against a project that has since been replaced
	// must not answer from the old index.
	if lastAnalyzeProject == nil || lastAnalyzeProject.FuncByID(fn.ID) != fn {
		return false, false
	}
	fmt.Println(fn.Detail(lastAnalyzeProject))
	recordBrain("analyst", true, "func pick "+fn.Display())
	friendNoteFunc(fn.Display())
	return true, true
}

// exactDottedFunc resolves a dotted function reference exactly the
// way the pick-list prints it: "pkg.Receiver.Name", "Receiver.Name",
// or "pkg.Name", case-insensitively (the user may retype it in any
// case). The query must be dotted — bare names keep their existing
// forgiving paths (findFuncByQuery, Project.resolve), where a match
// is a guess among namesakes, not an exact reference.
func exactDottedFunc(p *analyze.Project, tok string) *analyze.Func {
	if p == nil || !strings.Contains(tok, ".") || strings.Contains(tok, "/") {
		return nil
	}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			if strings.EqualFold(tok, fn.Display()) {
				return fn
			}
			if fn.Receiver != "" &&
				strings.EqualFold(tok, strings.Trim(fn.Receiver, "*")+"."+fn.Name) {
				return fn
			}
			if strings.EqualFold(tok, fn.Pkg+"."+fn.Name) {
				return fn
			}
		}
	}
	return nil
}

// tryDottedSymbol claims a line that references one project function
// by its exact dotted name — bare, or inside a question the Q&A layer
// knows ("what does codec.decoderBincBytes.mapEnd do"). It runs
// before classification so the classifier never has to guess at a
// reference that is already exact. Returns false (the line falls
// through untouched) when no dotted token in the line resolves.
func tryDottedSymbol(line string, profile *UserProfile) bool {
	p := lastAnalyzeProject
	if p == nil {
		return false
	}
	trimmed := strings.TrimSpace(line)
	if trimmed == "" || strings.HasPrefix(trimmed, "/") {
		return false
	}
	// Bare line: the whole line is the reference.
	if !strings.ContainsAny(trimmed, " \t") {
		tok := strings.Trim(trimmed, `"'(),;:`)
		fn := exactDottedFunc(p, tok)
		if fn == nil {
			return false
		}
		fmt.Println(fn.Detail(p))
		recordBrain("analyst", true, "exact symbol "+fn.Display())
		friendNoteFunc(fn.Display())
		friendObserveIntent("whatis")
		recordIntentOutcome(line, "whatis", 1, IntentOutcomeDispatched)
		return true
	}
	// Question shapes: a dotted token that resolves exactly earns
	// the line a trip through the Q&A layer, which parses the shape
	// itself. If the shape isn't one it knows, the line falls
	// through to the classifier as before.
	for _, f := range strings.Fields(trimmed) {
		tok := strings.Trim(f, `"'(),;:`)
		tok = strings.TrimSuffix(tok, "?")
		if exactDottedFunc(p, tok) == nil {
			continue
		}
		out, cands, ok := p.AnswerCands(trimmed)
		if !ok {
			return false
		}
		fmt.Println(out)
		setFuncPick(cands)
		recordBrain("analyst", true, "exact symbol question")
		friendNoteFunc(tok)
		friendObserveIntent("whatis")
		recordIntentOutcome(line, "whatis", 1, IntentOutcomeDispatched)
		return true
	}
	return false
}
