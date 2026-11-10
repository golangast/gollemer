package chat

// The wasm events piece (John 2026-11-09, "have something
// that you can say what events you want to happen" — the
// follow-on to the wasm piece, whose components rendered
// and mounted but answered nothing):
//
//	when the <component> is clicked, show|hide|toggle the <component>
//	when the <component> is clicked, set the <component>'s words to '<words>'
//	when the <switch> is flipped, …
//	when the <text field> changes, …
//	    [in <folder>]
//
// A component reference is [the] [name] <component words>
// — "the button", "the save button" — or a bare name
// ("when save is clicked"). The events are a closed set:
// a click (buttons and cards answer it), a change (the
// switch, the text field). The actions are a closed set:
// show, hide, toggle, and set a button's or card's words.
// The folder names the wasm app's folder (or its main.go);
// with none named, the session's last-touched wasm app is
// the reading, printed as everywhere a context target
// resolves; with neither, the answer is guidance — never
// a guess. Several mounted components fitting a reference
// ask ONE clarifying question, whose answer completes the
// original sentence through the same gate as if typed in
// full (clarify.go's discipline — facts re-read at plan
// time by analyze.planWasmEvent).
//
// Pages stay script-free: an event sentence aimed at a
// folder of plain pages gets that boundary in words, not
// an edit. The claim seats directly after the component
// claim in the unified loop: its sentences open with
// "when", a word no neighbouring grammar takes, and the
// component adds it must not steal keep their shapes
// byte-identical because this parse requires the whole
// event shape — verb, comma, action — before it claims.

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// wasmEventRef is one parsed component reference: the
// registry kind ("" when the sentence named the component
// by name alone), the "called" name ("" when none), and —
// when the words are neither a kind nor a kind with a
// name in front — the words as typed, in unknown, for the
// flow's no-such-component answer.
type wasmEventRef struct {
	kind    string
	name    string
	unknown string
}

// wasmEventAsk is one parsed event sentence.
type wasmEventAsk struct {
	source    wasmEventRef
	event     string // "click" | "change"
	action    string // "show" | "hide" | "toggle" | "setwords"
	target    wasmEventRef
	words     string // setwords only
	folder    string
	hasFolder bool
}

// canonical renders the ask as its full sentence — the
// clarifying completion's re-entry and the record of the
// shape, in the grammar's own words.
func (q wasmEventAsk) canonical() string {
	req := analyze.EditRequest{
		EventSourceKind: q.source.kind, EventSourceName: q.source.name,
		EventName: q.event, EventAction: q.action,
		EventTargetKind: q.target.kind, EventTargetName: q.target.name,
		EventWords: q.words,
	}
	s := analyze.WasmEventSentence(req)
	if q.hasFolder {
		s += " in " + q.folder
	}
	return s
}

// wasmComponentNameToken reports whether a word can be a
// component's name: one lowercase identifier-shaped word
// that isn't one of the grammar's own words.
func wasmComponentNameToken(w string) bool {
	if w == "" {
		return false
	}
	switch w {
	case "the", "a", "an", "when", "is", "are", "show", "hide",
		"toggle", "set", "words", "to", "in", "into", "at", "on",
		"called", "clicked", "pressed", "changes", "changed",
		"flipped", "toggled", "switched":
		return false
	}
	for i, r := range w {
		switch {
		case 'a' <= r && r <= 'z':
		case '0' <= r && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

// parseWasmEventRefWords reads a reference's words ([the]
// [name] <component words>, or a bare name) against the
// component registry. The whole phrase is tried as a kind
// first — "text field" is a kind, never a field named
// text — then as a name in front of a kind, then as a
// bare name.
func parseWasmEventRefWords(words []string) (wasmEventRef, bool) {
	if len(words) > 0 && words[0] == "the" {
		words = words[1:]
	}
	if len(words) == 0 {
		return wasmEventRef{}, false
	}
	joined := strings.Join(words, " ")
	if comp, ok := analyze.MaterialComponentByForm(joined); ok {
		return wasmEventRef{kind: comp.ID}, true
	}
	if len(words) == 1 {
		if wasmComponentNameToken(words[0]) {
			return wasmEventRef{name: words[0]}, true
		}
		return wasmEventRef{unknown: joined}, true
	}
	if wasmComponentNameToken(words[0]) {
		if comp, ok := analyze.MaterialComponentByForm(strings.Join(words[1:], " ")); ok {
			return wasmEventRef{kind: comp.ID, name: words[0]}, true
		}
	}
	return wasmEventRef{unknown: joined}, true
}

// wasmEventSuffixes are the event phrases that close a
// source clause, and the event each names — the closed
// set, longest first where one could shadow another.
var wasmEventSuffixes = []struct {
	text  string
	event string
}{
	{" is clicked", "click"},
	{" is pressed", "click"},
	{" is flipped", "change"},
	{" is toggled", "change"},
	{" is switched", "change"},
	{" is changed", "change"},
	{" changes", "change"},
}

// parseWasmEventAction reads an action clause: a
// show/hide/toggle with its target reference, or the
// set-words shape with its words kept exactly as typed
// (the caller's case, the caller's spacing inside the
// quotes). quoted reports whether the set-words text was
// quoted — the folder split may only ride behind quoted
// words, never inside free text it could be part of.
func parseWasmEventAction(seg string) (action string, target wasmEventRef, words string, quoted bool, ok bool) {
	low := strings.ToLower(seg)
	lowFields := strings.Fields(low)
	if len(lowFields) == 0 {
		return "", wasmEventRef{}, "", false, false
	}
	switch lowFields[0] {
	case "show", "hide", "toggle":
		ref, rok := parseWasmEventRefWords(lowFields[1:])
		if !rok {
			return "", wasmEventRef{}, "", false, false
		}
		return lowFields[0], ref, "", false, true
	case "set":
		const probe = "'s words to "
		idx := strings.Index(low, probe)
		if idx < 0 {
			return "", wasmEventRef{}, "", false, false
		}
		ref, rok := parseWasmEventRefWords(strings.Fields(strings.TrimSpace(low[3:idx])))
		if !rok {
			return "", wasmEventRef{}, "", false, false
		}
		raw := strings.TrimSpace(seg[idx+len(probe):])
		if len(raw) >= 2 && (raw[0] == '\'' && raw[len(raw)-1] == '\'' || raw[0] == '"' && raw[len(raw)-1] == '"') {
			words, quoted = raw[1:len(raw)-1], true
		} else {
			words = raw
		}
		if words == "" {
			return "", wasmEventRef{}, "", false, false
		}
		return "setwords", ref, words, quoted, true
	}
	return "", wasmEventRef{}, "", false, false
}

// parseWasmEvent reads the whole event sentence: "when",
// the source clause with its event phrase, a comma, the
// action clause, and an optional "in <folder>" at the
// very end (a folder, or the app's main.go inside one).
func parseWasmEvent(line string) (wasmEventAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(strings.ToLower(s), "when ") {
		return wasmEventAsk{}, false
	}
	rest := s[len("when "):]
	comma := strings.Index(rest, ",")
	if comma < 0 {
		return wasmEventAsk{}, false
	}
	srcSeg, actSeg := rest[:comma], strings.TrimSpace(rest[comma+1:])
	lowSrc := strings.ToLower(srcSeg)
	var out wasmEventAsk
	matched := false
	for _, suf := range wasmEventSuffixes {
		if strings.HasSuffix(lowSrc, suf.text) {
			out.event = suf.event
			ref, rok := parseWasmEventRefWords(strings.Fields(strings.ToLower(srcSeg[:len(srcSeg)-len(suf.text)])))
			if !rok {
				return wasmEventAsk{}, false
			}
			out.source = ref
			matched = true
			break
		}
	}
	if !matched {
		return wasmEventAsk{}, false
	}
	// The folder, when the sentence ends with one: split
	// it off only when the head still parses as a whole
	// action (and, for set-words, only behind quoted words
	// — free words may end in "in <word>" themselves).
	if i := strings.LastIndex(strings.ToLower(actSeg), " in "); i >= 0 {
		tail := strings.TrimSpace(actSeg[i+len(" in "):])
		if tail != "" && !strings.ContainsAny(tail, " \t") {
			if action, target, words, quoted, aok := parseWasmEventAction(strings.TrimSpace(actSeg[:i])); aok && (action != "setwords" || quoted) {
				out.action, out.target, out.words = action, target, words
				out.folder, out.hasFolder = tail, true
				return out, true
			}
		}
	}
	action, target, words, _, aok := parseWasmEventAction(actSeg)
	if !aok {
		return wasmEventAsk{}, false
	}
	out.action, out.target, out.words = action, target, words
	return out, true
}

// tryWasmEvent is the chat-loop entry: it claims event
// sentences and runs the flow on the chat's line reader.
func tryWasmEvent(line string, r *LineReader) bool {
	return runWasmEvent(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runWasmEvent carries one line from sentence to the edit
// spine. It returns true when the line belonged to this
// claim at all.
func runWasmEvent(line string, ask editAsk) bool {
	q, ok := parseWasmEvent(line)
	if !ok {
		return false
	}
	return runWasmEventAsk(q, ask)
}

// runWasmEventAsk is the event flow proper — also the
// clarifying completion's re-entry, with the picked
// component's name filled in.
func runWasmEventAsk(q wasmEventAsk, ask editAsk) bool {
	tag := cTag("gollemer [composer]>")
	if unknown := q.source.unknown; unknown != "" {
		fmt.Printf("%s There's no component called %q in the set an event can use — the components are: %s.\n", tag, unknown, strings.Join(analyze.MaterialComponentForms(), ", "))
		recordBrain("composer", false, "wasm event refused: unknown source component")
		return true
	}
	if unknown := q.target.unknown; unknown != "" {
		fmt.Printf("%s There's no component called %q in the set an event can use — the components are: %s.\n", tag, unknown, strings.Join(analyze.MaterialComponentForms(), ", "))
		recordBrain("composer", false, "wasm event refused: unknown target component")
		return true
	}
	rootRaw, rootAbs, ok := staticAssetsRoot(tag, "wasm event")
	if !ok {
		return true
	}
	folderRel := ""
	if q.hasFolder {
		_, rel, refusal := fenceResolve(rootAbs, q.folder)
		if refusal != "" {
			fmt.Printf("%s Not adding that event: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("composer", false, "wasm event refused: folder outside project")
			return true
		}
		folderRel = rel
		if strings.HasSuffix(strings.ToLower(q.folder), ".go") {
			if path.Base(rel) != "main.go" {
				fmt.Printf("%s Not adding that event: an event belongs to a wasm app — name its folder (\"in jim\"), or the app's main.go inside it.\n", tag)
				recordBrain("composer", false, "wasm event refused: go file target")
				return true
			}
			folderRel = path.Dir(rel)
		}
	} else if workContextLive(rootAbs) && workCtx.wasmFolder != "" && wasmAppFolderOf(rootAbs, workCtx.wasmFolder) {
		// A session path is already project-relative from
		// this session's own records — no fence, but the
		// reading is printed, as everywhere a context
		// target resolves.
		folderRel = workCtx.wasmFolder
		fmt.Printf("Reading the app as the wasm app in %s — the last wasm app this session touched.\n", folderRel)
	} else {
		msg := "the sentence doesn't name the app's folder, and this session hasn't touched a wasm app — name the folder: \"when the button is clicked, show the card in jim/web\""
		if page, _, found := sessionPage(rootAbs); found {
			msg += fmt.Sprintf(". The page from this session, %s, is a plain page, and plain pages stay script-free — events live in wasm apps", page)
		}
		fmt.Printf("%s Not adding that event: %s.\n", tag, msg)
		recordBrain("composer", false, "wasm event refused: no app named")
		return true
	}
	if !wasmAppFolderOf(rootAbs, folderRel) {
		abs := filepath.Join(rootAbs, filepath.FromSlash(folderRel))
		if st, err := os.Stat(abs); err != nil || !st.IsDir() {
			if err == nil {
				fmt.Printf("%s Not adding that event: %s isn't a folder — an event belongs to a wasm app, and an app belongs to a folder.\n", tag, folderRel)
			} else {
				fmt.Printf("%s Not adding that event: there's no folder %s in this project for the app to live in — say \"create a folder %s\" first, then \"add a wasm app in %s\", then ask again.\n", tag, folderRel, folderRel, folderRel)
			}
		} else if folderHasHTML(rootAbs, folderRel) {
			fmt.Printf("%s Not adding that event: the pages in %s are plain HTML, and plain pages stay script-free — nothing in them listens for a click. Events live in a wasm app, where the handler is Go: \"add a wasm app in %s\" makes one, and the same event sentence works there.\n", tag, folderRel, folderRel)
		} else {
			fmt.Printf("%s Not adding that event: there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then the event can go into it.\n", tag, folderRel, folderRel)
		}
		recordBrain("composer", false, "wasm event refused: no app in folder")
		return true
	}
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "wasm event") {
		return true
	}
	// Several mounted components fitting one reference ask
	// one question — the source first, then the target —
	// and the answer re-enters here with the name filled
	// in, every fact re-read from the app as it stands.
	instances := lastAnalyzeProject.WasmAppComponents(folderRel)
	if _, cands := analyze.ResolveWasmComponentRef(instances, q.source.kind, q.source.name); len(cands) > 0 {
		return askWasmEventPick(tag, q, cands, true, ask)
	}
	if _, cands := analyze.ResolveWasmComponentRef(instances, q.target.kind, q.target.name); len(cands) > 0 {
		return askWasmEventPick(tag, q, cands, false, ask)
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:            analyze.EditWasmEvent,
		WasmFolder:      folderRel,
		EventSourceKind: q.source.kind,
		EventSourceName: q.source.name,
		EventName:       q.event,
		EventAction:     q.action,
		EventTargetKind: q.target.kind,
		EventTargetName: q.target.name,
		EventWords:      q.words,
	}, ask)
}

// askWasmEventPick asks the one clarifying question for an
// ambiguous reference: the candidates by name (or, for
// the unnamed one, by the id it carries), numbered. The
// answer completes the original ask here, with the picked
// component's identity filled into that reference.
func askWasmEventPick(tag string, q wasmEventAsk, cands []analyze.WasmComponentInstance, source bool, ask editAsk) bool {
	kind, role := q.target.kind, "act on"
	if source {
		kind, role = q.source.kind, "start from"
	}
	words := analyze.MaterialComponentWords(kind)
	intro := fmt.Sprintf("Not adding that event yet: the app has more than one %s — which one should the event %s?", words, role)
	cs := []clarifyCandidate{}
	for _, in := range cands {
		if in.Name != "" {
			cs = append(cs, clarifyCandidate{Label: in.Name, Desc: "the " + words + " called " + in.Name})
		} else {
			cs = append(cs, clarifyCandidate{Label: in.ID, Desc: "the " + words + " with no name", Words: []string{"unnamed"}})
		}
	}
	askPickQuestion(tag, intro, "Answer with a name or a number", cs, func(idx int, ask2 editAsk) bool {
		chosen := cands[idx]
		name := chosen.Name
		if name == "" {
			name = chosen.ID // the id resolves it, uniquely (the planner's rule)
		}
		q2 := q
		if source {
			q2.source.name = name
		} else {
			q2.target.name = name
		}
		return runWasmEventAsk(q2, ask2)
	})
	recordBrain("composer", false, "wasm event question: which component")
	return true
}

// folderHasHTML reports whether a project folder holds a
// plain page directly — the events boundary's fact: the
// sentence was aimed at pages, which stay script-free.
func folderHasHTML(rootAbs, folderRel string) bool {
	entries, err := os.ReadDir(filepath.Join(rootAbs, filepath.FromSlash(folderRel)))
	if err != nil {
		return false
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".html") {
			return true
		}
	}
	return false
}
